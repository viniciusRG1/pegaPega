import os
import time
import argparse
import numpy as np
import torch

from treinamento import (
    AMBIENTE,
    CURRICULO,
    TIMES,
    HIDDEN,
    TwoTeamHideAndSeek,
    ActorCriticLSTM,
    mascara_acoes,
)


def carregarAgentes(caminhoCheckpoint, device):
    if not os.path.exists(caminhoCheckpoint):
        raise FileNotFoundError(f"Checkpoint nao encontrado: {caminhoCheckpoint}")

    ck = torch.load(caminhoCheckpoint, map_location=device, weights_only=False)
    obsDim = ck["obs_dim"]
    nvec = ck["nvec"]
    estado = ck.get("estado", {"iteracao": 0, "fase": 0})
    cfgAmbiente = ck.get("ambiente", AMBIENTE)

    agentes = {}
    for t in TIMES:
        rede = ActorCriticLSTM(obsDim, nvec, hidden=HIDDEN).to(device)
        rede.load_state_dict(ck["agentes"][t])
        rede.eval()
        agentes[t] = rede

    return agentes, estado, cfgAmbiente


def rodarVisualizacao(caminhoCheckpoint, numEpisodios, pausaRender, faseForcada, deterministico):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agentes, estado, cfgAmbiente = carregarAgentes(caminhoCheckpoint, device)

    indiceFase = estado.get("fase", 0)
    if faseForcada is not None:
        indiceFase = min(max(faseForcada, 1), len(CURRICULO)) - 1
    indiceFase = min(indiceFase, len(CURRICULO) - 1)
    faseAtual = CURRICULO[indiceFase]

    print(f"Checkpoint: {caminhoCheckpoint}")
    print(f"Iteracao salva: {estado.get('iteracao', 0)} | Fase: {faseAtual['nome']}")

    env = TwoTeamHideAndSeek(**cfgAmbiente, prep_fraction=faseAtual["prep_fraction"])
    env.renderizar = True
    env.pausa_render = pausaRender

    mascaras = {
        t: mascara_acoes(env.nvec, env.grupos, env.neutro, faseAtual["acoes"][t]).to(device)
        for t in TIMES
    }

    nTime = {"hider": env.n_hiders, "seeker": env.n_seekers}
    fatia = {"hider": slice(0, env.n_hiders), "seeker": slice(env.n_hiders, env.n_agents)}

    try:
        for ep in range(1, numEpisodios + 1):
            obs = env.reset()
            done = False
            lstm = {
                t: (
                    torch.zeros(1, nTime[t], HIDDEN, device=device),
                    torch.zeros(1, nTime[t], HIDDEN, device=device),
                )
                for t in TIMES
            }
            acoes = np.zeros((env.n_agents, env.n_dims), dtype=np.int64)
            infoEp = None

            while not done:
                for t in TIMES:
                    o = torch.as_tensor(obs[fatia[t]], dtype=torch.float32, device=device)
                    d = torch.zeros(nTime[t], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        a, _, _, _, lstm[t] = agentes[t].get_action_and_value(
                            o, lstm[t], d, mascaras[t], deterministico=deterministico
                        )
                    acoes[fatia[t]] = a.cpu().numpy()

                obs, _, done, infoEp = env.step(acoes)

            if infoEp:
                print(
                    f"Ep {ep}/{numEpisodios} | Portas: {infoEp['portas']:.1f} | "
                    f"Visto: {infoEp['visto']:.2f} | rH: {infoEp['jogo_h']:+.2f} | rS: {infoEp['jogo_s']:+.2f}"
                )
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("Visualizacao encerrada.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./checkpoints_curriculo/checkpoint_recente.pt")
    parser.add_argument("--episodios", type=int, default=10)
    parser.add_argument("--pausa", type=float, default=0.03)
    parser.add_argument("--fase", type=int, default=None)
    parser.add_argument("--det", action="store_true")
    args = parser.parse_args()

    rodarVisualizacao(args.ckpt, args.episodios, args.pausa, args.fase, args.det)