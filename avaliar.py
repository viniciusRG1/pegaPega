"""Avalia um checkpoint sem abrir janela: roda N episodios e imprime as metricas medias.

Exemplos:
    python avaliar.py                                   # modelo de demonstracao, 200 episodios
    python avaliar.py --ckpt checkpoints_curriculo/fase_2_perseguir.pt --episodios 100
    python avaliar.py --todos                           # compara todos os checkpoints de fase
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from treinamento import (AMBIENTE, CURRICULO, HIDDEN, TIMES, ActorCriticLSTM, TwoTeamHideAndSeek,
                         fase_do_checkpoint, mascara_acoes)

# Modelo de demonstracao do trabalho: fim da fase 4b_invadir, em que barricada e rampa aparecem juntas.
# O fim do curriculo (jogo livre) fica em checkpoints_final/checkpoint_recente.pt.
MODELO_DEMO = "./checkpoints_final/fase_4b_invadir.pt"

METRICAS = ("visto", "nunca_visto", "portas", "duas_portas", "abrigo", "dentro", "subiu", "invasao",
            "celulas_h", "celulas_s",
            "jogo_h", "jogo_s")
DESCRICAO = {
    "visto": "fracao do tempo (pos-preparacao) em que algum escondedor esta a vista",
    "nunca_visto": "fracao dos episodios em que nenhum escondedor foi visto",
    "portas": "portas bloqueadas por caixa trancada quando a preparacao termina (0 a 2)",
    "duas_portas": "fracao dos episodios com as duas portas bloqueadas ao fim da preparacao",
    "abrigo": "portas bloqueadas com os dois escondedores dentro da sala, ao fim da preparacao (0 a 2)",
    "dentro": "fracao do tempo em que algum pegador esta dentro da sala",
    "subiu": "fracao dos episodios em que um pegador chegou ao alto da rampa (0,5 m)",
    "invasao": "fracao dos episodios em que um pegador entrou na sala com as duas portas fechadas",
    "celulas_h": "celulas diferentes visitadas por escondedor",
    "celulas_s": "celulas diferentes visitadas por pegador",
    "jogo_h": "recompensa do jogo somada no episodio (escondedores)",
    "jogo_s": "recompensa do jogo somada no episodio (pegadores)",
}


def carregar_agentes(caminho, device):
    ck = torch.load(caminho, map_location=device, weights_only=False)
    agentes = {}
    for t in TIMES:
        rede = ActorCriticLSTM(ck["obs_dim"], ck["nvec"], hidden=HIDDEN).to(device)
        rede.load_state_dict(ck["agentes"][t])
        rede.eval()
        agentes[t] = rede
    estado = dict(ck.get("estado", {"iteracao": 0}))
    estado["fase"] = fase_do_checkpoint(ck)
    return agentes, estado, ck.get("ambiente", AMBIENTE)


def indice_fase(estado, fase_forcada=None):
    i = estado.get("fase", 0) if fase_forcada is None else fase_forcada - 1
    return min(max(i, 0), len(CURRICULO) - 1)


def jogar_episodio(env, agentes, mascaras, device, deterministico=False, ao_passo=None):
    """Roda um episodio e devolve o resumo calculado pelo ambiente."""
    n_time = {"hider": env.n_hiders, "seeker": env.n_seekers}
    fatia = {"hider": slice(0, env.n_hiders), "seeker": slice(env.n_hiders, env.n_agents)}
    obs = env.reset()
    lstm = {t: (torch.zeros(1, n_time[t], HIDDEN, device=device),
                torch.zeros(1, n_time[t], HIDDEN, device=device)) for t in TIMES}
    acoes = np.zeros((env.n_agents, env.n_dims), dtype=np.int64)
    fim, info = False, None
    while not fim:
        for t in TIMES:
            o = torch.as_tensor(obs[fatia[t]], dtype=torch.float32, device=device)
            d = torch.zeros(n_time[t], dtype=torch.float32, device=device)
            with torch.no_grad():
                a, _, _, _, lstm[t] = agentes[t].get_action_and_value(
                    o, lstm[t], d, mascaras[t], deterministico=deterministico)
            acoes[fatia[t]] = a.cpu().numpy()
        obs, _, fim, info = env.step(acoes)
        if ao_passo is not None:
            ao_passo(env)
    return info


def avaliar(caminho, episodios=200, fase=None, deterministico=False, semente=0, device=None):
    device = device or torch.device("cpu")
    agentes, estado, cfg_amb = carregar_agentes(caminho, device)
    f = CURRICULO[indice_fase(estado, fase)]
    np.random.seed(semente)
    torch.manual_seed(semente)
    env = TwoTeamHideAndSeek(**cfg_amb, prep_fraction=f["prep_fraction"],
                             sala_fechada=f.get("sala_fechada", False))
    mascaras = {t: mascara_acoes(env.nvec, env.grupos, env.neutro, f["acoes"][t]).to(device) for t in TIMES}
    resumos = []
    try:
        for _ in range(episodios):
            r = jogar_episodio(env, agentes, mascaras, device, deterministico)
            r["nunca_visto"] = float(r["visto"] == 0)
            r["duas_portas"] = float(r["portas"] >= 2)
            resumos.append(r)
    finally:
        env.close()
    saida = {k: float(np.mean([r[k] for r in resumos])) for k in METRICAS}
    saida["erro_padrao"] = {k: float(np.std([r[k] for r in resumos]) / np.sqrt(len(resumos))) for k in METRICAS}
    saida.update(checkpoint=caminho, iteracao=estado.get("iteracao", 0), fase=f["nome"], episodios=episodios)
    return saida


def imprimir(r):
    print(f"\n{r['checkpoint']}  (iteracao {r['iteracao']}, regras da fase {r['fase']}, {r['episodios']} episodios)")
    for k in METRICAS:
        print(f"  {k:11s} {r[k]:+8.3f} +- {r['erro_padrao'][k]:.3f}   {DESCRICAO[k]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=MODELO_DEMO)
    ap.add_argument("--todos", action="store_true", help="avalia todos os .pt da pasta do --ckpt")
    ap.add_argument("--episodios", type=int, default=200)
    ap.add_argument("--fase", type=int, default=None, help="usa as regras desta fase (1..%d)" % len(CURRICULO))
    ap.add_argument("--det", action="store_true", help="acao mais provavel em vez de amostrar")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--json", default=None, help="grava os resultados neste arquivo")
    args = ap.parse_args()

    caminhos = [args.ckpt]
    if args.todos:
        caminhos = sorted(glob.glob(os.path.join(os.path.dirname(args.ckpt), "*.pt")))
    resultados = []
    for c in caminhos:
        r = avaliar(c, args.episodios, args.fase, args.det, args.semente)
        imprimir(r)
        resultados.append(r)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2)


if __name__ == "__main__":
    main()
