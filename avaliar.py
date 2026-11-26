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

from treinamento import (AMBIENTE, CURRICULO, HIDDEN, TIMES, ActorCriticLSTM, ambiente_da_fase,
                         fase_do_checkpoint, mascara_acoes)

# Modelo de demonstracao do trabalho: ultima etapa da segunda rodada (escondedores que travam a rampa
# e vedam a sala contra pegadores que sabem usar a rampa). As etapas anteriores estao na mesma pasta.
MODELO_DEMO = "./checkpoints_rodada2/etapa_c_defesa.pt"

METRICAS = ("visto", "nunca_visto", "portas", "duas_portas", "abrigo", "vedadas", "selada", "rampa_travada",
            "dentro", "subiu", "invasao",
            "celulas_h", "celulas_s",
            "jogo_h", "jogo_s")
DESCRICAO = {
    "visto": "fracao do tempo (pos-preparacao) em que algum escondedor esta a vista",
    "nunca_visto": "fracao dos episodios em que nenhum escondedor foi visto",
    "portas": "portas bloqueadas por caixa trancada quando a preparacao termina (0 a 2)",
    "duas_portas": "fracao dos episodios com as duas portas bloqueadas ao fim da preparacao",
    "abrigo": "portas bloqueadas com os dois escondedores dentro da sala, ao fim da preparacao (0 a 2)",
    "vedadas": "portas vedadas de verdade (caixa trancada e centrada no vao) ao fim da preparacao (0 a 2)",
    "selada": "fracao dos episodios com as duas portas vedadas e os dois escondedores dentro, ao fim da preparacao",
    "rampa_travada": "fracao dos episodios com a rampa trancada pelos escondedores fora de uso, ao fim da preparacao",
    "dentro": "fracao do tempo em que algum pegador esta dentro da sala",
    "subiu": "fracao dos episodios em que um pegador chegou ao alto da rampa (0,5 m)",
    "invasao": "fracao dos episodios em que um pegador entrou na sala com as duas portas vedadas",
    "celulas_h": "celulas diferentes visitadas por escondedor",
    "celulas_s": "celulas diferentes visitadas por pegador",
    "jogo_h": "recompensa do jogo somada no episodio (escondedores)",
    "jogo_s": "recompensa do jogo somada no episodio (pegadores)",
}


def carregar_agentes(caminho, device):
    ck = torch.load(caminho, map_location=device, weights_only=False)
    agentes = {}
    for t in TIMES:
        entradas = ck["agentes"][t]["fc.0.weight"].shape[1]
        rede = ActorCriticLSTM(entradas, ck["nvec"], hidden=HIDDEN).to(device)
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


def avaliar(caminho, episodios=200, fase=None, deterministico=False, semente=0, device=None, horizonte=None):
    device = device or torch.device("cpu")
    agentes, estado, cfg_amb = carregar_agentes(caminho, device)
    f = CURRICULO[indice_fase(estado, fase)]
    if horizonte:
        cfg_amb, f = dict(cfg_amb, horizon=horizonte), dict(f, horizonte=None)
    np.random.seed(semente)
    torch.manual_seed(semente)
    env = ambiente_da_fase(cfg_amb, f)
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
    saida.update(checkpoint=caminho, iteracao=estado.get("iteracao", 0), fase=f["nome"], episodios=episodios,
                 horizonte=env.horizon)
    return saida


def imprimir(r):
    print(f"\n{r['checkpoint']}  (iteracao {r['iteracao']}, regras da fase {r['fase']}, "
          f"{r['horizonte']} passos por episodio, {r['episodios']} episodios)")
    for k in METRICAS:
        print(f"  {k:13s} {r[k]:+8.3f} +- {r['erro_padrao'][k]:.3f}   {DESCRICAO[k]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=MODELO_DEMO)
    ap.add_argument("--todos", action="store_true", help="avalia todos os .pt da pasta do --ckpt")
    ap.add_argument("--episodios", type=int, default=200)
    ap.add_argument("--fase", type=int, default=None, help="usa as regras desta fase (1..%d)" % len(CURRICULO))
    ap.add_argument("--det", action="store_true", help="acao mais provavel em vez de amostrar")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--horizonte", type=int, default=None, help="passos do simulador por episodio")
    ap.add_argument("--json", default=None, help="grava os resultados neste arquivo")
    args = ap.parse_args()

    caminhos = [args.ckpt]
    if args.todos:
        caminhos = sorted(glob.glob(os.path.join(os.path.dirname(args.ckpt), "*.pt")))
    resultados = []
    for c in caminhos:
        r = avaliar(c, args.episodios, args.fase, args.det, args.semente, horizonte=args.horizonte)
        imprimir(r)
        resultados.append(r)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2)


if __name__ == "__main__":
    main()
