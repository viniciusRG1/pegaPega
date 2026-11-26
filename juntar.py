"""Monta um checkpoint com os escondedores de um arquivo e os pegadores de outro.

As oficinas treinam um time de cada vez, entao da para rodar as duas em paralelo e juntar
os resultados antes das fases de jogo.

Exemplo:
    python juntar.py --escondedores a/checkpoint_recente.pt --pegadores b/checkpoint_recente.pt \
        --saida c/inicio.pt --fase 9 --horizonte 240
"""
import argparse
import os

import torch

from ambiente import CURRICULO
from ppo import alargar_entrada

PARTES = ("agentes", "opts", "scalers", "norm_rew")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--escondedores", required=True)
    ap.add_argument("--pegadores", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--fase", type=int, default=None, help="fase gravada no arquivo (1..%d)" % len(CURRICULO))
    ap.add_argument("--horizonte", type=int, default=None, help="passos do simulador por episodio")
    ap.add_argument("--tam-caixa", type=float, default=None, help="lado da caixa em metros")
    ap.add_argument("--tam-porta", type=int, default=None, help="largura da porta em celulas da grade")
    args = ap.parse_args()

    h = torch.load(args.escondedores, map_location="cpu", weights_only=False)
    s = torch.load(args.pegadores, map_location="cpu", weights_only=False)
    assert list(h["nvec"]) == list(s["nvec"]), "redes incompativeis"
    novo = dict(h)
    for parte in PARTES:
        novo[parte] = {"hider": h[parte]["hider"], "seeker": s[parte]["seeker"]}
    # Se um dos times foi treinado com a observacao mais larga, o outro e adaptado.
    novo["obs_dim"] = max(h["obs_dim"], s["obs_dim"])
    for t in ("hider", "seeker"):
        novo["agentes"][t], _ = alargar_entrada(novo["agentes"][t], novo["obs_dim"])
    amb_s = s.get("ambiente", {})
    novo["estado"] = dict(h["estado"], iteracao=max(h["estado"]["iteracao"], s["estado"]["iteracao"]),
                          iters_fase=0, metricas={})
    novo["ambiente"] = dict(h.get("ambiente", {}))
    for chave in ("obs_portas", "obs_rampa"):
        novo["ambiente"][chave] = bool(novo["ambiente"].get(chave) or amb_s.get(chave))
    if args.horizonte:
        novo["ambiente"]["horizon"] = args.horizonte
    if args.tam_caixa:
        novo["ambiente"]["tam_caixa"] = args.tam_caixa
    if args.tam_porta:
        novo["ambiente"]["tam_porta"] = args.tam_porta
    if args.fase:
        novo["estado"]["fase"] = args.fase - 1
        novo["fase_nome"] = CURRICULO[args.fase - 1]["nome"]
    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    torch.save(novo, args.saida)
    print(f"{args.saida}: escondedores de {args.escondedores} (iteracao {h['estado']['iteracao']}), "
          f"pegadores de {args.pegadores} (iteracao {s['estado']['iteracao']}), fase {novo.get('fase_nome')}, "
          f"ambiente {novo['ambiente']}")


if __name__ == "__main__":
    main()
