"""Desenha as curvas de treino a partir do historico.csv gravado pelo treinamento.py.

Exemplos:
    python graficos.py                                        # checkpoints_final/historico.csv
    python graficos.py --csv checkpoints_final/historico.csv --saida resultados/treino.png
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAINEIS = (
    ("portas", "Portas bloqueadas ao fim da preparacao (0 a 2)", "#2a78d6"),
    ("abrigo", "Abrigo: portas bloqueadas com os dois escondedores dentro da sala (0 a 2)", "#4a3aa7"),
    ("visto", "Fracao do tempo com escondedor a vista", "#e34948"),
    ("dentro", "Fracao do tempo com pegador dentro da sala", "#eda02c"),
    ("jogo_h", "Recompensa do jogo por episodio (escondedores)", "#1baf7b"),
)
COR_FASE = ("#f4f4f2", "#e9eef5")
SO_COM_JOGO = ("visto", "dentro", "jogo_h", "jogo_s")


def ler(caminho):
    with open(caminho, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for linha in linhas:
        for k, v in linha.items():
            if k != "fase":
                linha[k] = float(v) if v not in ("", None) else float("nan")
        # Fase sem jogo (pegadores parados o episodio todo): essas metricas nao existem.
        if linha["visto"] == 0 and linha["dentro"] == 0 and linha["jogo_h"] == 0:
            for k in SO_COM_JOGO:
                linha[k] = float("nan")
    return linhas


def trechos_de_fase(linhas):
    trechos = []
    for linha in linhas:
        if not trechos or trechos[-1][0] != linha["fase"]:
            trechos.append([linha["fase"], linha["iteracao"], linha["iteracao"]])
        trechos[-1][2] = linha["iteracao"]
    return trechos


def desenhar(linhas, saida, titulo):
    x = [linha["iteracao"] for linha in linhas]
    trechos = trechos_de_fase(linhas)
    fig, eixos = plt.subplots(len(PAINEIS), 1, figsize=(11, 2.5 * len(PAINEIS)), sharex=True)
    for eixo, (chave, rotulo, cor) in zip(eixos, PAINEIS):
        for i, (nome, ini, fim) in enumerate(trechos):
            eixo.axvspan(ini, fim, color=COR_FASE[i % 2], zorder=0)
        eixo.plot(x, [linha[chave] for linha in linhas], color=cor, linewidth=1.8)
        eixo.set_title(rotulo, loc="left", fontsize=11, pad=20 if eixo is eixos[0] else 6)
        eixo.grid(axis="y", color="#d9d9d6", linewidth=0.6)
        for lado in ("top", "right"):
            eixo.spines[lado].set_visible(False)
    for nome, ini, fim in trechos:
        eixos[0].text((ini + fim) / 2, 1.02, nome, ha="center", va="bottom", fontsize=9, color="#555",
                      transform=eixos[0].get_xaxis_transform())
    eixos[-1].set_xlabel("iteracao de treino")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    fig.savefig(saida, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="./checkpoints_final/historico.csv")
    ap.add_argument("--saida", default="./resultados/treino.png")
    ap.add_argument("--titulo", default="Curriculo do pega-pega: metricas ao longo do treino (media movel)")
    args = ap.parse_args()
    linhas = ler(args.csv)
    desenhar(linhas, args.saida, args.titulo)
    print(f"{args.saida}: {len(linhas)} iteracoes, fases {[t[0] for t in trechos_de_fase(linhas)]}")


if __name__ == "__main__":
    main()
