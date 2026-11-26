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


# Segunda rodada: cada etapa foi treinada em uma pasta, com seu proprio historico.
#   (titulo, [pastas em sequencia], [(coluna, rotulo, cor)])
ETAPAS_RODADA2 = (
    ("Oficina de vedacao (escondedores): sala vedada com os dois dentro", ("h4", "h5"),
     (("selada", "sala vedada", "#4a3aa7"),)),
    ("Pegadores no jogo contra a sala vedada", ("s2",),
     (("subiu", "sobe a rampa", "#eda02c"), ("invasao", "entra na sala vedada", "#e34948"),
      ("visto", "escondedor a vista", "#888888"))),
    ("Oficina de defesa (escondedores): travar a rampa e vedar a sala", ("d3", "d4"),
     (("rampa_travada", "rampa travada", "#1baf7b"), ("selada", "sala vedada", "#4a3aa7"))),
    ("Escondedores no jogo contra pegadores que ja conhecem a defesa", ("g1",),
     (("visto", "escondedor a vista", "#888888"), ("selada", "sala vedada a tempo", "#4a3aa7"),
      ("rampa_travada", "rampa travada", "#1baf7b"))),
)


def media_movel(v, n=40):
    out, soma, fila = [], 0.0, []
    for x in v:
        if x == x:
            fila.append(x)
            soma += x
            if len(fila) > n:
                soma -= fila.pop(0)
        out.append(soma / len(fila) if fila else float("nan"))
    return out


def desenhar_rodada2(pasta, saida):
    fig, eixos = plt.subplots(len(ETAPAS_RODADA2), 1, figsize=(11, 2.7 * len(ETAPAS_RODADA2)))
    for eixo, (titulo, pastas, series) in zip(eixos, ETAPAS_RODADA2):
        linhas, cortes = [], []
        for sub in pastas:
            caminho = os.path.join(pasta, sub, "historico.csv")
            if not os.path.exists(caminho):
                continue
            with open(caminho, encoding="utf-8") as f:
                novas = list(csv.DictReader(f))
            if linhas:
                cortes.append(len(linhas))
            linhas += novas
        x = list(range(1, len(linhas) + 1))
        for coluna, rotulo, cor in series:
            v = [float(l[coluna]) if l.get(coluna) not in ("", None, "nan") else float("nan") for l in linhas]
            eixo.plot(x, media_movel(v), color=cor, linewidth=1.8, label=rotulo)
        for c in cortes:
            eixo.axvline(c, color="#999999", linewidth=0.8, linestyle="--")
        eixo.set_ylim(0, 1)
        eixo.set_title(titulo, loc="left", fontsize=11)
        eixo.grid(axis="y", color="#d9d9d6", linewidth=0.6)
        eixo.legend(loc="center right", fontsize=9, frameon=False)
        for lado in ("top", "right"):
            eixo.spines[lado].set_visible(False)
        eixo.set_xlabel("iteracao de treino dentro da etapa")
    fig.suptitle("Segunda rodada: fracao dos episodios (ou do tempo) ao longo do treino, media movel de 40 iteracoes",
                 x=0.01, ha="left", fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    fig.savefig(saida, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rodada2", default=None, metavar="PASTA",
                    help="desenha as etapas da segunda rodada a partir das subpastas com historico.csv")
    ap.add_argument("--csv", default="./checkpoints_final/historico.csv")
    ap.add_argument("--saida", default="./resultados/treino.png")
    ap.add_argument("--titulo", default="Curriculo do pega-pega: metricas ao longo do treino (media movel)")
    args = ap.parse_args()
    if args.rodada2:
        saida = args.saida if args.saida != ap.get_default("saida") else "./resultados/treino_rodada2.png"
        desenhar_rodada2(args.rodada2, saida)
        print(saida)
        return
    linhas = ler(args.csv)
    desenhar(linhas, args.saida, args.titulo)
    print(f"{args.saida}: {len(linhas)} iteracoes, fases {[t[0] for t in trechos_de_fase(linhas)]}")


if __name__ == "__main__":
    main()
