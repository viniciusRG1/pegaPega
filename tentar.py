"""Abre uma janela e mostra os agentes de um checkpoint jogando.

Exemplos:
    python tentar.py                                    # modelo de demonstracao
    python tentar.py --ckpt checkpoints_final/fase_3_barricada.pt --episodios 5
    python tentar.py --fase 7 --det                     # regras do jogo completo, acoes deterministicas

Na janela: botao esquerdo gira a camera, botao direito desloca, rolagem aproxima, ESC fecha.
Para gravar video sem abrir janela use o gravar_video.py.
"""
import argparse
import time

import numpy as np
import torch

from avaliar import MODELO_DEMO, carregar_agentes, indice_fase, jogar_episodio
from treinamento import CURRICULO, TIMES, TwoTeamHideAndSeek, mascara_acoes


def rodar_visualizacao(caminho, episodios, pausa, fase_forcada, deterministico):
    device = torch.device("cpu")
    agentes, estado, cfg_amb = carregar_agentes(caminho, device)
    fase = CURRICULO[indice_fase(estado, fase_forcada)]
    print(f"Checkpoint: {caminho}")
    print(f"Iteracao salva: {estado.get('iteracao', 0)} | Regras da fase: {fase['nome']}")

    env = TwoTeamHideAndSeek(**cfg_amb, prep_fraction=fase["prep_fraction"],
                             sala_fechada=fase.get("sala_fechada", False))
    env.renderizar = True
    env.pausa_render = pausa
    mascaras = {t: mascara_acoes(env.nvec, env.grupos, env.neutro, fase["acoes"][t]).to(device) for t in TIMES}
    try:
        for ep in range(1, episodios + 1):
            info = jogar_episodio(env, agentes, mascaras, device, deterministico)
            print(f"Ep {ep}/{episodios} | Portas bloqueadas: {info['portas']:.0f} | "
                  f"A vista: {100 * info['visto']:.0f}% | rH: {info['jogo_h']:+.1f} | rS: {info['jogo_s']:+.1f}")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("Visualizacao encerrada.")
    finally:
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=MODELO_DEMO)
    ap.add_argument("--episodios", type=int, default=10)
    ap.add_argument("--pausa", type=float, default=0.03, help="segundos entre quadros")
    ap.add_argument("--fase", type=int, default=None, help="usa as regras desta fase (1..%d)" % len(CURRICULO))
    ap.add_argument("--det", action="store_true", help="acao mais provavel em vez de amostrar")
    ap.add_argument("--semente", type=int, default=None)
    args = ap.parse_args()
    if args.semente is not None:
        np.random.seed(args.semente)
        torch.manual_seed(args.semente)
    rodar_visualizacao(args.ckpt, args.episodios, args.pausa, args.fase, args.det)
