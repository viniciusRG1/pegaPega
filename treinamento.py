"""Treino do pega-pega: `python treinamento.py --help`.

O codigo esta dividido em dois modulos, e este arquivo junta os dois:
    ambiente.py  ambiente de dois times, recompensas auxiliares, curriculo e processos paralelos
    ppo.py       rede, PPO, checkpoints e laco de treino

Os processos filhos (um por ambiente paralelo) reimportam este arquivo ao nascer. Eles so
precisam do ambiente, entao o PyTorch e carregado apenas no processo principal.
"""
import multiprocessing as mp

from ambiente import *  # noqa: F401,F403

if mp.current_process().name == "MainProcess":
    from ppo import *  # noqa: F401,F403

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()  # noqa: F405
