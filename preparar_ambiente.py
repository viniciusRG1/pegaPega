"""Prepara o processo para rodar o ambiente de esconde-esconde da OpenAI.

Basta `import preparar_ambiente` antes de importar `mae_envs`. O modulo:

1. poe no `sys.path` as copias locais de `mujoco-worldgen` e `multi-agent-emergence-environments`;
2. usa o `mujoco_py` de verdade se ele estiver instalado e, se nao estiver, cai na camada de
   compatibilidade `compat/mujoco_py`, que roda em cima do pacote oficial `mujoco`;
3. aplica os remendos de que o codigo de 2019 precisa para rodar com NumPy e Gym atuais.

Tudo aqui e idempotente: importar duas vezes (ou em processos filhos) nao tem efeito extra.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RAIZ = os.path.dirname(os.path.abspath(__file__))


def _carregar_dlls_windows():
    """No Windows o mujoco_py precisa achar as DLLs do MuJoCo 2.1 e do conda."""
    if os.name != "nt":
        return
    casa = os.path.expanduser("~")
    prefixos = [sys.prefix, os.environ.get("CONDA_PREFIX", "")]
    pastas = [os.path.join(casa, ".mujoco", "mujoco210", "bin")]
    for p in prefixos:
        if p:
            pastas += [os.path.join(p, "Library", "bin"), os.path.join(p, "Scripts"), os.path.join(p, "bin")]
    for d in pastas:
        if os.path.isdir(d):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _caminhos():
    for pasta in ("mujoco-worldgen", "multi-agent-emergence-environments"):
        caminho = os.path.join(RAIZ, pasta)
        if caminho not in sys.path:
            sys.path.insert(0, caminho)


def _escolher_mujoco():
    """Devolve 'mujoco_py' (original) ou 'compat' (pacote oficial `mujoco`)."""
    if os.environ.get("PEGAPEGA_MUJOCO", "").lower() != "compat":
        try:
            import mujoco_py  # noqa: F401
            return "compat" if str(getattr(mujoco_py, "__version__", "")).startswith("compat") else "mujoco_py"
        except Exception:
            pass
    compat = os.path.join(RAIZ, "compat")
    if compat not in sys.path:
        sys.path.insert(0, compat)
    sys.modules.pop("mujoco_py", None)
    import mujoco_py  # noqa: F401
    return "compat"


def _remendar_numpy():
    import numpy as np
    for nome, tipo in (("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str)):
        if nome not in np.__dict__:
            setattr(np, nome, tipo)


def _remendar_gym():
    import gym
    import gym.spaces

    if getattr(gym, "_pegapega_remendado", False):
        return
    gym._pegapega_remendado = True

    # O DiscretizeActionWrapper troca itens de gym.spaces.Tuple, que hoje e imutavel.
    init_original = gym.spaces.Tuple.__init__

    def init_tuple(self, spaces, *args, **kwargs):
        init_original(self, spaces, *args, **kwargs)
        self.spaces = list(self.spaces)

    gym.spaces.Tuple.__init__ = init_tuple

    # Gym 0.26 mudou reset() -> (obs, info) e step() -> 5 valores; o mae_envs usa a API antiga.
    def reset_wrapper(self, **kwargs):
        return self.env.reset(**kwargs)

    def step_wrapper(self, action):
        return self.env.step(action)

    gym.Wrapper.reset = reset_wrapper
    gym.Wrapper.step = step_wrapper

    def reset_obs(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict) else res
        return self.observation(obs)

    def step_obs(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, rew, term, trunc, info = res
            return self.observation(obs), rew, bool(term or trunc), info
        obs, rew, done, info = res
        return self.observation(obs), rew, done, info

    gym.ObservationWrapper.reset = reset_obs
    gym.ObservationWrapper.step = step_obs


_carregar_dlls_windows()
_caminhos()
_remendar_numpy()
_remendar_gym()
BACKEND_MUJOCO = _escolher_mujoco()
