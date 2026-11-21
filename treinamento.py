import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import argparse
import itertools
import multiprocessing as mp
import numpy as np

# 1. Patch de compatibilidade para NumPy 1.24+
np.bool = bool
np.int = int
np.float = float
np.object = object
np.str = str

import gym
import gym.spaces

# 2. Patch para permitir mutação em gym.spaces.Tuple (exigido pelo DiscretizeActionWrapper)
_orig_tuple_init = gym.spaces.Tuple.__init__
def _patched_tuple_init(self, spaces, *args, **kwargs):
    _orig_tuple_init(self, spaces, *args, **kwargs)
    self.spaces = list(self.spaces)
gym.spaces.Tuple.__init__ = _patched_tuple_init

# 3. Patch para Gym 0.26+ aceitar retornos antigos (obs no reset e 4 valores no step)
if hasattr(gym.Wrapper, "reset"):
    def _legacy_wrapper_reset(self, **kwargs):
        return self.env.reset(**kwargs)
    def _legacy_wrapper_step(self, action):
        return self.env.step(action)
    gym.Wrapper.reset = _legacy_wrapper_reset
    gym.Wrapper.step = _legacy_wrapper_step

if hasattr(gym, "ObservationWrapper"):
    def _legacy_obs_reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict) else res
        return self.observation(obs)
    def _legacy_obs_step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, rew, term, trunc, info = res
            return self.observation(obs), rew, bool(term or trunc), info
        obs, rew, done, info = res
        return self.observation(obs), rew, done, info
    gym.ObservationWrapper.reset = _legacy_obs_reset
    gym.ObservationWrapper.step = _legacy_obs_step

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def carregar_dlls():
    dll_dirs = [
        os.path.expanduser(r"~\.mujoco\mujoco210\bin"),
        r"C:\Users\vinic\.mujoco\mujoco210\bin",
        r"C:\Users\vinic\miniconda3\envs\mujoco_env\Library\bin",
        r"C:\Users\vinic\miniconda3\envs\mujoco_env\Scripts",
        r"C:\Users\vinic\miniconda3\envs\mujoco_env\bin",
    ]
    for d in dll_dirs:
        if os.path.exists(d):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


carregar_dlls()
from mae_envs.envs.hide_and_seek import make_env  # noqa: E402


# =============================================================================
# CONFIGURACAO
# =============================================================================
AMBIENTE = dict(n_hiders=2, n_seekers=2, n_boxes=2, n_ramps=1, frame_skip=4, horizon=120)

TREINO = dict(
    num_envs=12, steps_per_env=120, seq_len=20, num_minibatches=4, update_epochs=4,
    num_iterations=20000, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
    vf_coef=0.5, target_kl=0.03, salvar_cada=10, pasta="./checkpoints_curriculo",
)

TUDO = ("mover", "agarrar", "trancar")

CURRICULO = [
    dict(nome="1_explorar", treinar=("hider", "seeker"), prep_fraction=0.0, ent_coef=0.02,
         acoes={"hider": ("mover",), "seeker": ("mover",)},
         pesos={"hider": {"novidade": 1.0}, "seeker": {"novidade": 1.0}},
         min_iters=30, max_iters=150, meta=("celulas", 30.0)),

    dict(nome="2_perseguir", treinar=("hider", "seeker"), prep_fraction=0.5, ent_coef=0.01,
         acoes={"hider": ("mover",), "seeker": ("mover",)},
         pesos={"hider": {"jogo": 1.0, "distancia": 0.3},
                "seeker": {"jogo": 1.0, "distancia": 0.3, "dentro_sala": 0.3, "novidade": 0.05}},
         min_iters=150, max_iters=800, meta=("visto", 0.65)),

    dict(nome="3_barricada", treinar=("hider",), prep_fraction=0.5, ent_coef=0.01,
         acoes={"hider": TUDO, "seeker": ("mover",)},
         pesos={"hider": {"jogo": 1.0, "ir_caixa": 0.5, "caixa_porta": 1.0, "trancar_porta": 2.0}},
         min_iters=200, max_iters=1500, meta=("portas", 1.5)),

    dict(nome="4_rampa", treinar=("seeker",), prep_fraction=0.5, ent_coef=0.01,
         acoes={"hider": TUDO, "seeker": TUDO},
         pesos={"seeker": {"jogo": 1.0, "ir_rampa": 0.3, "rampa_parede": 1.0, "dentro_sala": 0.5}},
         min_iters=200, max_iters=1500, meta=("visto", 0.35)),

    dict(nome="5_jogo", treinar=("hider", "seeker"), prep_fraction=0.5, ent_coef=0.01,
         acoes={"hider": TUDO, "seeker": TUDO},
         pesos={"hider": {"jogo": 1.0, "trancar_porta": 0.5}, "seeker": {"jogo": 1.0, "dentro_sala": 0.2}},
         min_iters=0, max_iters=None, meta=None),
]

COMPONENTES = ("jogo", "novidade", "distancia", "ir_caixa", "caixa_porta", "trancar_porta",
               "ir_rampa", "rampa_parede", "dentro_sala")
GRUPO_DA_ACAO = {"action_movement": "mover", "action_pull": "agarrar", "action_glueall": "trancar"}
TIMES = ("hider", "seeker")
HIDDEN = 256
TAM_CELULA = 0.5
RAIO_PORTA = 0.45
MAX_PORTAS = 2
ESC_AGENTE = np.array([1 / 3, 1 / 3, 1, 1, 1, .2, .2, .2, .1, 1, 1], np.float32)
ESC_OBJETO = np.array([1 / 3, 1 / 3, 1, 1, 1, .2, .2, .2, .05, 1, 1, 1], np.float32)


# =============================================================================
# AMBIENTE
# =============================================================================
def _polar(v):
    ang = v[..., 3:4]
    return np.concatenate([v[..., :3], np.cos(ang), np.sin(ang), v[..., 4:]], -1)


def _esc(v, esc):
    return v * esc if v.shape[-1] == len(esc) else v


def _bool_done(d):
    if isinstance(d, dict):
        return bool(d.get("__all__", any(d.values())))
    return bool(np.any(d))


def _custo_atribuicao(d):
    nl, nc = d.shape
    if nl == 0 or nc == 0:
        return 0.0
    if nc >= nl:
        return min(sum(d[i, p[i]] for i in range(nl)) for p in itertools.permutations(range(nc), nl))
    return min(sum(d[p[j], j] for j in range(nc)) for p in itertools.permutations(range(nl), nc))


def _atribuir(d):
    na, nc = d.shape
    if nc == 0:
        return np.full(na, -1)
    if nc >= na:
        p = min(itertools.permutations(range(nc), na), key=lambda q: sum(d[i, q[i]] for i in range(na)))
        return np.array(p)
    p = min(itertools.permutations(range(na), nc), key=lambda q: sum(d[q[j], j] for j in range(nc)))
    out = d.argmin(1)
    for j, i in enumerate(p):
        out[i] = j
    return out


def _dist_segmento(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


class TwoTeamHideAndSeek:
    def __init__(self, n_hiders=2, n_seekers=2, n_boxes=2, n_ramps=1, frame_skip=4, horizon=120, prep_fraction=0.4):
        self.n_hiders, self.n_seekers = n_hiders, n_seekers
        self.n_agents = n_hiders + n_seekers
        self.n_boxes, self.n_ramps = n_boxes, n_ramps
        self.frame_skip, self.horizon = frame_skip, horizon
        self.renderizar, self.pausa_render = False, 0.0
        n = self.n_agents
        self.ordem_outros = np.array([np.argsort([(i - j) % n for j in range(1, n)]) for i in range(n)])
        self._env, self.prep_fraction = None, None
        self.configurar(prep_fraction)

        sp = self._env.action_space.spaces
        self.layout, self.nvec, self.grupos, self.neutro = [], [], [], []
        for k in sorted(sp.keys()):
            sub = sp[k].spaces[0]
            nv = [int(v) for v in np.ravel(sub.nvec)] if hasattr(sub, "nvec") else [int(sub.n)]
            self.layout.append((k, len(nv), not hasattr(sub, "nvec")))
            self.nvec += nv
            self.grupos += [GRUPO_DA_ACAO.get(k, k)] * len(nv)
            self.neutro += [(v - 1) // 2 for v in nv]
        self.n_dims = len(self.nvec)
        self.obs_dim = self.reset().shape[1]

    def configurar(self, prep_fraction):
        if self._env is not None and prep_fraction == self.prep_fraction:
            return
        self.close()
        self._env = make_env(n_hiders=self.n_hiders, n_seekers=self.n_seekers, n_boxes=self.n_boxes,
                             n_ramps=self.n_ramps, horizon=self.horizon, prep_fraction=prep_fraction)
        self.prep_fraction = prep_fraction
        self._lock_caixa = None
        w = self._env
        while w is not None:
            if type(w).__name__ == "LockObjWrapper" and getattr(w, "ac_obs_prefix", None) == "":
                self._lock_caixa = w
            w = getattr(w, "env", None)

    def reset(self):
        res = self._env.reset()
        obs = res[0] if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict) else res
        u = self._env.unwrapped
        self._sim = u.sim
        m = self._sim.model
        self._id_ag = np.array([m.body_name2id(f"agent{i}:particle") for i in range(self.n_agents)], dtype=int)
        self._id_cx = np.array([m.body_name2id(f"moveable_box{i}") for i in range(self.n_boxes)], dtype=int)
        self._id_rp = np.array([m.body_name2id(f"ramp{i}:ramp") for i in range(self.n_ramps)], dtype=int)

        paredes = next((md for md in u.modules if type(md).__name__ == "WallScenarios"), None)
        portas = getattr(paredes, "door_obs", None)
        self.portas = np.zeros((0, 2)) if portas is None else np.asarray(portas, dtype=np.float64)[:MAX_PORTAS, :2]
        cel = u.floor_size / u.grid_size
        q = u.metadata.get("quadrant_size", u.grid_size // 2)
        self.sala_x, self.sala_y = (u.grid_size - q + 0.5) * cel, (q + 0.5) * cel
        self.paredes_sala = [(np.array([self.sala_x, 0.0]), np.array([self.sala_x, self.sala_y])),
                             (np.array([self.sala_x, self.sala_y]), np.array([u.floor_size, self.sala_y]))]

        self.em_prep = self.prep_fraction > 0
        self._ler_estado()
        self.visitadas = [{self._celula(p)} for p in self.p_ag]
        self.pot = self._potenciais()
        self.ep = dict(passos_pos=0, visto=0.0, dentro=0.0, portas=None, jogo_h=0.0, jogo_s=0.0)
        return self._obs_agentes(obs)

    def render(self):
        self._env.render()

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = None

    def _montar_acao(self, acoes):
        out, i = {}, 0
        for k, sz, escalar in self.layout:
            a = acoes[:, i:i + sz]
            out[k] = a[:, 0].copy() if escalar else a.copy()
            i += sz
        return out

    def _obs_agentes(self, obs):
        n = self.n_agents
        lin = np.arange(n)[:, None]
        partes = [_esc(_polar(obs["observation_self"]), ESC_AGENTE)]
        outros = _esc(_polar(obs["agent_qpos_qvel"]), ESC_AGENTE)[lin, self.ordem_outros]
        m = obs["mask_aa_obs"][lin, self.ordem_outros].astype(np.float32)
        partes += [outros * m[..., None], m]
        for chave, chave_m in (("box_obs", "mask_ab_obs"), ("ramp_obs", "mask_ar_obs")):
            if chave in obs and chave_m in obs:
                mm = obs[chave_m].astype(np.float32)
                partes += [_esc(obs[chave], ESC_OBJETO) * mm[..., None], mm]
        portas = np.zeros((MAX_PORTAS, 3), np.float32)
        k = len(self.portas)
        portas[:k, :2] = self.portas / 3.0
        portas[:k, 2] = 1.0
        partes.append(np.repeat(portas.reshape(1, -1), n, 0))
        x = np.concatenate([np.asarray(p, np.float32).reshape(n, -1) for p in partes], -1)
        return np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

    def _celula(self, p):
        return int(p[0] // TAM_CELULA), int(p[1] // TAM_CELULA)

    def _dentro_sala(self, p):
        return p[0] > self.sala_x + 0.05 and p[1] < self.sala_y - 0.05

    def _ler_estado(self):
        xp = self._sim.data.body_xpos
        self.p_ag = xp[self._id_ag, :2].copy()
        self.p_cx = xp[self._id_cx, :2].copy() if len(self._id_cx) else np.zeros((0, 2))
        self.p_rp = xp[self._id_rp, :2].copy() if len(self._id_rp) else np.zeros((0, 2))
        lk = self._lock_caixa
        self.cx_trancada_h = np.zeros(len(self.p_cx), bool)
        if lk is not None and len(self.p_cx) and len(np.atleast_1d(lk.obj_locked)) == len(self.p_cx):
            self.cx_trancada_h = np.asarray(lk.obj_locked, bool) & (np.asarray(lk.which_locked) < self.n_hiders)

    def _potenciais(self):
        H, S = slice(0, self.n_hiders), slice(self.n_hiders, self.n_agents)
        pot = {k: np.zeros(self.n_agents) for k in
               ("distancia", "caixa_porta", "trancar_porta", "ir_rampa", "rampa_parede")}
        ph, ps = self.p_ag[H], self.p_ag[S]
        d_hs = np.linalg.norm(ph[:, None] - ps[None], axis=-1)
        pot["distancia"][H] = d_hs.min(1)
        pot["distancia"][S] = -d_hs.min(0)
        livres = np.ones(len(self.p_cx), bool)
        if len(self.p_cx) and len(self.portas):
            d_pc = np.linalg.norm(self.portas[:, None] - self.p_cx[None], axis=-1)
            cobre = (d_pc < RAIO_PORTA) & self.cx_trancada_h[None]
            pot["trancar_porta"][:] = float(np.any(cobre, axis=1).sum())
            pot["caixa_porta"][:] = -_custo_atribuicao(d_pc)
            livres = ~np.any(cobre, axis=0)
        idx_livres = np.flatnonzero(livres)
        alvo = np.full(self.n_hiders, -1)
        if len(idx_livres):
            alvo = idx_livres[_atribuir(np.linalg.norm(ph[:, None] - self.p_cx[idx_livres][None], axis=-1))]
        if len(self.p_rp):
            pot["ir_rampa"][S] = -np.linalg.norm(ps[:, None] - self.p_rp[None], axis=-1).min(1)
            pot["rampa_parede"][:] = -min(_dist_segmento(self.p_rp[0], a, b) for a, b in self.paredes_sala)
        pot["_alvo"] = alvo
        pot["_p_ag"], pot["_p_cx"] = self.p_ag.copy(), self.p_cx.copy()
        return pot

    def step(self, acoes):
        acoes = np.asarray(acoes, dtype=np.int64).reshape(self.n_agents, self.n_dims)
        acao = self._montar_acao(acoes)
        em_prep = self.em_prep
        jogo = np.zeros(self.n_agents)
        fim, info, k = False, {}, 0
        for k in range(1, self.frame_skip + 1):
            res = self._env.step({ch: v.copy() for ch, v in acao.items()})
            obs, rew, done, info = res[0], res[1], res[2], res[-1]
            if len(res) == 5:
                done = _bool_done(done) or _bool_done(res[3])
            jogo += np.asarray(rew, dtype=np.float64).ravel()[:self.n_agents]
            if self.renderizar:
                self._env.render()
                if self.pausa_render:
                    time.sleep(self.pausa_render)
            if _bool_done(done):
                fim = True
                break
        jogo /= max(k, 1)
        self.em_prep = bool(info.get("in_prep_phase", False)) if isinstance(info, dict) else False

        self._ler_estado()
        pot = self._potenciais()
        dp = {ch: pot[ch] - self.pot[ch] for ch in pot if not ch.startswith("_")}
        dp["ir_caixa"] = np.zeros(self.n_agents)
        for j, b in enumerate(self.pot["_alvo"]):
            if b >= 0:
                d0 = np.linalg.norm(self.pot["_p_ag"][j] - self.pot["_p_cx"][b])
                dp["ir_caixa"][j] = d0 - np.linalg.norm(self.p_ag[j] - self.p_cx[b])
        self.pot = pot

        H, S = slice(0, self.n_hiders), slice(self.n_hiders, self.n_agents)
        comp = np.zeros((self.n_agents, len(COMPONENTES)), dtype=np.float32)
        c = {ch: comp[:, i] for i, ch in enumerate(COMPONENTES)}
        novas = np.zeros(self.n_agents, np.float32)
        for i, p in enumerate(self.p_ag):
            cel = self._celula(p)
            if cel not in self.visitadas[i]:
                self.visitadas[i].add(cel)
                novas[i] = 1.0
        c["jogo"][:] = jogo
        c["novidade"][H] = novas[H]
        c["distancia"][H] = dp["distancia"][H]
        c["trancar_porta"][H] = dp["trancar_porta"][H]
        if em_prep:
            c["ir_caixa"][H] = dp["ir_caixa"][H]
            c["caixa_porta"][H] = dp["caixa_porta"][H]
        else:
            c["novidade"][S] = novas[S]
            c["distancia"][S] = dp["distancia"][S]
            c["ir_rampa"][S] = dp["ir_rampa"][S]
            c["rampa_parede"][S] = dp["rampa_parede"][S]
            c["dentro_sala"][S] = [float(self._dentro_sala(p)) for p in self.p_ag[S]]
            self.ep["passos_pos"] += 1
            self.ep["visto"] += float(jogo[S].max() > 0)
            self.ep["dentro"] += float(c["dentro_sala"][S].max() > 0)
            if self.ep["portas"] is None:
                self.ep["portas"] = float(pot["trancar_porta"][0])
        self.ep["jogo_h"] += float(jogo[H].mean())
        self.ep["jogo_s"] += float(jogo[S].mean())

        info_ep = None
        if fim:
            pp = max(self.ep["passos_pos"], 1)
            n_cel = [len(v) for v in self.visitadas]
            info_ep = dict(celulas=float(np.mean(n_cel)), celulas_h=float(np.mean(n_cel[H])),
                           celulas_s=float(np.mean(n_cel[S])), visto=self.ep["visto"] / pp,
                           dentro=self.ep["dentro"] / pp,
                           portas=self.ep["portas"] if self.ep["portas"] is not None else float(pot["trancar_porta"][0]),
                           jogo_h=self.ep["jogo_h"], jogo_s=self.ep["jogo_s"])
        return self._obs_agentes(obs), comp, fim, info_ep


def worker_process(remote, parent_remote, cfg_amb, prep_fraction):
    parent_remote.close()
    carregar_dlls()
    env = TwoTeamHideAndSeek(**cfg_amb, prep_fraction=prep_fraction)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, comp, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, comp, done, info))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "config":
                env.configurar(data)
                remote.send(env.reset())
            elif cmd == "close":
                env.close()
                remote.close()
                break
        except (EOFError, KeyboardInterrupt, BrokenPipeError):
            break


class ParallelEnvs:
    def __init__(self, num_envs, cfg_amb, prep_fraction):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = []
        for wr, r in zip(self.work_remotes, self.remotes):
            p = mp.Process(target=worker_process, args=(wr, r, cfg_amb, prep_fraction))
            p.daemon = True
            p.start()
            self.ps.append(p)
        for wr in self.work_remotes:
            wr.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def configurar(self, prep_fraction):
        for remote in self.remotes:
            remote.send(("config", prep_fraction))
        return np.stack([remote.recv() for remote in self.remotes])

    def step(self, acoes):
        for remote, a in zip(self.remotes, acoes):
            remote.send(("step", a))
        results = [remote.recv() for remote in self.remotes]
        obs, comp, dones, infos = zip(*results)
        return (np.stack(obs), np.stack(comp), np.array(dones, dtype=np.float32),
                [i for i in infos if i is not None])

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()


# =============================================================================
# REDE
# =============================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mascara_acoes(nvec, grupos, neutro, liberadas):
    m = torch.zeros(len(nvec), max(nvec))
    for d, (nv, g, ne) in enumerate(zip(nvec, grupos, neutro)):
        m[d, nv:] = -1e8
        if g not in liberadas:
            m[d, :] = -1e8
            m[d, ne] = 0.0
    return m


class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_dim, nvec, hidden=HIDDEN):
        super().__init__()
        self.nvec = [int(v) for v in nvec]
        self.n_dims, self.max_bins = len(self.nvec), max(self.nvec)
        self.fc = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.LayerNorm(hidden),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(hidden, hidden)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.ln_lstm = nn.LayerNorm(hidden)
        self.actor = layer_init(nn.Linear(hidden, self.n_dims * self.max_bins), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward_lstm(self, x, lstm_state, done):
        hidden = self.fc(x)
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        outputs = []
        for h_step, d_step in zip(hidden, done):
            mask = (1.0 - d_step).view(1, -1, 1)
            lstm_state = (mask * lstm_state[0], mask * lstm_state[1])
            out, lstm_state = self.lstm(h_step.unsqueeze(0), lstm_state)
            outputs.append(out)
        out = torch.cat(outputs, dim=0).flatten(0, 1)
        return self.ln_lstm(out), lstm_state

    def get_value(self, x, lstm_state, done):
        out, _ = self.forward_lstm(x, lstm_state, done)
        return self.critic(out).float()

    def get_action_and_value(self, x, lstm_state, done, mascara, action=None, deterministico=False):
        out, next_lstm_state = self.forward_lstm(x, lstm_state, done)
        logits = self.actor(out).float().view(-1, self.n_dims, self.max_bins) + mascara
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(-1) if deterministico else dist.sample()
        return (action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1),
                self.critic(out).float(), next_lstm_state)


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64).ravel()
        b_mean, b_var, b_n = x.mean(), x.var(), x.size
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + (delta ** 2) * self.count * b_n / tot) / tot
        self.count = tot

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


# =============================================================================
# PPO
# =============================================================================
def compute_gae(rew, val, done, next_v, next_done, gamma, lam):
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    lastg = np.zeros(rew.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        nxt_nonterm = 1.0 - (next_done if t == T - 1 else done[t + 1])
        nxt_val = next_v if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nxt_val * nxt_nonterm - val[t]
        adv[t] = lastg = delta + gamma * lam * nxt_nonterm * lastg
    return adv, adv + val


def train_ppo_tbptt_amp(agent, optimizer, scaler, mascara, b, cfg, ent_coef, usar_amp, device):
    T, N = b["logp"].shape
    L = cfg["seq_len"]
    n_chunks = T // L
    total_seqs = n_chunks * N
    seqs_mb = max(1, total_seqs // cfg["num_minibatches"])
    clip = cfg["clip_coef"]

    adv = b["adv"]
    adv = torch.clamp((adv - adv.mean()) / (adv.std() + 1e-8), -5.0, 5.0)

    def to_seq(x):
        sh = x.shape[2:]
        return x.reshape(n_chunks, L, N, *sh).transpose(0, 1).reshape(L, total_seqs, *sh)

    obs_s, act_s, logp_s, adv_s, ret_s, val_s, done_s = (
        to_seq(x) for x in (b["obs"], b["act"], b["logp"], adv, b["ret"], b["val"], b["done"]))
    init_h = b["h"].reshape(1, total_seqs, -1)
    init_c = b["c"].reshape(1, total_seqs, -1)

    idx = np.arange(total_seqs)
    kls, ents = [], []
    for _ in range(cfg["update_epochs"]):
        np.random.shuffle(idx)
        kl_epoca = []
        for start in range(0, total_seqs, seqs_mb):
            mb = idx[start:start + seqs_mb]
            with torch.amp.autocast(device.type, enabled=usar_amp):
                _, newlogp, entropy, newv, _ = agent.get_action_and_value(
                    obs_s[:, mb].reshape(-1, obs_s.shape[-1]),
                    (init_h[:, mb].contiguous(), init_c[:, mb].contiguous()),
                    done_s[:, mb].reshape(-1), mascara,
                    act_s[:, mb].reshape(-1, act_s.shape[-1]))
                m_adv = adv_s[:, mb].reshape(-1)
                m_ret = ret_s[:, mb].reshape(-1)
                m_val = val_s[:, mb].reshape(-1)
                log_ratio = torch.clamp(newlogp - logp_s[:, mb].reshape(-1), -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                pg_loss = torch.max(-m_adv * ratio, -m_adv * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)).mean()
                newv = newv.squeeze(-1)
                v_unc = (newv - m_ret) ** 2
                v_clip = (m_val + torch.clamp(newv - m_val, -clip, clip) - m_ret) ** 2
                v_loss = 0.5 * torch.max(v_unc, v_clip).mean()
                loss = pg_loss - ent_coef * entropy.mean() + cfg["vf_coef"] * v_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                kl_epoca.append(((ratio - 1) - log_ratio).mean().item())
                ents.append(entropy.mean().item())
        kls += kl_epoca
        if cfg["target_kl"] and np.mean(kl_epoca) > cfg["target_kl"]:
            break
    return float(np.mean(kls)), float(np.mean(ents))


# =============================================================================
# CURRICULO / CHECKPOINT
# =============================================================================
def deve_avancar(i_fase, estado):
    if i_fase >= len(CURRICULO) - 1:
        return False
    fase, n = CURRICULO[i_fase], estado["iters_fase"]
    if fase.get("max_iters") is not None and n >= fase["max_iters"]:
        return True
    if fase.get("meta") and n >= fase.get("min_iters", 0):
        nome, limiar = fase["meta"]
        v = estado["metricas"].get(nome)
        return v is not None and v >= limiar
    return False


def salvar(caminho, agentes, opts, scalers, norm_rew, estado, obs_dim, nvec):
    torch.save({
        "agentes": {t: agentes[t].state_dict() for t in TIMES},
        "opts": {t: opts[t].state_dict() for t in TIMES},
        "scalers": {t: scalers[t].state_dict() for t in TIMES},
        "norm_rew": {t: norm_rew[t].state_dict() for t in TIMES},
        "estado": estado, "obs_dim": obs_dim, "nvec": list(nvec), "ambiente": AMBIENTE,
        "fase_nome": CURRICULO[estado["fase"]]["nome"],
    }, caminho + ".tmp")
    for tentativa in range(5):
        try:
            os.replace(caminho + ".tmp", caminho)
            return
        except PermissionError:
            time.sleep(0.5)
    print(f"Aviso: nao consegui salvar {caminho} (arquivo em uso).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", type=int, default=None, help="forca a fase (1..%d)" % len(CURRICULO))
    ap.add_argument("--envs", type=int, default=None, help="numero de ambientes em paralelo")
    args = ap.parse_args()

    cfg = dict(TREINO)
    if args.envs:
        cfg["num_envs"] = args.envs
    num_envs, T, L = cfg["num_envs"], cfg["steps_per_env"], cfg["seq_len"]
    assert T % L == 0, "steps_per_env precisa ser multiplo de seq_len"
    n_chunks = T // L

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usar_amp = device.type == "cuda"
    print(f"Dispositivo em uso: {device}")

    dummy = TwoTeamHideAndSeek(**AMBIENTE, prep_fraction=CURRICULO[0]["prep_fraction"])
    obs_dim, nvec, grupos, neutro, n_dims = dummy.obs_dim, dummy.nvec, dummy.grupos, dummy.neutro, dummy.n_dims
    n_h, n_s = dummy.n_hiders, dummy.n_seekers
    dummy.close()
    n_time = {"hider": n_h, "seeker": n_s}
    fatia = {"hider": slice(0, n_h), "seeker": slice(n_h, n_h + n_s)}

    agentes = {t: ActorCriticLSTM(obs_dim, nvec).to(device) for t in TIMES}
    opts = {t: optim.Adam(agentes[t].parameters(), lr=cfg["lr"], eps=1e-5) for t in TIMES}
    scalers = {t: torch.amp.GradScaler(device.type, enabled=usar_amp) for t in TIMES}
    norm_rew = {t: RunningMeanStd() for t in TIMES}
    estado = dict(iteracao=0, fase=0, iters_fase=0, metricas={})

    os.makedirs(cfg["pasta"], exist_ok=True)
    ck_recente = os.path.join(cfg["pasta"], "checkpoint_recente.pt")
    if os.path.exists(ck_recente):
        try:
            ck = torch.load(ck_recente, map_location=device, weights_only=False)
            assert ck["obs_dim"] == obs_dim and list(ck["nvec"]) == list(nvec)
            for t in TIMES:
                agentes[t].load_state_dict(ck["agentes"][t])
                opts[t].load_state_dict(ck["opts"][t])
                scalers[t].load_state_dict(ck["scalers"][t])
                norm_rew[t].load_state_dict(ck["norm_rew"][t])
            estado = ck["estado"]
            estado["fase"] = min(estado["fase"], len(CURRICULO) - 1)
            print(f"Treino retomado na iteracao {estado['iteracao'] + 1} (fase {CURRICULO[estado['fase']]['nome']}).")
        except Exception as e:
            print(f"Checkpoint incompativel ignorado ({type(e).__name__}). Iniciando treino limpo.")
    if args.fase is not None:
        estado.update(fase=min(max(args.fase, 1), len(CURRICULO)) - 1, iters_fase=0, metricas={})
        norm_rew = {t: RunningMeanStd() for t in TIMES}
        print(f"Fase forcada: {CURRICULO[estado['fase']]['nome']}")

    i_fase = estado["fase"]
    envs = ParallelEnvs(num_envs, AMBIENTE, CURRICULO[i_fase]["prep_fraction"])
    print(f"===== FASE {CURRICULO[i_fase]['nome']} =====")

    def novos_estados_lstm():
        return {t: (torch.zeros(1, num_envs * n_time[t], HIDDEN, device=device),
                    torch.zeros(1, num_envs * n_time[t], HIDDEN, device=device)) for t in TIMES}

    try:
        obs = envs.reset()
        dones = np.ones(num_envs, dtype=np.float32)
        lstm = novos_estados_lstm()
        ret_run = {t: np.zeros(num_envs * n_time[t]) for t in TIMES}

        while estado["iteracao"] < cfg["num_iterations"]:
            t0 = time.time()
            if estado["fase"] != i_fase:
                i_fase = estado["fase"]
                obs = envs.configurar(CURRICULO[i_fase]["prep_fraction"])
                dones = np.ones(num_envs, dtype=np.float32)
                lstm = novos_estados_lstm()
                norm_rew = {t: RunningMeanStd() for t in TIMES}
                ret_run = {t: np.zeros(num_envs * n_time[t]) for t in TIMES}
                print(f"===== FASE {CURRICULO[i_fase]['nome']} =====")
            fase = CURRICULO[i_fase]
            mascaras = {t: mascara_acoes(nvec, grupos, neutro, fase["acoes"][t]).to(device) for t in TIMES}
            pesos = {t: np.array([fase["pesos"].get(t, {}).get(c, 0.0) for c in COMPONENTES], np.float32)
                     for t in TIMES}

            buf = {}
            for t in TIMES:
                N = num_envs * n_time[t]
                buf[t] = dict(obs=np.zeros((T, N, obs_dim), np.float32), done=np.zeros((T, N), np.float32),
                              fim=np.zeros((T, N), np.float32), act=np.zeros((T, N, n_dims), np.int64),
                              logp=np.zeros((T, N), np.float32), val=np.zeros((T, N), np.float32),
                              rew=np.zeros((T, N), np.float32), h=np.zeros((n_chunks, N, HIDDEN), np.float32),
                              c=np.zeros((n_chunks, N, HIDDEN), np.float32))
            episodios = []
            acoes = np.zeros((num_envs, n_h + n_s, n_dims), np.int64)

            for step in range(T):
                for t in TIMES:
                    B, n_t = buf[t], n_time[t]
                    o = obs[:, fatia[t]].reshape(-1, obs_dim)
                    d = np.repeat(dones, n_t)
                    B["obs"][step], B["done"][step] = o, d
                    if step % L == 0:
                        B["h"][step // L] = lstm[t][0][0].cpu().numpy()
                        B["c"][step // L] = lstm[t][1][0].cpu().numpy()
                    with torch.no_grad():
                        a, lp, _, v, lstm[t] = agentes[t].get_action_and_value(
                            torch.as_tensor(o, device=device), lstm[t], torch.as_tensor(d, device=device),
                            mascaras[t])
                    a = a.cpu().numpy()
                    B["act"][step], B["logp"][step], B["val"][step] = a, lp.cpu().numpy(), v.squeeze(-1).cpu().numpy()
                    acoes[:, fatia[t]] = a.reshape(num_envs, n_t, n_dims)

                obs, comp, dones, infos = envs.step(acoes)
                episodios += infos
                for t in TIMES:
                    buf[t]["rew"][step] = (comp[:, fatia[t]] @ pesos[t]).reshape(-1)
                    buf[t]["fim"][step] = np.repeat(dones, n_time[t])

            r_medio = {t: float(buf[t]["rew"].mean()) for t in TIMES}
            kl_txt = []
            for t in fase["treinar"]:
                B = buf[t]
                for step in range(T):
                    ret_run[t] = ret_run[t] * cfg["gamma"] + B["rew"][step]
                    norm_rew[t].update(ret_run[t])
                    ret_run[t] *= (1.0 - B["fim"][step])
                rew_n = np.clip(B["rew"] / np.sqrt(norm_rew[t].var + 1e-8), -10, 10).astype(np.float32)

                o = torch.as_tensor(obs[:, fatia[t]].reshape(-1, obs_dim), device=device)
                d_next = np.repeat(dones, n_time[t])
                with torch.no_grad():
                    next_v = agentes[t].get_value(o, lstm[t], torch.as_tensor(d_next, device=device))
                adv, ret = compute_gae(rew_n, B["val"], B["done"], next_v.squeeze(-1).cpu().numpy(), d_next,
                                       cfg["gamma"], cfg["gae_lambda"])
                bt = {k: torch.as_tensor(v, device=device) for k, v in
                      dict(obs=B["obs"], act=B["act"], logp=B["logp"], adv=adv, ret=ret, val=B["val"],
                           done=B["done"], h=B["h"], c=B["c"]).items()}
                kl, ent = train_ppo_tbptt_amp(agentes[t], opts[t], scalers[t], mascaras[t], bt, cfg,
                                              fase.get("ent_coef", 0.01), usar_amp, device)
                kl_txt.append(f"{t[0].upper()} kl {kl:.3f} ent {ent:.2f}")

            if episodios:
                for k in episodios[0]:
                    novo = float(np.mean([e[k] for e in episodios]))
                    ant = estado["metricas"].get(k)
                    estado["metricas"][k] = novo if ant is None else 0.9 * ant + 0.1 * novo

            estado["iteracao"] += 1
            estado["iters_fase"] += 1
            m = estado["metricas"]
            max_it = fase.get("max_iters") or "-"
            print(f"Iter {estado['iteracao']} | {fase['nome']} {estado['iters_fase']}/{max_it} | "
                  f"rH {r_medio['hider']:+.2f} rS {r_medio['seeker']:+.2f} | cel {m.get('celulas', 0):.1f} "
                  f"visto {m.get('visto', 0):.2f} dentro {m.get('dentro', 0):.2f} portas {m.get('portas', 0):.2f} | "
                  f"{' | '.join(kl_txt)} | {time.time() - t0:.1f}s")

            if deve_avancar(i_fase, estado):
                salvar(os.path.join(cfg["pasta"], f"fase_{fase['nome']}.pt"), agentes, opts, scalers, norm_rew,
                       estado, obs_dim, nvec)
                print(f"Fase {fase['nome']} concluida em {estado['iters_fase']} iteracoes.")
                estado.update(fase=i_fase + 1, iters_fase=0, metricas={})

            if estado["iteracao"] % cfg["salvar_cada"] == 0 or estado["fase"] != i_fase:
                salvar(ck_recente, agentes, opts, scalers, norm_rew, estado, obs_dim, nvec)
            if estado["iteracao"] % 250 == 0:
                salvar(os.path.join(cfg["pasta"], f"ckpt_iter_{estado['iteracao']}.pt"), agentes, opts, scalers,
                       norm_rew, estado, obs_dim, nvec)
    except KeyboardInterrupt:
        print("Interrompido pelo usuario.")
    finally:
        if estado["iteracao"] > 0:
            salvar(ck_recente, agentes, opts, scalers, norm_rew, estado, obs_dim, nvec)
        print("Fechando ambientes em paralelo...")
        envs.close()
        print("Treino finalizado com seguranca.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()