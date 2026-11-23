"""Ambiente de dois times, recompensas auxiliares, curriculo e processos paralelos.

Este modulo nao importa o PyTorch de proposito: cada ambiente paralelo roda em um processo
proprio, e sem o PyTorch cada um ocupa bem menos memoria.
"""
import os
import time
import itertools
import multiprocessing as mp

import preparar_ambiente  # noqa: F401  (caminhos, remendos de NumPy/Gym e escolha do MuJoCo)

import numpy as np

from mae_envs.envs.hide_and_seek import make_env  # noqa: E402



# =============================================================================
# CONFIGURACAO
# =============================================================================
AMBIENTE = dict(n_hiders=2, n_seekers=2, n_boxes=2, n_ramps=1, frame_skip=4, horizon=120)

TREINO = dict(
    num_envs=12, steps_per_env=120, seq_len=20, num_minibatches=4, update_epochs=6,
    num_iterations=20000, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
    vf_coef=0.5, target_kl=0.03, salvar_cada=10, pasta="./checkpoints_curriculo",
)

TUDO = ("mover", "agarrar", "trancar")

# Cada fase ensina uma habilidade e congela o time que nao esta aprendendo.
#   treinar        times que recebem atualizacao de PPO na fase
#   prep_fraction  fracao do episodio em que os pegadores ficam parados (1.0 = episodio inteiro)
#   acoes          grupos de acao liberados por time (os outros ficam presos na acao neutra)
#   pesos          peso de cada componente de recompensa (ver COMPONENTES)
#   meta           (metrica, limiar): a fase termina quando a media movel passa do limiar,
#                  respeitando min_iters, ou ao chegar em max_iters
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

    # Oficina de barricada: os pegadores ficam parados o episodio todo e so contam as
    # recompensas auxiliares, para os escondedores aprenderem a levar caixas ate as portas.
    dict(nome="3_barricada", treinar=("hider",), prep_fraction=1.0, ent_coef=0.01,
         acoes={"hider": TUDO, "seeker": ("mover",)},
         pesos={"hider": {"ir_caixa": 1.0, "encarar": 0.3, "caixa_porta": 3.0, "trancar_porta": 5.0,
                          "trancar_longe": 1.0, "portas_fechadas": 0.2, "na_sala": 2.0}},
         min_iters=200, max_iters=3000, meta=("abrigo", 1.5)),

    # Agora valendo: metade do episodio para se preparar e os pegadores (ainda sem usar
    # objetos) entram em jogo. Os escondedores precisam fechar a sala a tempo e ficar nela.
    dict(nome="3b_esconder", treinar=("hider",), prep_fraction=0.5, ent_coef=0.01,
         acoes={"hider": TUDO, "seeker": ("mover",)},
         pesos={"hider": {"jogo": 1.0, "ir_caixa": 1.0, "encarar": 0.3, "caixa_porta": 3.0,
                          "trancar_porta": 5.0, "trancar_longe": 1.0, "na_sala": 2.0}},
         min_iters=150, max_iters=400, meta=("abrigo", 1.2)),

    # Oficina da rampa: a sala ja comeca fechada (uma caixa trancada em cada porta) e os
    # escondedores ficam parados dentro dela. O unico jeito de entrar e levar a rampa ate a
    # parede, subir e pular. Os pegadores nao recebem a acao de trancar (ver README).
    # A recompensa do jogo fica de fora: como a caixa (0,5 m) nao tapa o vao da porta (0,7 m),
    # com ela os pegadores aprendem a espiar pelas frestas em vez de usar a rampa.
    dict(nome="4_rampa", treinar=("seeker",), prep_fraction=0.0, sala_fechada=True, ent_coef=0.01,
         acoes={"hider": (), "seeker": ("mover", "agarrar")},
         pesos={"seeker": {"ir_rampa": 1.0, "encarar": 0.3, "lado_baixo": 1.0,
                           "rampa_parede": 3.0, "rampa_mira": 2.0, "altura": 4.0, "dentro_sala": 1.0}},
         min_iters=200, max_iters=1300, meta=("dentro", 0.35)),

    # De volta ao jogo: os pegadores enfrentam os escondedores treinados, que as vezes fecham
    # a sala, e tem so a segunda metade do episodio para entrar.
    dict(nome="4b_invadir", treinar=("seeker",), prep_fraction=0.5, ent_coef=0.01,
         acoes={"hider": TUDO, "seeker": ("mover", "agarrar")},
         pesos={"seeker": {"jogo": 1.0, "ir_rampa": 0.5, "lado_baixo": 0.5, "rampa_parede": 1.5,
                           "rampa_mira": 1.0, "altura": 2.0, "dentro_sala": 0.5}},
         min_iters=100, max_iters=300, meta=("visto", 0.55)),

    dict(nome="5_jogo", treinar=("hider", "seeker"), prep_fraction=0.5, ent_coef=0.01,
         acoes={"hider": TUDO, "seeker": ("mover", "agarrar")},
         pesos={"hider": {"jogo": 1.0, "caixa_porta": 1.0, "trancar_porta": 2.0, "trancar_longe": 0.5,
                          "na_sala": 0.5},
                "seeker": {"jogo": 1.0, "dentro_sala": 0.2, "rampa_parede": 0.5, "altura": 1.0}},
         min_iters=0, max_iters=None, meta=None),
]
NOMES_FASES = [f["nome"] for f in CURRICULO]

COMPONENTES = ("jogo", "novidade", "distancia", "ir_caixa", "caixa_porta", "trancar_porta",
               "ir_rampa", "rampa_parede", "dentro_sala", "encarar", "trancar_longe", "portas_fechadas",
               "na_sala", "lado_baixo", "rampa_mira", "altura")
GRUPO_DA_ACAO = {"action_movement": "mover", "action_pull": "agarrar", "action_glueall": "trancar"}
TIMES = ("hider", "seeker")
HIDDEN = 256
TAM_CELULA = 0.5
RAIO_PORTA = 0.45
MAX_PORTAS = 2
# Para que lado fica o interior da sala em cada parede interna (mesma ordem de paredes_sala).
NORMAL_PARA_DENTRO = (np.array([1.0, 0.0]), np.array([0.0, -1.0]))
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


def _alinhamento(frente, alvo):
    """Cosseno entre a direcao em que o agente olha e a direcao do alvo (1 = de frente)."""
    return float(np.dot(frente, alvo) / (np.linalg.norm(alvo) + 1e-9))


def _dist_segmento(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


class TwoTeamHideAndSeek:
    def __init__(self, n_hiders=2, n_seekers=2, n_boxes=2, n_ramps=1, frame_skip=4, horizon=120, prep_fraction=0.4,
                 sala_fechada=False):
        self.n_hiders, self.n_seekers = n_hiders, n_seekers
        self.n_agents = n_hiders + n_seekers
        self.n_boxes, self.n_ramps = n_boxes, n_ramps
        self.frame_skip, self.horizon = frame_skip, horizon
        self.renderizar, self.pausa_render = False, 0.0
        self.ao_quadro = None  # funcao chamada a cada passo do simulador (usada para gravar video)
        n = self.n_agents
        self.ordem_outros = np.array([np.argsort([(i - j) % n for j in range(1, n)]) for i in range(n)])
        self._env, self.prep_fraction = None, None
        self.configurar(prep_fraction, sala_fechada)

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

    def configurar(self, prep_fraction, sala_fechada=False):
        # sala_fechada: o episodio ja comeca com uma caixa trancada em cada porta (oficina da rampa).
        self.sala_fechada = bool(sala_fechada)
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
        obs = self._sortear_mundo()
        if self.sala_fechada:
            for _ in range(50):
                if self._fechar_sala():
                    break
                obs = self._sortear_mundo()
            # Um passo parado para a observacao refletir as caixas ja nas portas.
            neutro = np.tile(np.array(self.neutro, dtype=np.int64), (self.n_agents, 1))
            obs = self._env.step(self._montar_acao(neutro))[0]
            self._ler_estado()
        self.visitadas = [{self._celula(p)} for p in self.p_ag]
        self.pot = self._potenciais()
        self.ep = dict(passos_pos=0, visto=0.0, dentro=0.0, portas=None, abrigo=None, jogo_h=0.0, jogo_s=0.0,
                       subiu=0.0, invasao=0.0)
        return self._obs_agentes(obs)

    def _sortear_mundo(self):
        res = self._env.reset()
        obs = res[0] if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict) else res
        u = self._env.unwrapped
        self._sim = u.sim
        m = self._sim.model
        self._id_ag = np.array([m.body_name2id(f"agent{i}:particle") for i in range(self.n_agents)], dtype=int)
        self._id_cx = np.array([m.body_name2id(f"moveable_box{i}") for i in range(self.n_boxes)], dtype=int)
        self._id_rp = np.array([m.body_name2id(f"ramp{i}:ramp") for i in range(self.n_ramps)], dtype=int)
        self._adr_rz = np.array([m.get_joint_qpos_addr(f"agent{i}:rz") for i in range(self.n_agents)], dtype=int)

        paredes = next((md for md in u.modules if type(md).__name__ == "WallScenarios"), None)
        portas = getattr(paredes, "door_obs", None)
        self.portas = np.zeros((0, 2)) if portas is None else np.asarray(portas, dtype=np.float64)[:MAX_PORTAS, :2]
        cel = u.floor_size / u.grid_size
        q = u.metadata.get("quadrant_size", u.grid_size // 2)
        self.sala_x, self.sala_y = (u.grid_size - q + 0.5) * cel, (q + 0.5) * cel
        self.paredes_sala = [(np.array([self.sala_x, 0.0]), np.array([self.sala_x, self.sala_y])),
                             (np.array([self.sala_x, self.sala_y]), np.array([u.floor_size, self.sala_y]))]

        # Rampa: guarda, no referencial dela, a direcao em que a superficie sobe.
        self._geom_rp, self._sobe_local = None, None
        if self.n_ramps:
            g = m.geom_name2id("ramp0:ramp")
            malha = int(m.geom_dataid[g])
            ini, qtd = int(m.mesh_vertadr[malha]), int(m.mesh_vertnum[malha])
            v = np.asarray(m.mesh_vert[ini:ini + qtd], dtype=np.float64)
            R = np.asarray(self._sim.data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
            w = v @ R.T
            topo = w[w[:, 2] > w[:, 2].max() - 0.05]
            self._geom_rp, self._sobe_local = g, R.T @ (topo.mean(0) - w.mean(0))

        self.em_prep = self.prep_fraction > 0
        self._ler_estado()
        return obs

    def _fechar_sala(self):
        """Tranca uma caixa em cada porta, pelo lado de dentro, como se os escondedores ja
        tivessem feito a barricada. Devolve False se o mundo sorteado nao serve."""
        sim, m, lk = self._sim, self._sim.model, self._lock_caixa
        if lk is None or len(self.portas) != len(self._id_cx):
            return False
        alvos = np.array([pt + 0.31 * (np.array([1.0, 0.0]) if abs(pt[0] - self.sala_x) < 1e-6
                                       else np.array([0.0, -1.0])) for pt in self.portas])
        if len(alvos) > 1 and np.linalg.norm(alvos[0] - alvos[1]) < 0.6:
            return False                                    # portas coladas: as caixas se sobreporiam
        ph = self.p_ag[:self.n_hiders]
        if np.linalg.norm(ph[:, None] - alvos[None], axis=-1).min() < 0.75:
            return False                                    # escondedor em cima da porta (destrancaria a caixa)
        if any(self._dentro_sala(pt) for pt in self.p_rp):
            return False                                    # rampa presa dentro da sala
        qpos = sim.data.qpos
        for i, alvo in enumerate(alvos):
            corpo = self._id_cx[i]
            R = np.asarray(sim.data.body_xmat[corpo]).reshape(3, 3)
            giro = np.arctan2(R[1, 0], R[0, 0])
            delta = np.array([alvo[0] - self.p_cx[i][0], alvo[1] - self.p_cx[i][1], 0.0])
            juntas = np.flatnonzero(np.asarray(m.jnt_bodyid) == corpo)
            for j in juntas:
                adr = int(m.jnt_qposadr[j])
                if int(m.jnt_type[j]) == 3:                 # dobradica: deixa a caixa alinhada as paredes
                    qpos[adr] -= giro
                else:
                    qpos[adr] += float(np.dot(np.asarray(m.jnt_axis[j]), delta))
            for j in juntas:
                m.jnt_limited[j] = 1
                m.jnt_range[j, :] = qpos[int(m.jnt_qposadr[j])]
            lk.obj_locked[i] = True
            lk.which_locked[i] = 0                          # so quem trancou destranca: fica com o escondedor 0
            m.mat_emission[m.geom_matid[np.asarray(m.geom_bodyid) == corpo]] = 1
        sim.forward()
        self._ler_estado()
        return True

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
        # Direcao para onde cada agente olha (o cone de visao e o "agarrar" dependem dela).
        rz = np.asarray(self._sim.data.qpos)[self._adr_rz]
        self.frente = np.stack([np.sin(rz), -np.cos(rz)], axis=-1)
        self.z_ag = xp[self._id_ag, 2].copy()
        self.sobe = np.zeros(2)                             # direcao (no chao) em que a rampa sobe
        if self._geom_rp is not None:
            R = np.asarray(self._sim.data.geom_xmat[self._geom_rp], dtype=np.float64).reshape(3, 3)
            d = (R @ self._sobe_local)[:2]
            self.sobe = d / (np.linalg.norm(d) + 1e-9)
        lk = self._lock_caixa
        self.cx_trancada_h = np.zeros(len(self.p_cx), bool)
        if lk is not None and len(self.p_cx) and len(np.atleast_1d(lk.obj_locked)) == len(self.p_cx):
            self.cx_trancada_h = np.asarray(lk.obj_locked, bool) & (np.asarray(lk.which_locked) < self.n_hiders)

    def _potenciais(self):
        H, S = slice(0, self.n_hiders), slice(self.n_hiders, self.n_agents)
        pot = {k: np.zeros(self.n_agents) for k in
               ("distancia", "caixa_porta", "trancar_porta", "trancar_longe", "ir_rampa", "rampa_parede",
                "na_sala", "lado_baixo", "rampa_mira", "altura")}
        ph, ps = self.p_ag[H], self.p_ag[S]
        d_hs = np.linalg.norm(ph[:, None] - ps[None], axis=-1)
        pot["distancia"][H] = d_hs.min(1)
        pot["distancia"][S] = -d_hs.min(0)
        # Escondedor fora da sala nao esta protegido por barricada nenhuma.
        na_sala = np.array([float(self._dentro_sala(p)) for p in ph])
        pot["na_sala"][H] = na_sala
        n_portas = 0.0
        livres = np.ones(len(self.p_cx), bool)
        if len(self.p_cx) and len(self.portas):
            d_pc = np.linalg.norm(self.portas[:, None] - self.p_cx[None], axis=-1)
            cobre = (d_pc < RAIO_PORTA) & self.cx_trancada_h[None]
            n_portas = float(np.any(cobre, axis=1).sum())
            # Porta fechada so conta para quem esta do lado de dentro: fechar a sala por fora
            # (arrastando a caixa atras de si pela porta) deixa o escondedor exposto.
            pot["trancar_porta"][H] = n_portas * na_sala
            # Caixa trancada longe de qualquer porta nao serve de barricada e nao pode mais ser movida.
            longe = self.cx_trancada_h & ~np.any(d_pc < RAIO_PORTA, axis=0)
            pot["trancar_longe"][:] = -float(longe.sum())
            pot["caixa_porta"][:] = -_custo_atribuicao(d_pc)
            livres = ~np.any(cobre, axis=0)
        idx_livres = np.flatnonzero(livres)
        alvo = np.full(self.n_hiders, -1)
        if len(idx_livres):
            alvo = idx_livres[_atribuir(np.linalg.norm(ph[:, None] - self.p_cx[idx_livres][None], axis=-1))]
        if len(self.p_rp):
            c = self.p_rp[0]
            pot["ir_rampa"][S] = -np.linalg.norm(ps[:, None] - self.p_rp[None], axis=-1).min(1)
            dists = [_dist_segmento(c, a, b) for a, b in self.paredes_sala]
            k = int(np.argmin(dists))
            pot["rampa_parede"][:] = -dists[k]
            # A rampa so serve se o lado alto estiver virado para a parede da sala...
            pot["rampa_mira"][:] = float(np.dot(self.sobe, NORMAL_PARA_DENTRO[k])) * float(np.clip(1.0 - dists[k], 0.0, 1.0))
            # ...e so da para empurra-la e subir estando do lado baixo dela.
            v = c[None] - ps
            pot["lado_baixo"][S] = (v @ self.sobe) / (np.linalg.norm(v, axis=1) + 1e-9)
        # Altura do pegador: subir na rampa. Quem ja pulou para dentro conta como altura cheia,
        # para nao ser punido por descer do outro lado.
        pot["altura"][S] = [0.5 if self._dentro_sala(pt) else float(np.clip(z - 0.15, 0.0, 0.5))
                            for pt, z in zip(ps, self.z_ag[S])]
        pot["_alvo"] = alvo
        pot["_n_portas"] = n_portas
        pot["_abrigo"] = n_portas * float(na_sala.min()) if len(na_sala) else 0.0
        pot["_p_ag"], pot["_p_cx"] = self.p_ag.copy(), self.p_cx.copy()
        pot["_frente"], pot["_p_rp"] = self.frente.copy(), self.p_rp.copy()
        return pot

    def step(self, acoes):
        acoes = np.asarray(acoes, dtype=np.int64).reshape(self.n_agents, self.n_dims)
        acao = self._montar_acao(acoes)
        em_prep = self.em_prep
        jogo = np.zeros(self.n_agents)
        quadros_visto = 0.0  # passos do simulador, dentro deste passo de decisao, com escondedor a vista
        fim, info, k = False, {}, 0
        for k in range(1, self.frame_skip + 1):
            res = self._env.step({ch: v.copy() for ch, v in acao.items()})
            obs, rew, done, info = res[0], res[1], res[2], res[-1]
            if len(res) == 5:
                done = _bool_done(done) or _bool_done(res[3])
            rew_k = np.asarray(rew, dtype=np.float64).ravel()[:self.n_agents]
            jogo += rew_k
            quadros_visto += float(rew_k[self.n_hiders:].max() > 0)
            if self.renderizar:
                self._env.render()
                if self.pausa_render:
                    time.sleep(self.pausa_render)
            if self.ao_quadro is not None:
                self.em_prep = bool(info.get("in_prep_phase", False)) if isinstance(info, dict) else False
                self.ao_quadro(self, np.asarray(rew, dtype=np.float64).ravel())
            if _bool_done(done):
                fim = True
                break
        jogo /= max(k, 1)
        self.em_prep = bool(info.get("in_prep_phase", False)) if isinstance(info, dict) else False

        self._ler_estado()
        pot = self._potenciais()
        dp = {ch: pot[ch] - self.pot[ch] for ch in pot if not ch.startswith("_")}
        dp["ir_caixa"] = np.zeros(self.n_agents)
        dp["encarar"] = np.zeros(self.n_agents)
        ant = self.pot
        for j, b in enumerate(ant["_alvo"]):
            if b >= 0:
                d0 = np.linalg.norm(ant["_p_ag"][j] - ant["_p_cx"][b])
                dp["ir_caixa"][j] = d0 - np.linalg.norm(self.p_ag[j] - self.p_cx[b])
                dp["encarar"][j] = (_alinhamento(self.frente[j], self.p_cx[b] - self.p_ag[j])
                                    - _alinhamento(ant["_frente"][j], ant["_p_cx"][b] - ant["_p_ag"][j]))
        if len(self.p_rp):
            for j in range(self.n_hiders, self.n_agents):
                dp["encarar"][j] = (_alinhamento(self.frente[j], self.p_rp[0] - self.p_ag[j])
                                    - _alinhamento(ant["_frente"][j], ant["_p_rp"][0] - ant["_p_ag"][j]))
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
        c["trancar_longe"][H] = dp["trancar_longe"][H]
        c["na_sala"][H] = dp["na_sala"][H]
        # Unica auxiliar que nao e diferenca de potencial: paga a cada passo pelas portas ja
        # fechadas, para premiar quem fecha cedo (o jogo so da metade do episodio para isso).
        c["portas_fechadas"][H] = pot["trancar_porta"][H] / MAX_PORTAS
        if em_prep:
            c["ir_caixa"][H] = dp["ir_caixa"][H]
            c["caixa_porta"][H] = dp["caixa_porta"][H]
            c["encarar"][H] = dp["encarar"][H]
        else:
            c["encarar"][S] = dp["encarar"][S]
            c["novidade"][S] = novas[S]
            c["distancia"][S] = dp["distancia"][S]
            c["ir_rampa"][S] = dp["ir_rampa"][S]
            c["rampa_parede"][S] = dp["rampa_parede"][S]
            c["rampa_mira"][S] = dp["rampa_mira"][S]
            c["lado_baixo"][S] = dp["lado_baixo"][S]
            c["altura"][S] = dp["altura"][S]
            c["dentro_sala"][S] = [float(self._dentro_sala(p)) for p in self.p_ag[S]]
            self.ep["passos_pos"] += 1
            # Contado por passo do simulador: com frame_skip > 1 a media do passo de decisao
            # esconderia os instantes em que o escondedor so aparece em parte dele.
            self.ep["visto"] += quadros_visto / max(k, 1)
            self.ep["dentro"] += float(c["dentro_sala"][S].max() > 0)
            # Pegador no alto da rampa, e pegador dentro da sala com as duas portas fechadas.
            self.ep["subiu"] = max(self.ep["subiu"], float(self.z_ag[S].max() >= 0.5))
            if pot["_n_portas"] >= MAX_PORTAS and c["dentro_sala"][S].max() > 0:
                self.ep["invasao"] = 1.0
            if self.ep["portas"] is None:
                self.ep["portas"], self.ep["abrigo"] = float(pot["_n_portas"]), float(pot["_abrigo"])
        self.ep["jogo_h"] += float(jogo[H].mean())
        self.ep["jogo_s"] += float(jogo[S].mean())

        info_ep = None
        if fim:
            pp = max(self.ep["passos_pos"], 1)
            n_cel = [len(v) for v in self.visitadas]
            info_ep = dict(celulas=float(np.mean(n_cel)), celulas_h=float(np.mean(n_cel[H])),
                           celulas_s=float(np.mean(n_cel[S])), visto=self.ep["visto"] / pp,
                           dentro=self.ep["dentro"] / pp,
                           portas=self.ep["portas"] if self.ep["portas"] is not None else float(pot["_n_portas"]),
                           abrigo=self.ep["abrigo"] if self.ep["abrigo"] is not None else float(pot["_abrigo"]),
                           jogo_h=self.ep["jogo_h"], jogo_s=self.ep["jogo_s"],
                           subiu=self.ep["subiu"], invasao=self.ep["invasao"])
        return self._obs_agentes(obs), comp, fim, info_ep


def worker_process(remote, parent_remote, cfg_amb, prep_fraction, sala_fechada=False):
    parent_remote.close()
    env = TwoTeamHideAndSeek(**cfg_amb, prep_fraction=prep_fraction, sala_fechada=sala_fechada)
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
                env.configurar(*data)
                remote.send(env.reset())
            elif cmd == "close":
                env.close()
                remote.close()
                break
        except (EOFError, KeyboardInterrupt, BrokenPipeError):
            break


class ParallelEnvs:
    def __init__(self, num_envs, cfg_amb, prep_fraction, sala_fechada=False):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = []
        for wr, r in zip(self.work_remotes, self.remotes):
            p = mp.Process(target=worker_process, args=(wr, r, cfg_amb, prep_fraction, sala_fechada))
            p.daemon = True
            p.start()
            self.ps.append(p)
        for wr in self.work_remotes:
            wr.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def configurar(self, prep_fraction, sala_fechada=False):
        for remote in self.remotes:
            remote.send(("config", (prep_fraction, sala_fechada)))
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
