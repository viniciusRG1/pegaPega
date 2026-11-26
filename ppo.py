"""Rede, PPO, checkpoints e laco de treino."""
import os
import sys
import time
import argparse
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import preparar_ambiente
from ambiente import (AMBIENTE, COMPONENTES, CURRICULO, HIDDEN, NOMES_FASES, TIMES, TREINO, ParallelEnvs,
                      TwoTeamHideAndSeek)


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


def train_ppo_tbptt_amp(agent, optimizer, scaler, mascara, b, cfg, ent_coef, usar_amp, device, so_critico=False):
    # so_critico: atualiza apenas a cabeca de valor (aquecimento apos trocar de fase ou de recompensa).
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
            if so_critico:
                for nome, param in agent.named_parameters():
                    if not nome.startswith("critic."):
                        param.grad = None
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                kl_epoca.append(((ratio - 1) - log_ratio).mean().item())
                ents.append(entropy.mean().item())
        kls += kl_epoca
        if cfg["target_kl"] and np.mean(kl_epoca) > cfg["target_kl"] and not so_critico:
            break
    return float(np.mean(kls)), float(np.mean(ents))


def alargar_entrada(pesos_rede, obs_dim):
    """Adapta os pesos de uma rede treinada com menos entradas: as entradas novas (sempre
    acrescentadas ao fim da observacao) comecam com peso zero, entao o comportamento nao muda."""
    w = pesos_rede["fc.0.weight"]
    if w.shape[1] == obs_dim:
        return pesos_rede, False
    assert w.shape[1] < obs_dim, "o checkpoint tem mais entradas que o ambiente"
    novo = dict(pesos_rede)
    novo["fc.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], obs_dim - w.shape[1], dtype=w.dtype)], dim=1)
    return novo, True


# =============================================================================
# CURRICULO / CHECKPOINT
# =============================================================================
class Historico:
    """Acrescenta uma linha por iteracao em um CSV, para os graficos de treino."""
    COLUNAS = ("iteracao", "fase", "iters_fase", "rew_hider", "rew_seeker", "celulas", "celulas_h",
               "celulas_s", "visto", "dentro", "portas", "abrigo", "jogo_h", "jogo_s", "vedadas", "selada",
               "subiu", "invasao", "rampa_travada", "segundos")

    def __init__(self, caminho):
        self.caminho = caminho
        if not os.path.exists(caminho):
            with open(caminho, "w", encoding="utf-8") as f:
                f.write(",".join(self.COLUNAS) + "\n")

    def registrar(self, estado, nome_fase, r_medio, segundos):
        m = estado["metricas"]
        linha = [estado["iteracao"], nome_fase, estado["iters_fase"], r_medio["hider"], r_medio["seeker"]]
        linha += [m.get(k, float("nan")) for k in self.COLUNAS[5:-1]] + [segundos]
        with open(self.caminho, "a", encoding="utf-8") as f:
            f.write(",".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in linha) + "\n")


def fase_do_checkpoint(ck):
    """Indice da fase em que o checkpoint estava, procurando pelo nome quando ele foi salvo."""
    nome = ck.get("fase_nome")
    if nome in NOMES_FASES:
        return NOMES_FASES.index(nome)
    return min(ck.get("estado", {}).get("fase", 0), len(CURRICULO) - 1)


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


def salvar(caminho, agentes, opts, scalers, norm_rew, estado, obs_dim, nvec, cfg_amb=None):
    torch.save({
        "agentes": {t: agentes[t].state_dict() for t in TIMES},
        "opts": {t: opts[t].state_dict() for t in TIMES},
        "scalers": {t: scalers[t].state_dict() for t in TIMES},
        "norm_rew": {t: norm_rew[t].state_dict() for t in TIMES},
        "estado": estado, "obs_dim": obs_dim, "nvec": list(nvec), "ambiente": dict(cfg_amb or AMBIENTE),
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
    ap.add_argument("--pasta", default=None, help="pasta dos checkpoints (padrao: %s)" % TREINO["pasta"])
    ap.add_argument("--iniciar-de", default=None,
                    help="checkpoint de partida quando a pasta ainda nao tem checkpoint_recente.pt")
    ap.add_argument("--iters", type=int, default=None, help="para depois de N iteracoes nesta execucao")
    ap.add_argument("--iters-fase-final", type=int, default=None,
                    help="para quando a ultima fase do curriculo completar N iteracoes")
    ap.add_argument("--frame-skip", type=int, default=None,
                    help="passos do simulador por decisao (padrao: o do checkpoint, senao %d)" % AMBIENTE["frame_skip"])
    ap.add_argument("--horizonte", type=int, default=None,
                    help="passos do simulador por episodio (padrao: o do checkpoint, senao %d)" % AMBIENTE["horizon"])
    ap.add_argument("--obs-portas", action="store_true",
                    help="acrescenta a observacao a posicao de agente e caixas relativa a cada porta")
    ap.add_argument("--aquecer", type=int, default=0,
                    help="nas primeiras N iteracoes desta execucao so a cabeca de valor e treinada")
    ap.add_argument("--passos", type=int, default=None, help="decisoes coletadas por ambiente a cada iteracao")
    ap.add_argument("--obs-rampa", action="store_true",
                    help="acrescenta a observacao a posicao da rampa (sempre visivel) e se ela foi trancada")
    ap.add_argument("--tam-caixa", type=float, default=None, help="lado da caixa em metros (padrao 0,5)")
    ap.add_argument("--tam-porta", type=int, default=None,
                    help="largura da porta em celulas da grade (padrao 2: vao de 0,7 m; 1: vao de 0,5 m)")
    ap.add_argument("--lr", type=float, default=None, help="troca a taxa de aprendizado")
    ap.add_argument("--ent", type=float, default=None, help="troca o coeficiente de entropia da fase")
    ap.add_argument("--peso", action="append", default=[], metavar="TIME:COMPONENTE=VALOR",
                    help="troca um peso de recompensa da fase, ex.: hider:trancar_longe=4 (pode repetir)")
    ap.add_argument("--threads", type=int, default=4, help="threads do PyTorch na CPU")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # o log aparece na hora mesmo redirecionado

    cfg = dict(TREINO)
    if args.envs:
        cfg["num_envs"] = args.envs
    if args.pasta:
        cfg["pasta"] = args.pasta
    if args.passos:
        cfg["steps_per_env"] = args.passos
    torch.set_num_threads(max(1, args.threads))
    num_envs, T, L = cfg["num_envs"], cfg["steps_per_env"], cfg["seq_len"]
    assert T % L == 0, "steps_per_env precisa ser multiplo de seq_len"
    n_chunks = T // L

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usar_amp = device.type == "cuda"
    print(f"Dispositivo em uso: {device}")

    os.makedirs(cfg["pasta"], exist_ok=True)
    ck_recente = os.path.join(cfg["pasta"], "checkpoint_recente.pt")
    origem = ck_recente if os.path.exists(ck_recente) else args.iniciar_de
    ck = None
    if origem:
        ck = torch.load(origem, map_location=device, weights_only=False)
    cfg_amb = dict(AMBIENTE)
    if ck is not None:
        cfg_amb.update(ck.get("ambiente", {}))
    if args.frame_skip:
        cfg_amb["frame_skip"] = args.frame_skip
    if args.horizonte:
        cfg_amb["horizon"] = args.horizonte
    if args.obs_portas:
        cfg_amb["obs_portas"] = True
    if args.obs_rampa:
        cfg_amb["obs_rampa"] = True
    if args.tam_caixa:
        cfg_amb["tam_caixa"] = args.tam_caixa
    if args.tam_porta:
        cfg_amb["tam_porta"] = args.tam_porta
    print(f"Ambiente: {cfg_amb} | MuJoCo: {preparar_ambiente.BACKEND_MUJOCO}")

    dummy = TwoTeamHideAndSeek(**cfg_amb, prep_fraction=CURRICULO[0]["prep_fraction"])
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

    if ck is not None:
        try:
            assert list(ck["nvec"]) == list(nvec)
            for t in TIMES:
                pesos_rede, alargada = alargar_entrada(ck["agentes"][t], obs_dim)
                agentes[t].load_state_dict(pesos_rede)
                if alargada:
                    print(f"Rede do time {t}: entrada alargada de {ck['agentes'][t]['fc.0.weight'].shape[1]} "
                          f"para {obs_dim}; otimizador reiniciado.")
                else:
                    try:
                        formas = [tuple(v["exp_avg"].shape) for v in ck["opts"][t]["state"].values() if "exp_avg" in v]
                        assert formas == [tuple(p.shape) for p in agentes[t].parameters()][:len(formas)]
                        opts[t].load_state_dict(ck["opts"][t])
                    except Exception:
                        print(f"Otimizador do time {t} incompativel; reiniciado.")
                        opts[t] = optim.Adam(agentes[t].parameters(), lr=cfg["lr"], eps=1e-5)
                scalers[t].load_state_dict(ck["scalers"][t])
                norm_rew[t].load_state_dict(ck["norm_rew"][t])
            estado = ck["estado"]
            estado["fase"] = fase_do_checkpoint(ck)
            print(f"Treino retomado de {origem} na iteracao {estado['iteracao'] + 1} "
                  f"(fase {CURRICULO[estado['fase']]['nome']}).")
        except Exception as e:
            print(f"Checkpoint incompativel ignorado ({type(e).__name__}). Iniciando treino limpo.")
    if args.fase is not None:
        estado.update(fase=min(max(args.fase, 1), len(CURRICULO)) - 1, iters_fase=0, metricas={})
        norm_rew = {t: RunningMeanStd() for t in TIMES}
        print(f"Fase forcada: {CURRICULO[estado['fase']]['nome']}")
    if args.lr:
        for t in TIMES:
            for g in opts[t].param_groups:
                g["lr"] = args.lr

    i_fase = estado["fase"]
    envs = ParallelEnvs(num_envs, cfg_amb, CURRICULO[i_fase]["prep_fraction"],
                        CURRICULO[i_fase].get("sala_fechada", False), CURRICULO[i_fase].get("horizonte"),
                        CURRICULO[i_fase].get("defesa_rampa", False))
    print(f"===== FASE {CURRICULO[i_fase]['nome']} =====")
    historico = Historico(os.path.join(cfg["pasta"], "historico.csv"))
    iter_inicial = estado["iteracao"]

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
                obs = envs.configurar(CURRICULO[i_fase]["prep_fraction"], CURRICULO[i_fase].get("sala_fechada", False),
                                      CURRICULO[i_fase].get("horizonte"), CURRICULO[i_fase].get("defesa_rampa", False))
                dones = np.ones(num_envs, dtype=np.float32)
                lstm = novos_estados_lstm()
                norm_rew = {t: RunningMeanStd() for t in TIMES}
                ret_run = {t: np.zeros(num_envs * n_time[t]) for t in TIMES}
                print(f"===== FASE {CURRICULO[i_fase]['nome']} =====")
            fase = CURRICULO[i_fase]
            mascaras = {t: mascara_acoes(nvec, grupos, neutro, fase["acoes"][t]).to(device) for t in TIMES}
            pesos = {t: np.array([fase["pesos"].get(t, {}).get(c, 0.0) for c in COMPONENTES], np.float32)
                     for t in TIMES}
            for troca_peso in args.peso:
                alvo, valor = troca_peso.split("=")
                t, c = alvo.split(":")
                pesos[t][COMPONENTES.index(c)] = float(valor)

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
                                              args.ent if args.ent is not None else fase.get("ent_coef", 0.01),
                                              usar_amp, device,
                                              so_critico=estado["iteracao"] - iter_inicial < args.aquecer)
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
                  f"visto {m.get('visto', 0):.2f} dentro {m.get('dentro', 0):.2f} portas {m.get('portas', 0):.2f} "
                  f"abrigo {m.get('abrigo', 0):.2f} ved {m.get('vedadas', 0):.2f} sel {m.get('selada', 0):.2f} "
                  f"sub {m.get('subiu', 0):.2f} inv {m.get('invasao', 0):.2f} trav {m.get('rampa_travada', 0):.2f} | "
                  f"{' | '.join(kl_txt)} | {time.time() - t0:.1f}s")

            historico.registrar(estado, fase["nome"], r_medio, time.time() - t0)

            if deve_avancar(i_fase, estado):
                salvar(os.path.join(cfg["pasta"], f"fase_{fase['nome']}.pt"), agentes, opts, scalers, norm_rew,
                       estado, obs_dim, nvec, cfg_amb)
                print(f"Fase {fase['nome']} concluida em {estado['iters_fase']} iteracoes.")
                estado.update(fase=i_fase + 1, iters_fase=0, metricas={})

            if estado["iteracao"] % cfg["salvar_cada"] == 0 or estado["fase"] != i_fase:
                salvar(ck_recente, agentes, opts, scalers, norm_rew, estado, obs_dim, nvec, cfg_amb)
            if estado["iteracao"] % 250 == 0:
                salvar(os.path.join(cfg["pasta"], f"ckpt_iter_{estado['iteracao']}.pt"), agentes, opts, scalers,
                       norm_rew, estado, obs_dim, nvec, cfg_amb)
            if args.iters and estado["iteracao"] - iter_inicial >= args.iters:
                print(f"Limite de {args.iters} iteracoes desta execucao atingido.")
                break
            if (args.iters_fase_final and estado["fase"] == len(CURRICULO) - 1
                    and estado["iters_fase"] >= args.iters_fase_final):
                print(f"Fase final completou {args.iters_fase_final} iteracoes.")
                break
    except KeyboardInterrupt:
        print("Interrompido pelo usuario.")
    finally:
        if estado["iteracao"] > 0:
            salvar(ck_recente, agentes, opts, scalers, norm_rew, estado, obs_dim, nvec, cfg_amb)
        print("Fechando ambientes em paralelo...")
        envs.close()
        print("Treino finalizado com seguranca.")
