import os
import sys
import time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym
from mae_envs.envs.hide_and_seek import make_env

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def get_space_info(space):
    if hasattr(space, "spaces"):
        if isinstance(space.spaces, dict):
            keys = sorted(space.spaces.keys())
            sub_infos = {k: get_space_info(space.spaces[k]) for k in keys}
            total_size = sum(info[0] for info in sub_infos.values())
            lows = np.concatenate([info[1] for info in sub_infos.values()]) if total_size > 0 else np.array([], dtype=np.float32)
            highs = np.concatenate([info[2] for info in sub_infos.values()]) if total_size > 0 else np.array([], dtype=np.float32)
            neutrals = np.concatenate([info[3] for info in sub_infos.values()]) if total_size > 0 else np.array([], dtype=np.float32)
            def unflatten_dict(x):
                out = {}
                idx = 0
                for k in keys:
                    sz, _, _, _, fn = sub_infos[k]
                    out[k] = fn(x[idx:idx + sz])
                    idx += sz
                return out
            return total_size, lows, highs, neutrals, unflatten_dict
        else:
            sub_infos = [get_space_info(sub_s) for sub_s in space.spaces]
            total_size = sum(info[0] for info in sub_infos)
            lows = np.concatenate([info[1] for info in sub_infos]) if total_size > 0 else np.array([], dtype=np.float32)
            highs = np.concatenate([info[2] for info in sub_infos]) if total_size > 0 else np.array([], dtype=np.float32)
            neutrals = np.concatenate([info[3] for info in sub_infos]) if total_size > 0 else np.array([], dtype=np.float32)
            def unflatten_tuple(x):
                out = []
                idx = 0
                for sz, _, _, _, fn in sub_infos:
                    out.append(fn(x[idx:idx + sz]))
                    idx += sz
                return tuple(out)
            return total_size, lows, highs, neutrals, unflatten_tuple
    elif type(space).__name__ == "Discrete":
        n = int(space.n)
        low = np.array([0.0], dtype=np.float32)
        high = np.array([float(n - 1)], dtype=np.float32)
        ntr = np.array([float(n - 1) / 2.0 if n > 2 else 0.0], dtype=np.float32)
        return 1, low, high, ntr, lambda x: int(np.clip(np.round(x[0]), 0, n - 1))
    elif type(space).__name__ == "MultiDiscrete" or hasattr(space, "nvec"):
        nvec = np.array(space.nvec, dtype=np.int64)
        sz = len(nvec)
        low = np.zeros(sz, dtype=np.float32)
        high = (nvec - 1).astype(np.float32)
        ntr = np.where(nvec > 2, (nvec - 1) / 2.0, 0.0).astype(np.float32)
        return sz, low, high, ntr, lambda x: np.clip(np.round(x), 0, nvec - 1).astype(np.int64)
    elif type(space).__name__ == "MultiBinary":
        n = int(space.n) if hasattr(space, "n") else int(np.prod(space.shape))
        low = np.zeros(n, dtype=np.float32)
        high = np.ones(n, dtype=np.float32)
        ntr = np.zeros(n, dtype=np.float32)
        return n, low, high, ntr, lambda x: np.clip(np.round(x), 0, 1).astype(np.int8)
    else:
        shp = getattr(space, "shape", None)
        if shp is None or len(shp) == 0:
            l = np.asarray(getattr(space, "low", -1.0), dtype=np.float32).flatten()
            h = np.asarray(getattr(space, "high", 1.0), dtype=np.float32).flatten()
            sz = len(l) if len(l) > 0 else 1
            ntr = np.zeros(sz, dtype=np.float32)
            return sz, l, h, ntr, lambda x: x[0] if sz == 1 else x
        else:
            sz = int(np.prod(shp))
            l = np.asarray(space.low, dtype=np.float32).flatten()
            h = np.asarray(space.high, dtype=np.float32).flatten()
            ntr = np.zeros(sz, dtype=np.float32)
            return sz, l, h, ntr, lambda x: x.reshape(shp)

def get_team_indices(space, n_hiders, n_seekers):
    n_agents = n_hiders + n_seekers
    hider_idx = []
    seeker_idx = []
    if hasattr(space, "spaces") and isinstance(space.spaces, dict):
        offset = 0
        for k in sorted(space.spaces.keys()):
            sz, _, _, _, _ = get_space_info(space.spaces[k])
            h_sz = (sz * n_hiders) // n_agents
            hider_idx.extend(range(offset, offset + h_sz))
            seeker_idx.extend(range(offset + h_sz, offset + sz))
            offset += sz
    else:
        sz, _, _, _, _ = get_space_info(space)
        h_sz = (sz * n_hiders) // n_agents
        hider_idx.extend(range(0, h_sz))
        seeker_idx.extend(range(h_sz, sz))
    return np.array(hider_idx, dtype=np.int64), np.array(seeker_idx, dtype=np.int64)

class TwoTeamHideAndSeek(gym.Env):
    def __init__(self, n_hiders=2, n_seekers=2, n_boxes=2, n_ramps=1, prep_steps=0, frame_skip=4):
        super().__init__()
        self._env = make_env(n_hiders=n_hiders, n_seekers=n_seekers, n_boxes=n_boxes, n_ramps=n_ramps)
        self.n_hiders = n_hiders
        self.n_seekers = n_seekers
        self.n_agents = n_hiders + n_seekers
        self.prep_steps = prep_steps
        self.frame_skip = frame_skip
        self.episode_step = 0

        self.visited_cells = set()
        self.prev_seeker_pos = None

        raw_obs = self._env.reset()
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        flat_obs = self._flatten_obs(raw_obs)

        self.obs_dim = flat_obs.shape[0]
        total_size, lows, highs, neutrals, self.unflatten_action = get_space_info(self._env.action_space)
        self.hider_idx, self.seeker_idx = get_team_indices(self._env.action_space, n_hiders, n_seekers)

        self.total_act_dim = total_size
        self.lows = lows
        self.highs = highs
        self.neutrals = neutrals
        self.mids = (lows + highs) / 2.0
        self.spans = (highs - lows) / 2.0

        self.act_dim_hider = len(self.hider_idx)
        self.act_dim_seeker = len(self.seeker_idx)

    def _get_sim(self):
        if hasattr(self._env, "sim"):
            return self._env.sim
        if hasattr(self._env, "unwrapped") and hasattr(self._env.unwrapped, "sim"):
            return self._env.unwrapped.sim
        return None

    def _get_seeker_positions(self):
        sim = self._get_sim()
        if sim is None:
            return None
        positions = []
        for i in range(self.n_hiders, self.n_agents):
            try:
                b_id = sim.model.body_name2id(f"agent{i}:particle")
                pos = sim.data.body_xpos[b_id][:2].copy()
                positions.append(pos)
            except Exception:
                pass
        return np.array(positions) if len(positions) > 0 else None

    def _flatten_obs(self, obs):
        if isinstance(obs, dict):
            arrays = [self._flatten_obs(obs[k]) for k in sorted(obs.keys())]
            res = np.concatenate(arrays).astype(np.float32)
        elif isinstance(obs, (list, tuple)):
            arrays = [self._flatten_obs(item) for item in obs]
            res = np.concatenate(arrays).astype(np.float32)
        else:
            res = np.asarray(obs, dtype=np.float32).flatten()
        return np.nan_to_num(res, nan=0.0, posinf=10.0, neginf=-10.0)

    def reset(self):
        self.episode_step = 0
        self.visited_cells.clear()
        res = self._env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        self.prev_seeker_pos = self._get_seeker_positions()
        return self._flatten_obs(obs)

    def step(self, act_hider, act_seeker):
        self.episode_step += 1

        raw_action = np.zeros(self.total_act_dim, dtype=np.float32)
        raw_action[self.hider_idx] = np.nan_to_num(act_hider, nan=0.0, posinf=1.0, neginf=-1.0)
        raw_action[self.seeker_idx] = np.nan_to_num(act_seeker, nan=0.0, posinf=1.0, neginf=-1.0)

        mapped_action = self.mids + np.tanh(raw_action) * self.spans
        if self.episode_step <= self.prep_steps:
            mapped_action[self.seeker_idx] = self.neutrals[self.seeker_idx]

        mapped_action = np.clip(mapped_action, self.lows, self.highs)
        action_orig = self.unflatten_action(mapped_action)

        tot_rew_h = 0.0
        tot_rew_s = 0.0
        fim = False

        for _ in range(self.frame_skip):
            action_orig = {k: np.array(v) for k, v in action_orig.items()}
            res = self._env.step(action_orig)
            if len(res) == 4:
                obs, rew, done, info = res
                truncated = False
            else:
                obs, rew, done, truncated, info = res

            if isinstance(rew, dict):
                r_vals = [rew[k] for k in sorted(rew.keys())]
                r_h = float(np.mean(r_vals[:self.n_hiders]))
                r_s = float(np.mean(r_vals[self.n_hiders:]))
            else:
                rew_arr = np.asarray(rew, dtype=np.float32).flatten()
                if len(rew_arr) >= self.n_agents:
                    r_h = float(np.mean(rew_arr[:self.n_hiders]))
                    r_s = float(np.mean(rew_arr[self.n_hiders:]))
                else:
                    r_h = float(rew_arr[0])
                    r_s = -float(rew_arr[0])

            tot_rew_h += r_h
            tot_rew_s += r_s

            done_bool = bool(done.get("__all__", any(done.values()))) if isinstance(done, dict) else bool(done)
            trunc_bool = bool(truncated.get("__all__", any(truncated.values()))) if isinstance(truncated, dict) else bool(truncated)

            if done_bool or trunc_bool:
                fim = True
                break

        sim = self._get_sim()
        if sim is not None:
            try:
                pos_h = [
                    sim.data.body_xpos[sim.model.body_name2id(f"agent{i}:particle")][:2].copy()
                    for i in range(self.n_hiders)
                ]
                pos_s = [
                    sim.data.body_xpos[sim.model.body_name2id(f"agent{i}:particle")][:2].copy()
                    for i in range(self.n_hiders, self.n_agents)
                ]
                dists = [np.linalg.norm(ps - ph) for ps in pos_s for ph in pos_h]
                if len(dists) > 0:
                    min_dist = float(np.min(dists))
                    tot_rew_s += -0.1 * min_dist
                    tot_rew_h += 0.1 * min_dist
            except Exception:
                pass

        current_seeker_pos = self._get_seeker_positions()
        exploration_bonus = 0.0

        if self.episode_step > self.prep_steps and current_seeker_pos is not None:
            if self.prev_seeker_pos is not None:
                dist_moved = np.mean(np.linalg.norm(current_seeker_pos - self.prev_seeker_pos, axis=1))
                exploration_bonus += float(dist_moved * 0.2)

            for pos in current_seeker_pos:
                cell = (int(np.round(pos[0])), int(np.round(pos[1])))
                if cell not in self.visited_cells:
                    self.visited_cells.add(cell)
                    exploration_bonus += 0.25

            self.prev_seeker_pos = current_seeker_pos

        if self.episode_step <= self.prep_steps:
            tot_rew_s = 0.0
        else:
            tot_rew_s += exploration_bonus

        if fim:
            self.episode_step = 0
            self.visited_cells.clear()
            self.prev_seeker_pos = None

        tot_rew_h = float(np.nan_to_num(tot_rew_h, nan=0.0))
        tot_rew_s = float(np.nan_to_num(tot_rew_s, nan=0.0))

        return self._flatten_obs(obs), tot_rew_h, tot_rew_s, fim

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

def worker_process(remote, parent_remote, n_h, n_s, n_b, n_r, prep, skip):
    parent_remote.close()
    carregar_dlls()
    env = TwoTeamHideAndSeek(n_hiders=n_h, n_seekers=n_s, n_boxes=n_b, n_ramps=n_r, prep_steps=prep, frame_skip=skip)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                act_h, act_s = data
                ob, r_h, r_s, done = env.step(act_h, act_s)
                if done:
                    ob = env.reset()
                remote.send((ob, r_h, r_s, done))
            elif cmd == "reset":
                ob = env.reset()
                remote.send(ob)
            elif cmd == "close":
                env.close()
                remote.close()
                break
        except EOFError:
            break

class ParallelEnvs:
    def __init__(self, num_envs, n_h=2, n_s=2, n_b=2, n_r=1, prep=0, skip=4):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = []
        for wr, r in zip(self.work_remotes, self.remotes):
            p = mp.Process(target=worker_process, args=(wr, r, n_h, n_s, n_b, n_r, prep, skip))
            p.daemon = True
            p.start()
            self.ps.append(p)
        for wr in self.work_remotes:
            wr.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step(self, acts_h, acts_s):
        for remote, a_h, a_s in zip(self.remotes, acts_h, acts_s):
            remote.send(("step", (a_h, a_s)))
        results = [remote.recv() for remote in self.remotes]
        obs, r_h, r_s, dones = zip(*results)
        return np.stack(obs), np.array(r_h, dtype=np.float32), np.array(r_s, dtype=np.float32), np.array(dones, dtype=np.float32)

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=1.0)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.LayerNorm(256),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(256, 256)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.ln_lstm = nn.LayerNorm(256)
        self.actor_mean = layer_init(nn.Linear(256, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

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
        out = self.ln_lstm(out)
        return out, lstm_state

    def get_value(self, x, lstm_state, done):
        out, _ = self.forward_lstm(x, lstm_state, done)
        return self.critic(out)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        out, next_lstm_state = self.forward_lstm(x, lstm_state, done)
        mean = 2.0 * torch.tanh(self.actor_mean(out))
        logstd = torch.clamp(self.actor_logstd, -1.0, 0.5).expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(axis=-1), dist.entropy().sum(axis=-1), self.critic(out), next_lstm_state

def train_ppo_tbptt_amp(agent, optimizer, scaler, obs_b, act_b, logp_b, adv_b, ret_b, val_b, done_b, h_b, c_b, steps_per_env, num_envs, seq_len, act_dim, num_minibatches, update_epochs, clip_coef, vf_coef, ent_coef):
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    adv_b = torch.clamp(adv_b, -5.0, 5.0)

    num_chunks = steps_per_env // seq_len
    total_seqs = num_chunks * num_envs
    seqs_per_mb = total_seqs // num_minibatches

    def to_seq(tensor_in, feat_dim=None):
        if feat_dim is None:
            return tensor_in.reshape(num_chunks, seq_len, num_envs).transpose(0, 1).reshape(seq_len, total_seqs)
        return tensor_in.reshape(num_chunks, seq_len, num_envs, feat_dim).transpose(0, 1).reshape(seq_len, total_seqs, feat_dim)

    obs_seq = to_seq(obs_b, obs_b.shape[-1])
    act_seq = to_seq(act_b, act_dim)
    logp_seq = to_seq(logp_b)
    adv_seq = to_seq(adv_b)
    ret_seq = to_seq(ret_b)
    val_seq = to_seq(val_b)
    done_seq = to_seq(done_b)

    init_h = h_b[0::seq_len].reshape(1, total_seqs, 256)
    init_c = c_b[0::seq_len].reshape(1, total_seqs, 256)

    seq_indices = np.arange(total_seqs)

    for _ in range(update_epochs):
        np.random.shuffle(seq_indices)
        for start in range(0, total_seqs, seqs_per_mb):
            end = start + seqs_per_mb
            mb_inds = seq_indices[start:end]

            m_obs = obs_seq[:, mb_inds].reshape(-1, obs_b.shape[-1])
            m_act = act_seq[:, mb_inds].reshape(-1, act_dim)
            m_logprob = logp_seq[:, mb_inds].reshape(-1)
            m_adv = adv_seq[:, mb_inds].reshape(-1)
            m_ret = ret_seq[:, mb_inds].reshape(-1)
            m_val = val_seq[:, mb_inds].reshape(-1)
            m_done = done_seq[:, mb_inds].reshape(-1)

            m_lstm_init = (
                init_h[:, mb_inds].contiguous(),
                init_c[:, mb_inds].contiguous()
            )

            with torch.amp.autocast("cuda"):
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    m_obs, m_lstm_init, m_done, m_act
                )

                log_ratio = torch.clamp(newlogprob - m_logprob, -10.0, 10.0)
                ratio = torch.exp(log_ratio)

                pg_loss1 = -m_adv * ratio
                pg_loss2 = -m_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newvalue.squeeze(-1) - m_ret) ** 2
                v_clipped = m_val + torch.clamp(newvalue.squeeze(-1) - m_val, -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - m_ret) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo em uso: {device}")

    num_envs = 12
    steps_per_env = 120
    seq_len = 20
    num_minibatches = 4
    num_iterations = 20000

    n_hiders = 2
    n_seekers = 2
    n_boxes = 2
    n_ramps = 1
    prep_steps = 0
    frame_skip = 4

    dummy_env = TwoTeamHideAndSeek(n_hiders, n_seekers, n_boxes, n_ramps, prep_steps, frame_skip)
    obs_dim = dummy_env.obs_dim
    act_dim_hider = dummy_env.act_dim_hider
    act_dim_seeker = dummy_env.act_dim_seeker
    dummy_env.close()

    envs = ParallelEnvs(num_envs, n_hiders, n_seekers, n_boxes, n_ramps, prep_steps, frame_skip)

    agent_hider = ActorCriticLSTM(obs_dim, act_dim_hider).to(device)
    agent_seeker = ActorCriticLSTM(obs_dim, act_dim_seeker).to(device)

    base_lr = 3e-4
    opt_hider = optim.Adam(agent_hider.parameters(), lr=base_lr, eps=1e-5)
    opt_seeker = optim.Adam(agent_seeker.parameters(), lr=base_lr, eps=1e-5)

    scaler_hider = torch.amp.GradScaler("cuda")
    scaler_seeker = torch.amp.GradScaler("cuda")

    update_epochs = 4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.05

    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_recente = "./checkpoints/checkpoint_recente.pt"

    start_iteration = 1
    best_hider_rew = -float("inf")
    best_seeker_rew = -float("inf")

    if os.path.exists(checkpoint_recente):
        try:
            ckpt = torch.load(checkpoint_recente, map_location=device)
            agent_hider.load_state_dict(ckpt["agent_hider"])
            agent_seeker.load_state_dict(ckpt["agent_seeker"])
            opt_hider.load_state_dict(ckpt["opt_hider"])
            opt_seeker.load_state_dict(ckpt["opt_seeker"])
            if "scaler_hider" in ckpt:
                scaler_hider.load_state_dict(ckpt["scaler_hider"])
                scaler_seeker.load_state_dict(ckpt["scaler_seeker"])
            start_iteration = ckpt["iteration"] + 1
            best_hider_rew = ckpt.get("best_hider_rew", -float("inf"))
            best_seeker_rew = ckpt.get("best_seeker_rew", -float("inf"))
            print(f"Treino retomado na iteracao {start_iteration}.")
        except Exception:
            print("Checkpoint antigo incompativel ignorado. Iniciando treino limpo.")

    try:
        obs = envs.reset()
        dones = np.zeros(num_envs, dtype=np.float32)

        lstm_h_hider = (torch.zeros(1, num_envs, 256).to(device), torch.zeros(1, num_envs, 256).to(device))
        lstm_h_seeker = (torch.zeros(1, num_envs, 256).to(device), torch.zeros(1, num_envs, 256).to(device))

        for iteration in range(start_iteration, num_iterations + 1):
            t_start = time.time()

            frac = 1.0 - (iteration - 1.0) / num_iterations
            lr_now = max(frac * base_lr, 1e-6)
            opt_hider.param_groups[0]["lr"] = lr_now
            opt_seeker.param_groups[0]["lr"] = lr_now

            obs_buf = np.zeros((steps_per_env, num_envs, obs_dim), dtype=np.float32)
            done_buf = np.zeros((steps_per_env, num_envs), dtype=np.float32)

            h_h_buf = np.zeros((steps_per_env, num_envs, 256), dtype=np.float32)
            c_h_buf = np.zeros((steps_per_env, num_envs, 256), dtype=np.float32)
            act_h_buf = np.zeros((steps_per_env, num_envs, act_dim_hider), dtype=np.float32)
            logp_h_buf = np.zeros((steps_per_env, num_envs), dtype=np.float32)
            val_h_buf = np.zeros((steps_per_env, num_envs), dtype=np.float32)
            rew_h_buf = np.zeros((steps_per_env, num_envs), dtype=np.float32)

            h_s_buf = np.zeros((steps_per_env, num_envs, 256), dtype=np.float32)
            c_s_buf = np.zeros((steps_per_env, num_envs, 256), dtype=np.float32)
            act_s_buf = np.zeros((steps_per_env, num_envs, act_dim_seeker), dtype=np.float32)
            logp_s_buf = np.zeros((steps_per_env, num_envs), dtype=np.float32)
            val_s_buf = np.zeros((steps_per_env, num_envs), dtype=np.float32)
            rew_s_buf = np.zeros((steps_per_env, num_envs), dtype=np.float32)

            for step in range(steps_per_env):
                obs_buf[step] = obs
                done_buf[step] = dones

                h_h_buf[step] = lstm_h_hider[0].squeeze(0).cpu().numpy()
                c_h_buf[step] = lstm_h_hider[1].squeeze(0).cpu().numpy()
                h_s_buf[step] = lstm_h_seeker[0].squeeze(0).cpu().numpy()
                c_s_buf[step] = lstm_h_seeker[1].squeeze(0).cpu().numpy()

                obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                dones_t = torch.tensor(dones, dtype=torch.float32).to(device)

                with torch.no_grad():
                    a_h, lp_h, _, v_h, lstm_h_hider = agent_hider.get_action_and_value(obs_t, lstm_h_hider, dones_t)
                    a_s, lp_s, _, v_s, lstm_h_seeker = agent_seeker.get_action_and_value(obs_t, lstm_h_seeker, dones_t)

                act_h = a_h.cpu().numpy()
                act_s = a_s.cpu().numpy()

                act_h_buf[step] = act_h
                logp_h_buf[step] = lp_h.cpu().numpy()
                val_h_buf[step] = v_h.cpu().numpy().squeeze(-1)

                act_s_buf[step] = act_s
                logp_s_buf[step] = lp_s.cpu().numpy()
                val_s_buf[step] = v_s.cpu().numpy().squeeze(-1)

                obs, r_h, r_s, dones = envs.step(act_h, act_s)

                rew_h_buf[step] = r_h
                rew_s_buf[step] = r_s

            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            dones_t = torch.tensor(dones, dtype=torch.float32).to(device)
            with torch.no_grad():
                next_v_h = agent_hider.get_value(obs_t, lstm_h_hider, dones_t).cpu().numpy().squeeze(-1)
                next_v_s = agent_seeker.get_value(obs_t, lstm_h_seeker, dones_t).cpu().numpy().squeeze(-1)

            def compute_gae(rew_b, val_b, next_v):
                adv = np.zeros((steps_per_env, num_envs), dtype=np.float32)
                lastg = np.zeros(num_envs, dtype=np.float32)
                for t in reversed(range(steps_per_env)):
                    nxt_nonterm = 1.0 - dones if t == steps_per_env - 1 else 1.0 - done_buf[t + 1]
                    nxt_val = next_v if t == steps_per_env - 1 else val_b[t + 1]
                    delta = rew_b[t] + gamma * nxt_val * nxt_nonterm - val_b[t]
                    adv[t] = lastg = delta + gamma * gae_lambda * nxt_nonterm * lastg
                return adv, adv + val_b

            adv_h, ret_h = compute_gae(rew_h_buf, val_h_buf, next_v_h)
            adv_s, ret_s = compute_gae(rew_s_buf, val_s_buf, next_v_s)

            t_obs_b = torch.tensor(obs_buf, dtype=torch.float32).to(device)
            t_done_b = torch.tensor(done_buf, dtype=torch.float32).to(device)

            train_ppo_tbptt_amp(
                agent_hider, opt_hider, scaler_hider,
                t_obs_b,
                torch.tensor(act_h_buf, dtype=torch.float32).to(device),
                torch.tensor(logp_h_buf, dtype=torch.float32).to(device),
                torch.tensor(adv_h, dtype=torch.float32).to(device),
                torch.tensor(ret_h, dtype=torch.float32).to(device),
                torch.tensor(val_h_buf, dtype=torch.float32).to(device),
                t_done_b,
                torch.tensor(h_h_buf, dtype=torch.float32).to(device),
                torch.tensor(c_h_buf, dtype=torch.float32).to(device),
                steps_per_env, num_envs, seq_len, act_dim_hider, num_minibatches, update_epochs,
                clip_coef, vf_coef, ent_coef
            )

            train_ppo_tbptt_amp(
                agent_seeker, opt_seeker, scaler_seeker,
                t_obs_b,
                torch.tensor(act_s_buf, dtype=torch.float32).to(device),
                torch.tensor(logp_s_buf, dtype=torch.float32).to(device),
                torch.tensor(adv_s, dtype=torch.float32).to(device),
                torch.tensor(ret_s, dtype=torch.float32).to(device),
                torch.tensor(val_s_buf, dtype=torch.float32).to(device),
                t_done_b,
                torch.tensor(h_s_buf, dtype=torch.float32).to(device),
                torch.tensor(c_s_buf, dtype=torch.float32).to(device),
                steps_per_env, num_envs, seq_len, act_dim_seeker, num_minibatches, update_epochs,
                clip_coef, vf_coef, ent_coef
            )

            m_h_rew = float(np.mean(rew_h_buf))
            m_s_rew = float(np.mean(rew_s_buf))
            elapsed = time.time() - t_start

            print(f"Iter {iteration}/{num_iterations} | H: {m_h_rew:.2f} | S: {m_s_rew:.2f} | lr: {lr_now:.2e} | Tempo: {elapsed:.2f}s")

            if m_h_rew > best_hider_rew:
                best_hider_rew = m_h_rew
                torch.save(agent_hider.state_dict(), "./checkpoints/melhor_hider.pt")

            if m_s_rew > best_seeker_rew:
                best_seeker_rew = m_s_rew
                torch.save(agent_seeker.state_dict(), "./checkpoints/melhor_seeker.pt")

            checkpoint_data = {
                "iteration": iteration,
                "agent_hider": agent_hider.state_dict(),
                "agent_seeker": agent_seeker.state_dict(),
                "opt_hider": opt_hider.state_dict(),
                "opt_seeker": opt_seeker.state_dict(),
                "scaler_hider": scaler_hider.state_dict(),
                "scaler_seeker": scaler_seeker.state_dict(),
                "best_hider_rew": best_hider_rew,
                "best_seeker_rew": best_seeker_rew,
            }
            torch.save(checkpoint_data, checkpoint_recente)

            if iteration % 250 == 0 or iteration == num_iterations:
                torch.save(checkpoint_data, f"./checkpoints/ckpt_iter_{iteration}.pt")

    finally:
        print("Fechando ambientes em paralelo...")
        envs.close()
        print("Treino finalizado com seguranca.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()