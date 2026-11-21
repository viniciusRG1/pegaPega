import os
import sys
import glob
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

import numpy as np
import torch
import torch.nn as nn
from mae_envs.envs.hide_and_seek import make_env

def get_space_info(space):
    if hasattr(space, "spaces"):
        if isinstance(space.spaces, dict):
            keys = sorted(space.spaces.keys())
            sub_infos = {k: get_space_info(space.spaces[k]) for k in keys}
            total_size = sum(info[0] for info in sub_infos.values())
            lows = np.concatenate([info[1] for info in sub_infos.values()]) if total_size > 0 else np.array([], dtype=np.float32)
            highs = np.concatenate([info[2] for info in sub_infos.values()]) if total_size > 0 else np.array([], dtype=np.float32)
            def unflatten_dict(x):
                out = {}
                idx = 0
                for k in keys:
                    sz, _, _, fn = sub_infos[k]
                    out[k] = fn(x[idx:idx + sz])
                    idx += sz
                return out
            return total_size, lows, highs, unflatten_dict
        else:
            sub_infos = [get_space_info(sub_s) for sub_s in space.spaces]
            total_size = sum(info[0] for info in sub_infos)
            lows = np.concatenate([info[1] for info in sub_infos]) if total_size > 0 else np.array([], dtype=np.float32)
            highs = np.concatenate([info[2] for info in sub_infos]) if total_size > 0 else np.array([], dtype=np.float32)
            def unflatten_tuple(x):
                out = []
                idx = 0
                for sz, _, _, fn in sub_infos:
                    out.append(fn(x[idx:idx + sz]))
                    idx += sz
                return tuple(out)
            return total_size, lows, highs, unflatten_tuple
    elif type(space).__name__ == "Discrete":
        n = int(space.n)
        return 1, np.array([0.0], dtype=np.float32), np.array([float(n - 1)], dtype=np.float32), lambda x: int(np.clip(np.round(x[0]), 0, n - 1))
    elif type(space).__name__ == "MultiDiscrete" or hasattr(space, "nvec"):
        nvec = np.array(space.nvec, dtype=np.int64)
        sz = len(nvec)
        return sz, np.zeros(sz, dtype=np.float32), (nvec - 1).astype(np.float32), lambda x: np.clip(np.round(x), 0, nvec - 1).astype(np.int64)
    elif type(space).__name__ == "MultiBinary":
        n = int(space.n) if hasattr(space, "n") else int(np.prod(space.shape))
        return n, np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32), lambda x: np.clip(np.round(x), 0, 1).astype(np.int8)
    else:
        shp = getattr(space, "shape", None)
        if shp is None or len(shp) == 0:
            l = np.asarray(getattr(space, "low", -1.0), dtype=np.float32).flatten()
            h = np.asarray(getattr(space, "high", 1.0), dtype=np.float32).flatten()
            sz = len(l) if len(l) > 0 else 1
            return sz, l, h, lambda x: x[0] if sz == 1 else x
        else:
            sz = int(np.prod(shp))
            l = np.asarray(space.low, dtype=np.float32).flatten()
            h = np.asarray(space.high, dtype=np.float32).flatten()
            return sz, l, h, lambda x: x.reshape(shp)

def flatten_obs(obs):
    if isinstance(obs, dict):
        arrays = [flatten_obs(obs[k]) for k in sorted(obs.keys())]
        return np.concatenate(arrays).astype(np.float32)
    elif isinstance(obs, (list, tuple)):
        arrays = [flatten_obs(item) for item in obs]
        return np.concatenate(arrays).astype(np.float32)
    return np.asarray(obs, dtype=np.float32).flatten()

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

env = make_env()
raw_obs = env.reset()
if isinstance(raw_obs, tuple):
    raw_obs = raw_obs[0]

flat_obs = flatten_obs(raw_obs)
obs_dim = flat_obs.shape[0]

total_size, low, high, unflatten_action = get_space_info(env.action_space)
act_dim = total_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = ActorCritic(obs_dim, act_dim).to(device)

checkpoints = sorted(glob.glob("./checkpoints/ppo_iter_*.pt"), key=os.path.getmtime)
if checkpoints:
    checkpoint_path = checkpoints[-1]
    print(f"Carregando modelo: {checkpoint_path}")
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()
else:
    print("Nenhum checkpoint encontrado.")
    sys.exit()

obs = raw_obs
for step in range(1000):
    flat_o = flatten_obs(obs)
    obs_t = torch.tensor(flat_o, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        action_mean = agent.actor_mean(obs_t)

    action_np = action_mean.cpu().numpy()[0]
    action_np = np.clip(action_np, low, high)
    action_orig = unflatten_action(action_np)

    res = env.step(action_orig)
    if len(res) == 4:
        obs, recompensa, fim, info = res
    else:
        obs, recompensa, done, trunc, info = res
        fim = done or trunc

    env.render()
    time.sleep(0.02)

    if fim:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

env.close()