import os
import time
import numpy as np
import torch
from mae_envs.envs.hide_and_seek import make_env
from treinamento import (
    carregar_dlls,
    get_space_info,
    get_team_indices,
    ActorCriticLSTM,
)

carregar_dlls()

def flatten_obs(obs):
    if isinstance(obs, dict):
        arrays = [flatten_obs(obs[k]) for k in sorted(obs.keys())]
        res = np.concatenate(arrays).astype(np.float32)
    elif isinstance(obs, (list, tuple)):
        arrays = [flatten_obs(item) for item in obs]
        res = np.concatenate(arrays).astype(np.float32)
    else:
        res = np.asarray(obs, dtype=np.float32).flatten()
    return np.nan_to_num(res, nan=0.0, posinf=10.0, neginf=-10.0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "./checkpoints/checkpoint_recente.pt"

    if not os.path.exists(checkpoint_path):
        print("Checkpoint nao encontrado.")
        return

    n_hiders, n_seekers, n_boxes, n_ramps = 2, 2, 2, 1
    frame_skip = 4

    env = make_env(n_hiders=n_hiders, n_seekers=n_seekers, n_boxes=n_boxes, n_ramps=n_ramps)

    # Desativa prep_phase dinamicamente sem import externo
    curr = env
    while curr is not None:
        if hasattr(curr, "in_prep_phase"):
            type(curr).in_prep_phase = property(lambda self: False, lambda self, v: None)
        curr = getattr(curr, "env", None)

    raw_obs = env.reset()
    obs = raw_obs[0] if isinstance(raw_obs, tuple) else raw_obs

    obs_dim = flatten_obs(obs).shape[0]
    total_size, lows, highs, neutrals, unflatten_action = get_space_info(env.action_space)
    hider_idx, seeker_idx = get_team_indices(env.action_space, n_hiders, n_seekers)

    mids = (lows + highs) / 2.0
    spans = (highs - lows) / 2.0

    agent_hider = ActorCriticLSTM(obs_dim, len(hider_idx)).to(device)
    agent_seeker = ActorCriticLSTM(obs_dim, len(seeker_idx)).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    agent_hider.load_state_dict(ckpt["agent_hider"])
    agent_seeker.load_state_dict(ckpt["agent_seeker"])
    print(f"Visualizando iteracao {ckpt.get('iteration', 0)}...")

    agent_hider.eval()
    agent_seeker.eval()

    for ep in range(1, 6):
        raw_obs = env.reset()
        obs = raw_obs[0] if isinstance(raw_obs, tuple) else raw_obs

        lstm_h = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
        lstm_s = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
        done_t = torch.zeros(1, dtype=torch.float32).to(device)

        tot_h, tot_s = 0.0, 0.0

        while True:
            env.render()
            time.sleep(0.02)

            obs_t = torch.tensor(flatten_obs(obs), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                a_h, _, _, _, lstm_h = agent_hider.get_action_and_value(obs_t, lstm_h, done_t)
                a_s, _, _, _, lstm_s = agent_seeker.get_action_and_value(obs_t, lstm_s, done_t)

            raw_action = np.zeros(total_size, dtype=np.float32)
            raw_action[hider_idx] = a_h.cpu().numpy()[0]
            raw_action[seeker_idx] = a_s.cpu().numpy()[0]

            # Quebra zona morta do neutro: forca direcao ativa
            movimento_ativo = np.where(raw_action > 0.05, highs, np.where(raw_action < -0.05, lows, mids))
            mapped_action = np.clip(movimento_ativo, lows, highs)
            action_orig = unflatten_action(mapped_action)

            fim = False
            for _ in range(frame_skip):
                action_orig = {k: np.array(v) for k, v in action_orig.items()}
                res = env.step(action_orig)
                obs, rew, done = res[0], res[1], res[2]
                truncated = res[3] if len(res) > 4 else False

                rew_arr = np.asarray(rew, dtype=np.float32).flatten()
                r_h = float(np.mean(rew_arr[:n_hiders])) if len(rew_arr) >= 4 else float(rew_arr[0])
                r_s = float(np.mean(rew_arr[n_hiders:])) if len(rew_arr) >= 4 else -float(rew_arr[0])

                tot_h += r_h
                tot_s += r_s

                d_b = bool(done.get("__all__", any(done.values()))) if isinstance(done, dict) else bool(done)
                t_b = bool(truncated.get("__all__", any(truncated.values()))) if isinstance(truncated, dict) else bool(truncated)
                if d_b or t_b:
                    fim = True
                    break

            if fim:
                print(f"Episodio {ep} | Hiders: {tot_h:.2f} | Seekers: {tot_s:.2f}")
                break

if __name__ == "__main__":
    main()