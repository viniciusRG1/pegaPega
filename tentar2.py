from mae_envs.envs.hide_and_seek import make_env

env = make_env(n_hiders=2, n_seekers=2, n_boxes=2, n_ramps=1)
sim = getattr(env, "sim", None) or getattr(getattr(env, "unwrapped", None), "sim", None)

print("Sim detectado:", sim is not None)
if sim is not None:
    nomes_agentes = [n for n in sim.model.body_names if "agent" in n.lower()]
    print("Nomes dos corpos dos agentes:", nomes_agentes)