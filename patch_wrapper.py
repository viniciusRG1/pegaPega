import os
import mujoco_py

pkg_dir = os.path.dirname(mujoco_py.__file__)
wrapper_path = os.path.join(pkg_dir, "wrappers.pxi")

with open(wrapper_path, "r", encoding="utf-8") as f:
    txt = f.read()

txt = txt.replace(
    "if isinstance(addr, int):",
    "if not hasattr(addr, '__iter__'):"
)

with open(wrapper_path, "w", encoding="utf-8") as f:
    f.write(txt)

print("wrappers.pxi corrigido!")

bp = os.path.join("mujoco-worldgen", "mujoco_worldgen", "mujoco-worldgen/mujoco_worldgen/builder.py")
with open(bp, "r", encoding="utf-8") as f:
    btxt = f.read()

btxt = btxt.replace("np.atleast_1d(value)", "value")

with open(bp, "w", encoding="utf-8") as f:
    f.write(btxt)

print("builder.py restaurado!")