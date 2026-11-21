import os
import tempfile

path = "mujoco-worldgen/mujoco_worldgen/builder.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Atualiza importações
code = code.replace(
    "from mujoco_py import load_model_from_xml",
    "from mujoco_py import load_model_from_xml, load_model_from_path\nimport os\nimport tempfile",
)

# Substitui carregamento travado
old = "model = load_model_from_xml"
for line in code.splitlines():
    if old in line:
        old_line = line
        break

new_block = """        fd, tmp = tempfile.mkstemp(suffix='.xml')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(xml.replace(chr(92), '/'))
            model = load_model_from_path(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)"""

code = code.replace(old_line, new_block)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("Patch aplicado com sucesso!")