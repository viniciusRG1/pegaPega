"""Grava episodios de um checkpoint em video (MP4) ou GIF, sem abrir janela.

Com --melhores N --destaque rampa ficam os episodios em que os pegadores usaram a rampa.

Exemplos:
    python gravar_video.py                                   # modelo de demonstracao -> videos/
    python gravar_video.py --ckpt checkpoints_final/fase_3_barricada.pt --episodios 3 --gif
    python gravar_video.py --melhores 3 --de 20              # joga 20 episodios e guarda os 3 melhores p/ escondedores

Precisa do pacote oficial `mujoco` (o mesmo usado pela camada de compatibilidade).
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import preparar_ambiente  # noqa: F401
import mujoco

from avaliar import MODELO_DEMO, carregar_agentes, indice_fase, jogar_episodio
from treinamento import CURRICULO, TIMES, ambiente_da_fase, mascara_acoes


def _fonte(tamanho):
    for caminho in ("/System/Library/Fonts/Helvetica.ttc", "C:/Windows/Fonts/arial.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(caminho):
            try:
                return ImageFont.truetype(caminho, tamanho)
            except OSError:
                pass
    return ImageFont.load_default()


class Filmadora:
    """Renderiza a cena vista de cima e escreve por cima o estado do jogo."""

    def __init__(self, largura=720, altura=720):
        self.largura, self.altura = largura, altura
        self.quadros = []
        self._modelo = None
        self._r = None
        self.cam = mujoco.MjvCamera()
        self.fonte, self.fonte_p = _fonte(22), _fonte(16)

    def _preparar(self, modelo, tamanho_chao):
        if modelo is self._modelo:
            return
        if self._r is not None:
            self._r.close()
        modelo.vis.global_.offwidth = max(int(modelo.vis.global_.offwidth), self.largura)
        modelo.vis.global_.offheight = max(int(modelo.vis.global_.offheight), self.altura)
        # So aparencia: luz mais forte. Nao altera a fisica nem a visao dos agentes.
        modelo.vis.headlight.ambient[:] = 0.45
        modelo.vis.headlight.diffuse[:] = 0.65
        self._r = mujoco.Renderer(modelo, height=self.altura, width=self.largura)
        self._modelo = modelo
        # As paredes externas tem 4 m de altura e tapariam a camera: sao escondidas so na imagem.
        self._externas = {i for i in range(modelo.ngeom)
                          if modelo.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX and modelo.geom_size[i][2] > 1.0}
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = (tamanho_chao / 2, tamanho_chao / 2 - 0.25, 0.0)
        self.cam.distance = tamanho_chao * 1.5
        self.cam.elevation = -68.0
        self.cam.azimuth = 90.0

    def quadro(self, env, recompensa):
        u = env._env.unwrapped
        sim = u.sim
        if not hasattr(sim.model, "_m"):
            raise RuntimeError("A gravacao de video usa o pacote oficial `mujoco` "
                               "(rode com PEGAPEGA_MUJOCO=compat).")
        self._preparar(sim.model._m, u.floor_size)
        self._r.update_scene(sim.data._d, camera=self.cam)
        cena = self._r.scene
        for i in range(cena.ngeom):
            g = cena.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in self._externas:
                g.rgba[3] = 0.0
        img = Image.fromarray(self._r.render())
        d = ImageDraw.Draw(img)
        passo, total = u.t, u.horizon
        if env.em_prep:
            titulo, cor = "PREPARACAO  (pegadores parados)", (255, 215, 90)
        elif recompensa[env.n_hiders:].max() > 0:
            titulo, cor = "JOGO  -  escondedor A VISTA", (255, 110, 110)
        else:
            titulo, cor = "JOGO  -  escondedores ESCONDIDOS", (120, 235, 245)
        d.rectangle([0, 0, self.largura, 44], fill=(20, 20, 24))
        d.text((14, 9), titulo, fill=cor, font=self.fonte)
        d.text((self.largura - 120, 13), f"passo {passo:3d}/{total}", fill=(220, 220, 220), font=self.fonte_p)
        d.rectangle([0, self.altura - 30, self.largura, self.altura], fill=(20, 20, 24))
        rodape = f"verde = escondedores   vermelho = pegadores   portas vedadas: {int(env.pot['_n_vedadas'])}"
        if env.pot["_travada"]:
            rodape += "   rampa travada"
        d.text((14, self.altura - 25), rodape, fill=(220, 220, 220), font=self.fonte_p)
        self.quadros.append(np.asarray(img))

    def fechar(self):
        if self._r is not None:
            self._r.close()
            self._r = None


def salvar(quadros, caminho, fps):
    import imageio
    if caminho.endswith(".gif"):
        pequenos = [np.asarray(Image.fromarray(q).resize((q.shape[1] // 2, q.shape[0] // 2))) for q in quadros[::2]]
        imageio.mimsave(caminho, pequenos, duration=2.0 / fps, loop=0)
    else:
        imageio.mimsave(caminho, quadros, fps=fps, macro_block_size=8, quality=7)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=MODELO_DEMO)
    ap.add_argument("--saida", default="./videos")
    ap.add_argument("--nome", default=None, help="prefixo dos arquivos (padrao: nome do checkpoint)")
    ap.add_argument("--episodios", type=int, default=3)
    ap.add_argument("--melhores", type=int, default=None,
                    help="guarda so os N episodios em que os escondedores ficaram menos tempo a vista")
    ap.add_argument("--de", type=int, default=20, help="quantos episodios jogar ao usar --melhores")
    ap.add_argument("--destaque", choices=("escondedores", "rampa", "defesa"), default="escondedores",
                    help="criterio do --melhores: escondedores menos vistos, pegadores usando a rampa, ou "
                         "escondedores que travam a rampa e vedam a sala")
    ap.add_argument("--fase", type=int, default=None, help="usa as regras desta fase do curriculo")
    ap.add_argument("--horizonte", type=int, default=None, help="passos do simulador por episodio")
    ap.add_argument("--det", action="store_true", help="acao mais provavel em vez de amostrar")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--gif", action="store_true", help="grava GIF em vez de MP4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--tamanho", type=int, default=720)
    args = ap.parse_args()

    device = torch.device("cpu")
    agentes, estado, cfg_amb = carregar_agentes(args.ckpt, device)
    fase = CURRICULO[indice_fase(estado, args.fase)]
    np.random.seed(args.semente)
    torch.manual_seed(args.semente)
    if args.horizonte:
        cfg_amb, fase = dict(cfg_amb, horizon=args.horizonte), dict(fase, horizonte=None)
    env = ambiente_da_fase(cfg_amb, fase)
    mascaras = {t: mascara_acoes(env.nvec, env.grupos, env.neutro, fase["acoes"][t]).to(device) for t in TIMES}
    os.makedirs(args.saida, exist_ok=True)
    prefixo = args.nome or os.path.splitext(os.path.basename(args.ckpt))[0]
    ext = ".gif" if args.gif else ".mp4"

    filmadora = Filmadora(args.tamanho, args.tamanho)
    env.ao_quadro = filmadora.quadro
    gravados = []
    try:
        for _ in range(args.de if args.melhores else args.episodios):
            filmadora.quadros = []
            info = jogar_episodio(env, agentes, mascaras, device, args.det)
            gravados.append((info, filmadora.quadros))
            if args.melhores:
                if args.destaque == "rampa":
                    def chave(g):
                        return (-g[0]["subiu"], -g[0]["invasao"], -g[0]["dentro"])
                elif args.destaque == "defesa":
                    def chave(g):
                        return (-g[0]["rampa_travada"], -g[0]["selada"], g[0]["visto"])
                else:
                    def chave(g):
                        return (-g[0]["selada"], g[0]["visto"])
                gravados = sorted(gravados, key=chave)[:args.melhores]
    finally:
        filmadora.fechar()
        env.close()

    for i, (info, quadros) in enumerate(gravados, 1):
        caminho = os.path.join(args.saida, f"{prefixo}_ep{i}{ext}")
        salvar(quadros, caminho, args.fps)
        print(f"{caminho}: portas vedadas {info['vedadas']:.0f}, sala vedada a tempo: {'sim' if info['selada'] else 'nao'}, "
              f"rampa travada: {'sim' if info['rampa_travada'] else 'nao'}, a vista {100 * info['visto']:.0f}% do tempo, "
              f"pegador subiu na rampa: {'sim' if info['subiu'] else 'nao'}, "
              f"pegador dentro da sala {100 * info['dentro']:.0f}% do tempo")


if __name__ == "__main__":
    main()
