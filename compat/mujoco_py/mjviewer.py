"""Janela interativa minima, no lugar do `mujoco_py.MjViewer`.

Botao esquerdo gira a camera, botao direito desloca, a rolagem aproxima. ESC fecha.
"""
import glfw
import mujoco
import numpy as np


class MjViewer:
    def __init__(self, sim, largura=1000, altura=800, titulo="Pega-pega"):
        if not glfw.init():
            raise RuntimeError("Nao foi possivel iniciar o GLFW (sem janela grafica?).")
        self.window = glfw.create_window(largura, altura, titulo, None, None)
        if not self.window:
            raise RuntimeError("Nao foi possivel abrir a janela de visualizacao.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self._overlay = {}
        self._ultimo = None
        self._botao = None
        glfw.set_cursor_pos_callback(self.window, self._ao_mover)
        glfw.set_mouse_button_callback(self.window, self._ao_clicar)
        glfw.set_scroll_callback(self.window, self._ao_rolar)
        glfw.set_key_callback(self.window, self._ao_teclar)
        self.sim = None
        self.update_sim(sim)

    # --- troca de simulacao (a cada reset o worldgen cria um modelo novo) ----
    def update_sim(self, sim):
        self.sim = sim
        m = sim.model._m
        self.scn = mujoco.MjvScene(m, maxgeom=10000)
        self.ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        if self._ultimo is None:
            mujoco.mjv_defaultFreeCamera(m, self.cam)
            self.cam.elevation = -60.0
            self.cam.azimuth = 90.0
            self._ultimo = (0.0, 0.0)

    def add_overlay(self, gridpos, texto1, texto2):
        t1, t2 = self._overlay.get(gridpos, ("", ""))
        self._overlay[gridpos] = (t1 + texto1 + "\n", t2 + texto2 + "\n")

    def render(self):
        if self.window is None:
            return
        if glfw.window_should_close(self.window):
            self.close()
            raise KeyboardInterrupt
        glfw.make_context_current(self.window)
        m, d = self.sim.model._m, self.sim.data._d
        w, h = glfw.get_framebuffer_size(self.window)
        area = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(m, d, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(area, self.scn, self.ctx)
        for pos, (t1, t2) in self._overlay.items():
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, pos, area, t1, t2, self.ctx)
        self._overlay.clear()
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None

    # --- mouse / teclado -----------------------------------------------------
    def _ao_clicar(self, janela, botao, acao, mods):
        self._botao = botao if acao == glfw.PRESS else None
        self._ultimo = glfw.get_cursor_pos(janela)

    def _ao_mover(self, janela, x, y):
        if self._botao is None:
            return
        dx, dy = x - self._ultimo[0], y - self._ultimo[1]
        self._ultimo = (x, y)
        _, h = glfw.get_window_size(janela)
        acao = (mujoco.mjtMouse.mjMOUSE_ROTATE_V if self._botao == glfw.MOUSE_BUTTON_LEFT
                else mujoco.mjtMouse.mjMOUSE_MOVE_V)
        mujoco.mjv_moveCamera(self.sim.model._m, acao, dx / max(h, 1), dy / max(h, 1), self.scn, self.cam)

    def _ao_rolar(self, janela, dx, dy):
        mujoco.mjv_moveCamera(self.sim.model._m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * dy,
                              self.scn, self.cam)

    def _ao_teclar(self, janela, tecla, codigo, acao, mods):
        if tecla == glfw.KEY_ESCAPE and acao == glfw.PRESS:
            glfw.set_window_should_close(janela, True)


__all__ = ["MjViewer", "np"]
