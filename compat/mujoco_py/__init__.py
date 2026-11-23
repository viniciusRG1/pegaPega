"""Camada de compatibilidade: a API do `mujoco_py` em cima do pacote oficial `mujoco`.

O `mujoco_py` (usado pelo mujoco-worldgen e pelo multi-agent-emergence-environments)
foi descontinuado e nao tem build para Apple Silicon nem para Python recente. Este
modulo reimplementa so o pedaco da API que esses dois projetos usam, delegando para o
binding oficial (`pip install mujoco`).

Ele so entra em cena quando o `mujoco_py` de verdade nao esta instalado: quem decide
isso e o `preparar_ambiente.py`, que poe esta pasta no `sys.path` como plano B.
"""
import ctypes
import os
import re
import tempfile
from collections import namedtuple

import mujoco
import numpy as np

__version__ = "compat-" + mujoco.__version__

_OBJ = mujoco.mjtObj
_AVISOS_FATAIS = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                  mujoco.mjtWarning.mjWARN_BADQACC)


class MujocoException(Exception):
    pass


class _Const:
    """Constantes que o codigo antigo le de `mujoco_py.const`."""
    EQ_CONNECT = int(mujoco.mjtEq.mjEQ_CONNECT)
    EQ_WELD = int(mujoco.mjtEq.mjEQ_WELD)
    EQ_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)
    GRID_TOPLEFT = int(mujoco.mjtGridPos.mjGRID_TOPLEFT)
    GRID_TOPRIGHT = int(mujoco.mjtGridPos.mjGRID_TOPRIGHT)
    GRID_BOTTOMLEFT = int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT)
    GRID_BOTTOMRIGHT = int(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT)
    CAMERA_FREE = int(mujoco.mjtCamera.mjCAMERA_FREE)
    CAMERA_TRACKING = int(mujoco.mjtCamera.mjCAMERA_TRACKING)
    CAMERA_FIXED = int(mujoco.mjtCamera.mjCAMERA_FIXED)


const = _Const()

MjSimState = namedtuple("MjSimState", ["time", "qpos", "qvel", "act", "udd_state"])


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def _adaptar_xml(xml):
    """Ajusta o MJCF gerado pelo worldgen (escrito para o MuJoCo 2.1) ao MuJoCo atual.

    No MuJoCo 3 o `autolimits` passou a valer por padrao: qualquer junta com `range`
    viraria limitada. O mecanismo de "trancar" caixas depende de as juntas nascerem
    livres e so serem limitadas em tempo de execucao, entao o padrao antigo e restaurado.
    """
    if "autolimits" in xml:
        return xml
    if re.search(r"<compiler\b", xml):
        return re.sub(r"<compiler\b", '<compiler autolimits="false"', xml, count=1)
    return re.sub(r"(<mujoco\b[^>]*>)", r'\1<compiler autolimits="false"/>', xml, count=1)


def load_model_from_xml(xml):
    return PyMjModel(mujoco.MjModel.from_xml_string(_adaptar_xml(xml)))


def load_model_from_path(path):
    with open(path, "r", encoding="utf-8") as f:
        xml = f.read()
    # Os assets usam caminho absoluto; carregar de um arquivo na mesma pasta mantem
    # validos tambem os eventuais caminhos relativos.
    fd, tmp = tempfile.mkstemp(suffix=".xml", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_adaptar_xml(xml))
        return PyMjModel(mujoco.MjModel.from_xml_path(tmp))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_model_from_mjb(path):
    return PyMjModel(mujoco.MjModel.from_binary_path(path))


class PyMjModel:
    """Embrulha `mujoco.MjModel` expondo os nomes de atributos do mujoco_py."""

    def __init__(self, modelo):
        object.__setattr__(self, "_m", modelo)
        object.__setattr__(self, "_d", None)
        object.__setattr__(self, "_nomes", {})

    def __getattr__(self, nome):
        return getattr(object.__getattribute__(self, "_m"), nome)

    def __setattr__(self, nome, valor):
        setattr(self._m, nome, valor)

    # --- nomes <-> ids -----------------------------------------------------
    def _lista(self, tipo, n):
        if tipo not in self._nomes:
            self._nomes[tipo] = tuple(mujoco.mj_id2name(self._m, tipo, i) for i in range(n))
        return self._nomes[tipo]

    def _id(self, tipo, nome, rotulo):
        i = mujoco.mj_name2id(self._m, tipo, nome)
        if i < 0:
            raise ValueError(f'No "{rotulo}" with name {nome} exists.')
        return i

    body_names = property(lambda s: s._lista(_OBJ.mjOBJ_BODY, s._m.nbody))
    joint_names = property(lambda s: s._lista(_OBJ.mjOBJ_JOINT, s._m.njnt))
    geom_names = property(lambda s: s._lista(_OBJ.mjOBJ_GEOM, s._m.ngeom))
    site_names = property(lambda s: s._lista(_OBJ.mjOBJ_SITE, s._m.nsite))
    camera_names = property(lambda s: s._lista(_OBJ.mjOBJ_CAMERA, s._m.ncam))
    actuator_names = property(lambda s: s._lista(_OBJ.mjOBJ_ACTUATOR, s._m.nu))

    def body_name2id(self, nome):
        return self._id(_OBJ.mjOBJ_BODY, nome, "body")

    def joint_name2id(self, nome):
        return self._id(_OBJ.mjOBJ_JOINT, nome, "joint")

    def geom_name2id(self, nome):
        return self._id(_OBJ.mjOBJ_GEOM, nome, "geom")

    def site_name2id(self, nome):
        return self._id(_OBJ.mjOBJ_SITE, nome, "site")

    def camera_name2id(self, nome):
        return self._id(_OBJ.mjOBJ_CAMERA, nome, "camera")

    def actuator_name2id(self, nome):
        return self._id(_OBJ.mjOBJ_ACTUATOR, nome, "actuator")

    def body_id2name(self, i):
        return mujoco.mj_id2name(self._m, _OBJ.mjOBJ_BODY, int(i))

    def joint_id2name(self, i):
        return mujoco.mj_id2name(self._m, _OBJ.mjOBJ_JOINT, int(i))

    def geom_id2name(self, i):
        return mujoco.mj_id2name(self._m, _OBJ.mjOBJ_GEOM, int(i))

    def site_id2name(self, i):
        return mujoco.mj_id2name(self._m, _OBJ.mjOBJ_SITE, int(i))

    # --- enderecos de juntas -----------------------------------------------
    def _endereco(self, nome, enderecos, tamanhos):
        j = self.joint_name2id(nome)
        ini = int(enderecos[j])
        tam = tamanhos[int(self._m.jnt_type[j])]
        return ini if tam == 1 else (ini, ini + tam)

    def get_joint_qpos_addr(self, nome):
        return self._endereco(nome, self._m.jnt_qposadr, (7, 4, 1, 1))

    def get_joint_qvel_addr(self, nome):
        return self._endereco(nome, self._m.jnt_dofadr, (6, 3, 1, 1))

    # --- restricoes de igualdade (o "agarrar" usa soldas) -------------------
    @property
    def eq_active(self):
        # MuJoCo 3 moveu o estado ativo das restricoes do modelo para os dados.
        if hasattr(self._m, "eq_active"):
            return self._m.eq_active
        if self._d is None:
            return self._m.eq_active0
        return self._d.eq_active

    @property
    def eq_data(self):
        # MuJoCo 2.1: solda = [posicao relativa (3), quaternio (4)].
        # MuJoCo atual: [ancora (3), posicao relativa (3), quaternio (4), escala de torque, ...].
        # A fatia abaixo devolve a mesma vista de 7 colunas que o codigo antigo espera.
        dados = self._m.eq_data
        return dados[:, 3:10] if dados.shape[1] >= 11 else dados

    # --- serializacao --------------------------------------------------------
    def get_xml(self):
        fd, tmp = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        try:
            mujoco.mj_saveLastXML(tmp, self._m)
            with open(tmp, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            os.remove(tmp)

    def get_mjb(self):
        buf = np.zeros(mujoco.mj_sizeModel(self._m), np.uint8)
        mujoco.mj_saveModel(self._m, None, buf)
        return buf.tobytes()


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
class PyMjData:
    """Embrulha `mujoco.MjData` expondo os nomes de atributos do mujoco_py."""

    _APELIDOS = {"body_xpos": "xpos", "body_xquat": "xquat", "body_xmat": "xmat",
                 "body_xvelp": "cvel", "body_xipos": "xipos"}

    def __init__(self, modelo, dados):
        object.__setattr__(self, "_model", modelo)
        object.__setattr__(self, "_d", dados)

    def __getattr__(self, nome):
        d = object.__getattribute__(self, "_d")
        return getattr(d, PyMjData._APELIDOS.get(nome, nome))

    def __setattr__(self, nome, valor):
        setattr(self._d, PyMjData._APELIDOS.get(nome, nome), valor)

    def get_body_xpos(self, nome):
        return self._d.xpos[self._model.body_name2id(nome)]

    def get_body_xquat(self, nome):
        return self._d.xquat[self._model.body_name2id(nome)]

    def get_body_xmat(self, nome):
        return self._d.xmat[self._model.body_name2id(nome)].reshape(3, 3)

    def get_geom_xpos(self, nome):
        return self._d.geom_xpos[self._model.geom_name2id(nome)]

    def get_site_xpos(self, nome):
        return self._d.site_xpos[self._model.site_name2id(nome)]

    def get_site_xmat(self, nome):
        return self._d.site_xmat[self._model.site_name2id(nome)].reshape(3, 3)

    def _fatia(self, endereco):
        return slice(*endereco) if isinstance(endereco, tuple) else slice(endereco, endereco + 1)

    def get_joint_qpos(self, nome):
        end = self._model.get_joint_qpos_addr(nome)
        return self._d.qpos[slice(*end)] if isinstance(end, tuple) else self._d.qpos[end]

    def set_joint_qpos(self, nome, valor):
        self._d.qpos[self._fatia(self._model.get_joint_qpos_addr(nome))] = valor

    def get_joint_qvel(self, nome):
        end = self._model.get_joint_qvel_addr(nome)
        return self._d.qvel[slice(*end)] if isinstance(end, tuple) else self._d.qvel[end]

    def set_joint_qvel(self, nome, valor):
        self._d.qvel[self._fatia(self._model.get_joint_qvel_addr(nome))] = valor


# ---------------------------------------------------------------------------
# Simulacao
# ---------------------------------------------------------------------------
class MjSim:
    def __init__(self, model, data=None, nsubsteps=1, udd_callback=None,
                 substep_callback=None, render_callbacks=None):
        self.model = model if isinstance(model, PyMjModel) else PyMjModel(model)
        dados = mujoco.MjData(self.model._m)
        object.__setattr__(self.model, "_d", dados)
        self.data = PyMjData(self.model, dados)
        self.nsubsteps = int(nsubsteps)
        self.render_contexts = []
        self.udd_state = None
        self._udd_callback = None
        self._renderizadores = {}
        self._avisos = self._contar_avisos()
        self.udd_callback = udd_callback
        mujoco.mj_forward(self.model._m, dados)

    # --- estado definido pelo usuario ----------------------------------------
    @property
    def udd_callback(self):
        return self._udd_callback

    @udd_callback.setter
    def udd_callback(self, valor):
        self._udd_callback = valor
        self.udd_state = None
        self.step_udd()

    def step_udd(self):
        if self._udd_callback is None:
            self.udd_state = {}
        else:
            self.udd_state = self._udd_callback(self)

    # --- passo ---------------------------------------------------------------
    def _contar_avisos(self):
        return [int(self.data._d.warning[int(a)].number) for a in _AVISOS_FATAIS]

    def step(self, with_udd=True):
        if with_udd:
            self.step_udd()
        m, d = self.model._m, self.data._d
        for _ in range(self.nsubsteps):
            mujoco.mj_step(m, d)
        avisos = self._contar_avisos()
        if avisos != self._avisos:
            self._avisos = avisos
            # O mujoco_py transformava os avisos de instabilidade em excecao.
            raise MujocoException("Simulacao instavel (qpos/qvel/qacc invalido).")

    def forward(self):
        mujoco.mj_forward(self.model._m, self.data._d)

    def reset(self):
        mujoco.mj_resetData(self.model._m, self.data._d)
        self._avisos = self._contar_avisos()
        self.udd_state = None
        self.step_udd()

    def get_state(self):
        d = self.data._d
        return MjSimState(float(d.time), d.qpos.copy(), d.qvel.copy(),
                          d.act.copy() if self.model._m.na else None, dict(self.udd_state or {}))

    def set_state(self, estado):
        d = self.data._d
        d.time = estado.time
        d.qpos[:] = estado.qpos
        d.qvel[:] = estado.qvel
        if self.model._m.na and estado.act is not None:
            d.act[:] = estado.act
        self.udd_state = dict(estado.udd_state or {})

    # --- imagem --------------------------------------------------------------
    def render(self, width=None, height=None, *, camera_name=None, depth=False,
               mode="offscreen", device_id=-1):
        width, height = int(width or 500), int(height or 500)
        m, d = self.model._m, self.data._d
        chave = (width, height)
        if chave not in self._renderizadores:
            m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)
            m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
            self._renderizadores[chave] = mujoco.Renderer(m, height=height, width=width)
        r = self._renderizadores[chave]
        camera = -1 if camera_name is None else self.model.camera_name2id(camera_name)
        r.update_scene(d, camera=camera)
        rgb = r.render().copy()
        # O mujoco_py devolvia a imagem de cabeca para baixo; quem chama ja conta com isso.
        rgb = rgb[::-1]
        if not depth:
            return rgb
        r.enable_depth_rendering()
        r.update_scene(d, camera=camera)
        prof = r.render().copy()[::-1]
        r.disable_depth_rendering()
        return rgb, prof


# ---------------------------------------------------------------------------
# Funcoes soltas (mujoco_py.functions)
# ---------------------------------------------------------------------------
def _cru(x):
    if isinstance(x, PyMjModel):
        return x._m
    if isinstance(x, PyMjData):
        return x._d
    return x


class _Funcoes:
    def __getattr__(self, nome):
        f = getattr(mujoco, nome)

        def chamada(*args, **kwargs):
            return f(*[_cru(a) for a in args], **{k: _cru(v) for k, v in kwargs.items()})

        return chamada

    @staticmethod
    def mj_ray(model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid):
        grupo = None
        if geomgroup is not None:
            # O codigo antigo passa 5 grupos; o MuJoCo tem 6.
            grupo = np.ones(mujoco.mjNGROUP, np.uint8)
            g = np.asarray(geomgroup, np.uint8).ravel()[:mujoco.mjNGROUP]
            grupo[:len(g)] = g
        saida = np.full(1, -1, np.int32)
        # flg_static=1: no MuJoCo 2.1 so os geoms do corpo 0 contavam como "estaticos";
        # hoje qualquer corpo sem junta conta, o que faria o raio atravessar as paredes.
        dist = mujoco.mj_ray(_cru(model), _cru(data),
                             np.ascontiguousarray(pnt, np.float64).reshape(3),
                             np.ascontiguousarray(vec, np.float64).reshape(3),
                             grupo, 1, int(bodyexclude), saida)
        if isinstance(geomid, np.ndarray):
            geomid.ravel()[0] = saida[0]
        elif geomid:
            ctypes.c_int.from_address(int(geomid)).value = int(saida[0])
        return dist


functions = _Funcoes()


class ignore_mujoco_warnings:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def __getattr__(nome):
    if nome == "MjViewer":
        from .mjviewer import MjViewer
        return MjViewer
    raise AttributeError(f"module 'mujoco_py' (compat) has no attribute {nome!r}")
