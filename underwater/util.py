"""
util.py — bpy 通用工具：材质节点、几何节点、网格构建、逐帧关键帧烘焙。
兼容 Blender 4.0 ~ 5.x（5.0 移除了 action.fcurves，这里做了兼容处理）。
"""
import bpy
import numpy as np
from mathutils import Matrix, Vector


# ----------------------------------------------------------------------------
# 场景 / 集合 / 对象
# ----------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def collection(name):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
    return c


def add_object(name, data, coll, location=(0, 0, 0), parent=None):
    ob = bpy.data.objects.new(name, data)
    collection(coll).objects.link(ob)
    ob.location = location
    if parent is not None:
        ob.parent = parent
    return ob


def add_empty(name, coll, location=(0, 0, 0), parent=None, size=0.1):
    ob = add_object(name, None, coll, location, parent)
    ob.empty_display_size = size
    return ob


def mesh_from_arrays(name, verts, faces, smooth=True):
    me = bpy.data.meshes.new(name)
    verts = np.asarray(verts, dtype=np.float32)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    me.update()
    set_smooth(me, smooth)
    return me


def set_smooth(me, smooth=True):
    try:
        me.shade_smooth() if smooth else me.shade_flat()
    except AttributeError:
        me.polygons.foreach_set("use_smooth", [smooth] * len(me.polygons))


def add_point_attr(me, name, values, kind="FLOAT"):
    a = me.attributes.new(name, kind, "POINT")
    if kind == "FLOAT":
        a.data.foreach_set("value", np.asarray(values, np.float32).ravel())
    else:
        a.data.foreach_set("vector", np.asarray(values, np.float32).ravel())
    return a


def set_visibility(ob, camera=True, diffuse=True, glossy=True, transmission=True, scatter=True, shadow=True):
    ob.visible_camera = camera
    ob.visible_diffuse = diffuse
    ob.visible_glossy = glossy
    ob.visible_transmission = transmission
    ob.visible_volume_scatter = scatter
    ob.visible_shadow = shadow


def mesh_from_object(ob, name):
    """把曲线等对象求值后转成网格数据（不依赖 bpy.ops）"""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev)
    me.name = name
    return me


# ----------------------------------------------------------------------------
# 材质节点
# ----------------------------------------------------------------------------
class NT:
    """材质节点树的小封装：n = NT(mat.node_tree); n.new('ShaderNodeTexNoise', Scale=3)"""

    def __init__(self, tree):
        self.t = tree
        self.x = 0

    def new(self, typ, **inputs):
        nd = self.t.nodes.new(typ)
        self.x += 180
        nd.location = (self.x, 0)
        for k, v in inputs.items():
            self.set(nd.inputs[k], v)
        return nd

    def set(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.t.links.new(v, sock)
        else:
            if isinstance(v, (tuple, list)) and len(v) == 3 and sock.type == "RGBA":
                v = (*v, 1.0)
            sock.default_value = v

    def link(self, a, b):
        self.t.links.new(a, b)

    # --- 常用组合 ---
    def math(self, op, a, b=None, clamp=False):
        nd = self.t.nodes.new("ShaderNodeMath")
        nd.operation = op
        nd.use_clamp = clamp
        self.set(nd.inputs[0], a)
        if b is not None:
            self.set(nd.inputs[1], b)
        return nd.outputs[0]

    def vmath(self, op, a, b=None):
        nd = self.t.nodes.new("ShaderNodeVectorMath")
        nd.operation = op
        self.set(nd.inputs[0], a)
        if b is not None:
            self.set(nd.inputs[1], b)
        return nd.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else nd.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        nd = self.t.nodes.new("ShaderNodeMix")
        nd.data_type = "RGBA"
        nd.blend_type = blend
        self.set(nd.inputs[0], fac)
        self.set(nd.inputs[6], a)
        self.set(nd.inputs[7], b)
        return nd.outputs[2]

    def ramp(self, fac, stops, interp="LINEAR"):
        nd = self.t.nodes.new("ShaderNodeValToRGB")
        cr = nd.color_ramp
        cr.interpolation = interp
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for el, (p, c) in zip(cr.elements, stops):
            el.position = p
            el.color = (*c, 1.0) if len(c) == 3 else c
        self.set(nd.inputs[0], fac)
        return nd.outputs[0]

    def sep(self, v):
        nd = self.t.nodes.new("ShaderNodeSeparateXYZ")
        self.set(nd.inputs[0], v)
        return nd.outputs

    def comb(self, x=0.0, y=0.0, z=0.0):
        nd = self.t.nodes.new("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.set(nd.inputs[i], v)
        return nd.outputs[0]

    def mapping(self, vec, scale=(1, 1, 1), loc=(0, 0, 0), rot=(0, 0, 0)):
        nd = self.t.nodes.new("ShaderNodeMapping")
        self.set(nd.inputs["Vector"], vec)
        nd.inputs["Scale"].default_value = scale
        nd.inputs["Location"].default_value = loc
        nd.inputs["Rotation"].default_value = rot
        return nd.outputs[0]

    def noise(self, vec, scale=5.0, detail=2.0, rough=0.5, dist=0.0, w=None):
        nd = self.t.nodes.new("ShaderNodeTexNoise")
        if w is not None:
            nd.noise_dimensions = "4D"
            self.set(nd.inputs["W"], w)
        self.set(nd.inputs["Vector"], vec)
        self.set(nd.inputs["Scale"], scale)
        nd.inputs["Detail"].default_value = detail
        nd.inputs["Roughness"].default_value = rough
        nd.inputs["Distortion"].default_value = dist
        return nd

    def voronoi(self, vec, scale=5.0, feature="F1", w=None, rand=1.0, smooth=None):
        nd = self.t.nodes.new("ShaderNodeTexVoronoi")
        nd.feature = feature
        if w is not None:
            nd.voronoi_dimensions = "4D"
            self.set(nd.inputs["W"], w)
        self.set(nd.inputs["Vector"], vec)
        self.set(nd.inputs["Scale"], scale)
        nd.inputs["Randomness"].default_value = rand
        if smooth is not None and "Smoothness" in nd.inputs:
            nd.inputs["Smoothness"].default_value = smooth
        return nd

    def bump(self, height, strength=0.3, distance=0.02, normal=None):
        nd = self.t.nodes.new("ShaderNodeBump")
        self.set(nd.inputs["Height"], height)
        nd.inputs["Strength"].default_value = strength
        nd.inputs["Distance"].default_value = distance
        if normal is not None:
            self.set(nd.inputs["Normal"], normal)
        return nd.outputs[0]

    def texcoord(self, out="Object"):
        nd = self.t.nodes.new("ShaderNodeTexCoord")
        return nd.outputs[out]

    def attr(self, name, out="Vector"):
        nd = self.t.nodes.new("ShaderNodeAttribute")
        nd.attribute_name = name
        return nd.outputs[out]

    def objinfo(self, out="Random"):
        nd = self.t.nodes.new("ShaderNodeObjectInfo")
        return nd.outputs[out]

    def time(self):
        """返回一个随帧变化的“秒”值（驱动器 frame/30，简单表达式无需开启 Python 自动执行）"""
        nd = self.t.nodes.new("ShaderNodeValue")
        fc = nd.outputs[0].driver_add("default_value")
        fc.driver.type = "SCRIPTED"
        fc.driver.expression = "frame/30.0"
        return nd.outputs[0]


def new_material(name):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (1600, 0)
    return m, NT(nt), out


def principled(n, **kw):
    """kw 用下划线代替空格：Base_Color=..., Coat_Weight=..."""
    b = n.new("ShaderNodeBsdfPrincipled")
    for k, v in kw.items():
        n.set(b.inputs[k.replace("_", " ")], v)
    return b


def assign(ob_or_me, mat):
    data = ob_or_me.data if isinstance(ob_or_me, bpy.types.Object) else ob_or_me
    data.materials.append(mat)


# ----------------------------------------------------------------------------
# 几何节点
# ----------------------------------------------------------------------------
class GN(NT):
    def __init__(self, name):
        self.ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
        try:
            self.ng.is_modifier = True
        except Exception:
            pass
        it = self.ng.interface
        it.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        it.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        super().__init__(self.ng)
        self.gi = self.ng.nodes.new("NodeGroupInput")
        self.go = self.ng.nodes.new("NodeGroupOutput")
        self.params = {}

    def param(self, name, default=0.0):
        s = self.ng.interface.new_socket(name, in_out="INPUT", socket_type="NodeSocketFloat")
        s.default_value = default
        self.params[name] = s.identifier
        return self.gi.outputs[name]

    def geo_in(self):
        return self.gi.outputs[0]

    def geo_out(self, sock):
        self.link(sock, self.go.inputs[0])

    def seconds(self):
        return self.ng.nodes.new("GeometryNodeInputSceneTime").outputs["Seconds"]

    def position(self):
        return self.ng.nodes.new("GeometryNodeInputPosition").outputs[0]

    def named_float(self, name):
        nd = self.ng.nodes.new("GeometryNodeInputNamedAttribute")
        nd.data_type = "FLOAT"
        nd.inputs["Name"].default_value = name
        return nd.outputs["Attribute"]

    def set_position(self, geo, position=None, offset=None):
        nd = self.new("GeometryNodeSetPosition")
        self.link(geo, nd.inputs["Geometry"])
        if position is not None:
            self.set(nd.inputs["Position"], position)
        if offset is not None:
            self.set(nd.inputs["Offset"], offset)
        return nd.outputs[0]


def add_gn_modifier(ob, gn, name, **values):
    mod = ob.modifiers.new(name, "NODES")
    mod.node_group = gn.ng
    for k, v in values.items():
        mod[gn.params[k]] = v
    return mod


# ----------------------------------------------------------------------------
# 逐帧关键帧烘焙（快速：foreach_set）
# ----------------------------------------------------------------------------
_LINEAR = None


def _find_fcurve(idb, data_path, index):
    ad = idb.animation_data
    act = ad.action
    fcs = getattr(act, "fcurves", None)   # Blender < 5.0
    if fcs is not None:
        fc = fcs.find(data_path, index=index)
        if fc is not None:
            return fc
    from bpy_extras import anim_utils
    cb = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
    return cb.fcurves.find(data_path, index=index)


def bake(idb, data_path, frames, values, index=None):
    """idb: 拥有该属性的 ID（对象/相机数据）；values: (F,) 或 (F,K)"""
    frames = np.asarray(frames, np.float32)
    values = np.asarray(values, np.float32)
    if values.ndim == 1:
        comps = [(index if index is not None else -1, values)]
    else:
        comps = [(i, values[:, i]) for i in range(values.shape[1])]
    for idx, vals in comps:
        idb.keyframe_insert(data_path, index=idx, frame=float(frames[0]))
        fc = _find_fcurve(idb, data_path, max(idx, 0))
        kp = fc.keyframe_points
        if len(kp) < len(frames):
            kp.add(len(frames) - len(kp))
        co = np.empty(2 * len(frames), np.float32)
        co[0::2] = frames
        co[1::2] = vals
        kp.foreach_set("co", co)
        for p in kp:
            p.interpolation = "LINEAR"
        fc.update()


def eulers_from_matrices(mats, order="XYZ"):
    """3x3 旋转矩阵序列 → 连续（不跳变）的欧拉角"""
    out = np.zeros((len(mats), 3))
    prev = None
    for i, m in enumerate(mats):
        M = Matrix([list(r) for r in m])
        e = M.to_euler(order, prev) if prev is not None else M.to_euler(order)
        out[i] = e[:]
        prev = e
    return out


def look_matrix(fwd, up=(0, 0, 1)):
    """相机朝向：-Z 指向 fwd，Y 尽量朝上"""
    q = Vector(fwd).to_track_quat("-Z", "Y")
    return q.to_matrix()
