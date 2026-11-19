"""
lighting.py — 灯光模块
  太阳光（从水面斜射）+ 焦散投影板（gobo，只对阴影可见，产生晃动光斑和体积光柱）+ 环境天空
"""
import math

import bmesh
import bpy
from mathutils import Vector

import paths as P
from util import add_object, collection, new_material, set_visibility

COLL = "Lights"

# 太阳在水下的入射方向：与竖直夹角约 25°，从 +X 方向（略偏 -Y）照来
SUN_ZENITH_DEG = 25.0
SUN_AZIMUTH_DEG = -12.0
SUN_STRENGTH = 30.0


def sun_vector():
    z = math.radians(SUN_ZENITH_DEG)
    a = math.radians(SUN_AZIMUTH_DEG)
    return Vector((math.sin(z) * math.cos(a), math.sin(z) * math.sin(a), math.cos(z)))


def build_world():
    w = bpy.data.worlds.new("UnderwaterWorld")
    bpy.context.scene.world = w
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.01, 0.05, 0.07, 1)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.55, 0.78, 0.95, 1)
    mid = cr.elements.new(0.35)
    mid.color = (0.20, 0.42, 0.55, 1)
    nt.links.new(sep.outputs[2], ramp.inputs[0])
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.45
    nt.links.new(ramp.outputs[0], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    return w


def build_sun():
    ld = bpy.data.lights.new("Sun", "SUN")
    ld.energy = SUN_STRENGTH
    ld.angle = math.radians(0.8)
    ld.color = (1.0, 0.97, 0.90)
    try:
        ld.cycles.cast_shadow = True
    except Exception:
        pass
    ob = add_object("Sun", ld, COLL, location=(0, 0, P.SURFACE_Z + 5))
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (-sun_vector()).to_track_quat("-Z", "Y")
    return ob


def mat_caustic_gobo():
    """透明度带图案的投影板：两层 4D Voronoi 网线 + 大尺度明暗，随时间变化 → 海底焦散 + 体积光柱"""
    m, n, out = new_material("CausticGobo")
    t = n.time()
    co = n.texcoord("Object")
    w1 = n.math("MULTIPLY", t, 0.32)
    w2 = n.math("MULTIPLY", t, 0.21)
    warp = n.noise(co, scale=0.15, detail=2.0, w=n.math("MULTIPLY", t, 0.05))
    cw = n.vmath("ADD", co, n.vmath("MULTIPLY", n.vmath("SUBTRACT", warp.outputs["Color"], (0.5, 0.5, 0.5)), (2.5, 2.5, 0)))
    v1 = n.voronoi(cw, scale=0.62, feature="DISTANCE_TO_EDGE", w=w1)
    v2 = n.voronoi(cw, scale=1.35, feature="DISTANCE_TO_EDGE", w=w2)
    l1 = n.math("SUBTRACT", 1.0, n.math("DIVIDE", v1.outputs["Distance"], 0.10), clamp=True)
    l2 = n.math("SUBTRACT", 1.0, n.math("DIVIDE", v2.outputs["Distance"], 0.08), clamp=True)
    lines = n.math("MAXIMUM", n.math("POWER", l1, 1.5), n.math("MULTIPLY", n.math("POWER", l2, 1.5), 0.6))
    # 稀疏的大尺度亮区 → 体积光柱；网线 → 海底焦散
    big = n.noise(cw, scale=0.16, detail=2.0, w=n.math("MULTIPLY", t, 0.06))
    beams = n.math("SUBTRACT", n.math("MULTIPLY", big.outputs["Fac"], 3.2), 1.45, clamp=True)
    beams = n.math("POWER", beams, 1.6)
    trans = n.math("ADD", n.math("ADD", 0.07, n.math("MULTIPLY", lines, 0.42)),
                   n.math("MULTIPLY", beams, n.math("ADD", 0.45, n.math("MULTIPLY", lines, 0.55))))
    trans = n.math("MINIMUM", trans, 1.0)
    col = n.comb(trans, trans, trans)
    tb = n.new("ShaderNodeBsdfTransparent")
    n.link(col, tb.inputs["Color"])
    n.link(tb.outputs[0], out.inputs["Surface"])
    return m


def build_gobo():
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=90.0)
    me = bpy.data.meshes.new("CausticGobo")
    bm.to_mesh(me)
    bm.free()
    ob = add_object("CausticGobo", me, COLL, location=(0, 0, P.SURFACE_Z + 0.4))
    me.materials.append(mat_caustic_gobo())
    set_visibility(ob, camera=False, diffuse=False, glossy=False, transmission=False, scatter=False, shadow=True)
    return ob


def build():
    collection(COLL)
    build_world()
    build_sun()
    build_gobo()
