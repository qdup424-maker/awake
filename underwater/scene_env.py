"""
scene_env.py — 场景模块
  海底白沙 / 海水体积雾 / 波动水面 / 礁石 / 多种珊瑚 / 漂浮颗粒 / 上升气泡
全部程序化生成。
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector, noise

import paths as P
from util import (GN, NT, add_gn_modifier, add_object, add_point_attr, assign,
                  collection, mesh_from_arrays, mesh_from_object, new_material,
                  principled, set_smooth, set_visibility)

ENV = "Environment"
REEF = "Reef"


# ============================================================================
# 网格工具
# ============================================================================
def grid_mesh(name, X, Y, Z):
    ny, nx = X.shape
    me = bpy.data.meshes.new(name)
    me.vertices.add(nx * ny)
    me.vertices.foreach_set("co", np.stack([X, Y, Z], -1).astype(np.float32).ravel())
    idx = np.arange(nx * ny).reshape(ny, nx)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).ravel()
    me.loops.add(len(q))
    me.loops.foreach_set("vertex_index", q.astype(np.int32))
    me.polygons.add(len(q) // 4)
    me.polygons.foreach_set("loop_start", np.arange(0, len(q), 4, dtype=np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.full(len(q) // 4, 4, np.int32))
    except Exception:
        pass
    me.update(calc_edges=True)
    set_smooth(me, True)
    return me


def bm_to_mesh(bm, name, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    set_smooth(me, smooth)
    return me


def icosphere_bm(subdiv=4):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    return bm


# ============================================================================
# 材质
# ============================================================================
def mat_sand():
    m, n, out = new_material("Sand")
    co = n.texcoord("Object")
    # 沙纹：扭曲的波纹带 + 细噪声
    wave = n.new("ShaderNodeTexWave", Vector=co, Scale=0.9, Distortion=6.0, Detail=3.0)
    wave.wave_type = "BANDS"
    wave.bands_direction = "DIAGONAL"
    fine = n.noise(co, scale=35.0, detail=4.0, rough=0.6)
    big = n.noise(co, scale=0.12, detail=3.0)
    specks = n.voronoi(co, scale=22.0)
    h = n.math("ADD", n.math("MULTIPLY", wave.outputs["Fac"], 0.7), n.math("MULTIPLY", fine.outputs["Fac"], 0.5))
    col = n.ramp(big.outputs["Fac"], [(0.3, (0.60, 0.57, 0.48)), (0.55, (0.80, 0.77, 0.67)), (0.8, (0.88, 0.86, 0.78))])
    col = n.mix(n.math("LESS_THAN", specks.outputs["Distance"], 0.06), col, (0.95, 0.94, 0.9))
    col = n.mix(n.math("MULTIPLY", wave.outputs["Fac"], 0.25), col, (0.62, 0.6, 0.52), "MULTIPLY")
    b = principled(n, Base_Color=col, Roughness=0.95, Normal=n.bump(h, 0.35, 0.03))
    n.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_rock():
    m, n, out = new_material("ReefRock")
    co = n.texcoord("Object")
    rnd = n.objinfo("Random")
    big = n.noise(co, scale=1.3, detail=6.0, rough=0.6)
    pat = n.noise(co, scale=3.2, detail=8.0, rough=0.65)
    pat2 = n.noise(co, scale=7.0, detail=4.0, rough=0.6)
    vor = n.voronoi(co, scale=14.0)
    base = n.ramp(big.outputs["Fac"], [(0.35, (0.30, 0.27, 0.22)), (0.6, (0.46, 0.42, 0.34))])
    # 粉紫色珊瑚藻、黄绿藻斑、暗色小孔
    pink = n.math("GREATER_THAN", pat.outputs["Fac"], n.math("ADD", 0.50, n.math("MULTIPLY", rnd, 0.06)))
    col = n.mix(n.math("MULTIPLY", pink, 0.9), base, n.ramp(rnd, [(0.0, (0.78, 0.36, 0.48)), (0.5, (0.62, 0.30, 0.62)), (1.0, (0.85, 0.48, 0.40))]))
    green = n.math("GREATER_THAN", pat2.outputs["Fac"], 0.6)
    col = n.mix(n.math("MULTIPLY", green, 0.8), col, (0.42, 0.48, 0.16))
    holes = n.math("LESS_THAN", vor.outputs["Distance"], 0.12)
    col = n.mix(n.math("MULTIPLY", holes, 0.7), col, (0.05, 0.04, 0.03))
    h = n.math("ADD", pat.outputs["Fac"], n.math("MULTIPLY", n.math("POWER", vor.outputs["Distance"], 0.5), 0.6))
    b = principled(n, Base_Color=col, Roughness=0.9, Normal=n.bump(h, 0.9, 0.06))
    n.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_coral(name, palette, pattern="noise", pscale=12.0, sss=0.15, bump=0.4, tip_color=None):
    """palette: 每个实例按 Object Random 从中挑一个颜色"""
    m, n, out = new_material(name)
    co = n.texcoord("Object")
    rnd = n.objinfo("Random")
    stops = [(i / max(len(palette) - 1, 1), c) for i, c in enumerate(palette)]
    base = n.ramp(rnd, stops, interp="CONSTANT" if len(palette) > 1 else "LINEAR")
    if pattern == "brain":
        wv = n.new("ShaderNodeTexWave", Vector=co, Scale=pscale, Distortion=14.0, Detail=4.0)
        wv.wave_type = "BANDS"
        h = wv.outputs["Fac"]
        groove = n.math("POWER", h, 3.0)
        col = n.mix(n.math("MULTIPLY", n.math("SUBTRACT", 1.0, h), 0.55), base, (0.18, 0.15, 0.08), "MULTIPLY")
    elif pattern == "polyp":
        v = n.voronoi(co, scale=pscale)
        h = n.math("POWER", v.outputs["Distance"], 0.5)
        col = n.mix(n.math("MULTIPLY", h, 0.18), base, (1.0, 1.0, 1.0), "SCREEN")
    else:
        nz = n.noise(co, scale=pscale, detail=5.0)
        h = nz.outputs["Fac"]
        col = n.mix(n.math("MULTIPLY", h, 0.3), base, (0.3, 0.3, 0.3), "MULTIPLY")
    if tip_color is not None:
        z = n.sep(n.texcoord("Generated"))[2]
        col = n.mix(n.math("MULTIPLY", n.math("POWER", z, 3.0), 0.9), col, tip_color)
    b = principled(n, Base_Color=col, Roughness=0.6, Subsurface_Weight=sss,
                   Subsurface_Radius=(0.05, 0.03, 0.02), Normal=n.bump(h, bump, 0.02))
    try:
        b.inputs["Subsurface Scale"].default_value = 0.05
    except Exception:
        pass
    n.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_water_volume():
    m, n, out = new_material("WaterVolume")
    sc = n.new("ShaderNodeVolumeScatter", Color=(0.30, 0.78, 0.95, 1), Density=0.035, Anisotropy=0.4)
    ab = n.new("ShaderNodeVolumeAbsorption", Color=(0.42, 0.80, 0.88, 1), Density=0.12)
    add = n.new("ShaderNodeAddShader")
    n.link(sc.outputs[0], add.inputs[0])
    n.link(ab.outputs[0], add.inputs[1])
    n.link(add.outputs[0], out.inputs["Volume"])
    return m


def mat_water_surface():
    m, n, out = new_material("WaterSurface")
    t = n.time()
    co = n.texcoord("Object")
    n1 = n.noise(n.mapping(co, scale=(0.32, 0.45, 0.32)), scale=1.0, detail=4.0, rough=0.55, w=n.math("MULTIPLY", t, 0.22))
    n2 = n.noise(co, scale=1.6, detail=2.0, w=n.math("MULTIPLY", t, 0.5))
    h = n.math("ADD", n1.outputs["Fac"], n.math("MULTIPLY", n2.outputs["Fac"], 0.25))
    g = n.new("ShaderNodeBsdfGlass", Color=(0.9, 1.0, 1.0, 1), Roughness=0.03, IOR=1.333, Normal=n.bump(h, 0.9, 0.6))
    n.link(g.outputs[0], out.inputs["Surface"])
    return m


def mat_snow():
    m, n, out = new_material("MarineSnow")
    b = principled(n, Base_Color=(0.85, 0.9, 0.85), Roughness=0.6, Subsurface_Weight=0.6,
                   Emission_Color=(0.6, 0.8, 0.8), Emission_Strength=0.08)
    n.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_bubble():
    m, n, out = new_material("Bubble")
    # 气泡 = 水中的空气：IOR = 1/1.333
    g = n.new("ShaderNodeBsdfGlass", Color=(1, 1, 1, 1), Roughness=0.0, IOR=0.75)
    lw = n.new("ShaderNodeLayerWeight", Blend=0.25)
    em = n.new("ShaderNodeEmission", Color=(0.7, 0.95, 1.0, 1), Strength=0.6)
    mx = n.new("ShaderNodeMixShader")
    n.link(lw.outputs["Fresnel"], mx.inputs[0])
    n.link(g.outputs[0], mx.inputs[1])
    n.link(em.outputs[0], mx.inputs[2])
    n.link(mx.outputs[0], out.inputs["Surface"])
    return m


# ============================================================================
# 海底 / 海水 / 水面
# ============================================================================
def build_seabed():
    N = 380
    u = np.linspace(-1, 1, N)
    xs = 20.0 * u + 60.0 * u ** 3          # 中心密、外围稀
    X, Y = np.meshgrid(xs, xs)
    Z = P.seabed_height(X, Y)
    Z += 0.02 * np.sin(2.3 * X + 1.1 * Y) * np.cos(1.7 * Y - 0.6 * X)
    me = grid_mesh("Seabed", X, Y, Z)
    ob = add_object("Seabed", me, ENV)
    assign(ob, mat_sand())
    return ob


def build_water():
    # 体积雾盒：相机在里面，顶面贴着水面下方
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    me = bm_to_mesh(bm, "WaterVolume", smooth=False)
    vol = add_object("WaterVolume", me, ENV, location=(0, 0, (P.SURFACE_Z - 0.02 - 3.0) / 2))
    vol.scale = (150, 150, P.SURFACE_Z - 0.02 + 3.0)
    assign(vol, mat_water_volume())

    # 水面：玻璃材质 + 动态法线；从下往上看能看到斯涅尔窗和全反射
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=4, y_segments=4, size=80.0)
    # 法线朝上：Cycles 根据入射方向自动判断从水到空气（IOR 取倒数 → 全反射）
    me = bm_to_mesh(bm, "WaterSurface", smooth=True)
    surf = add_object("WaterSurface", me, ENV, location=(0, 0, P.SURFACE_Z))
    assign(surf, mat_water_surface())
    set_visibility(surf, camera=True, diffuse=False, glossy=True, transmission=True, scatter=False, shadow=False)
    return vol, surf


# ============================================================================
# 礁石
# ============================================================================
def rock_mesh(name, seed, subdiv=4, rough=1.0):
    bm = icosphere_bm(subdiv)
    o1 = Vector((seed * 1.7, seed * 0.3, seed * 2.1))
    o2 = Vector((seed * 0.9, -seed * 1.3, seed * 0.5))
    for v in bm.verts:
        p = v.co.copy()
        d = 1.0 + rough * (0.28 * noise.noise(p * 1.1 + o1) + 0.14 * noise.noise(p * 2.7 + o2)
                           + 0.07 * noise.noise(p * 6.0 + o1) + 0.035 * noise.noise(p * 13.0 + o2))
        v.co = p * d
        if v.co.z < -0.35:
            v.co.z = -0.35 + (v.co.z + 0.35) * 0.3
    return bm_to_mesh(bm, name)


def build_reef_mound(rng, mat):
    rocks = []
    meshes = [rock_mesh(f"RockMesh{i}", 3.1 * i + 1, 5) for i in range(5)]
    R = P.REEF_CENTER
    layout = [  # dx, dy, sx, sy, sz, zlift
        (0.0, 0.0, 1.7, 1.4, 1.05, 0.15), (1.2, 0.6, 1.1, 1.0, 0.8, 0.0), (-1.3, 0.4, 1.2, 1.0, 0.75, 0.0),
        (0.4, -1.1, 1.0, 0.9, 0.6, -0.05), (-0.6, 1.3, 1.1, 0.9, 0.7, 0.0), (1.9, -0.6, 0.7, 0.6, 0.45, -0.05),
        (-2.0, -0.7, 0.8, 0.7, 0.45, -0.05), (0.9, 1.9, 0.8, 0.7, 0.5, -0.05), (-1.7, 1.6, 0.6, 0.6, 0.4, -0.05),
        (2.4, 1.2, 0.55, 0.5, 0.35, -0.05), (-0.3, -2.0, 0.55, 0.55, 0.35, -0.05),
    ]
    for i, (dx, dy, sx, sy, sz, zl) in enumerate(layout):
        x, y = R[0] + dx, R[1] + dy
        z = float(P.seabed_height(x, y)) + zl
        ob = add_object(f"Rock_{i}", meshes[i % len(meshes)], REEF, location=(x, y, z))
        ob.scale = (sx, sy, sz)
        ob.rotation_euler = (0, 0, rng.uniform(0, 6.28))
        rocks.append(ob)
    # 外围散落的小礁石
    for i in range(26):
        r = rng.uniform(5.0, 24.0)
        a = rng.uniform(0, 2 * math.pi)
        x, y = R[0] + r * math.cos(a), R[1] + r * math.sin(a)
        s = rng.uniform(0.2, 0.7)
        ob = add_object(f"RockS_{i}", meshes[i % len(meshes)], REEF,
                        location=(x, y, float(P.seabed_height(x, y)) - 0.05))
        ob.scale = (s * rng.uniform(0.8, 1.4), s * rng.uniform(0.8, 1.3), s * rng.uniform(0.4, 0.8))
        ob.rotation_euler = (0, 0, rng.uniform(0, 6.28))
        rocks.append(ob)
    for me in meshes:
        me.materials.append(mat)
    return rocks


# ============================================================================
# 珊瑚
# ============================================================================
def brain_mesh(name, seed, knobby=False):
    bm = icosphere_bm(5 if knobby else 4)
    o = Vector((seed, seed * 1.3, -seed))
    for v in bm.verts:
        p = v.co.copy()
        if knobby:
            d = 1.0 + 0.10 * abs(noise.noise(p * 5.0 + o)) + 0.06 * noise.noise(p * 2.0 + o)
        else:
            d = 1.0 + 0.08 * noise.noise(p * 1.5 + o)
        v.co = p * d
        if v.co.z < -0.15:
            v.co.z = -0.15 + (v.co.z + 0.15) * 0.15
        v.co.z *= 0.8
    return bm_to_mesh(bm, name)


def branch_curve_mesh(name, seed, levels=3, length=0.32, radius=0.03, children=(2, 3),
                      spread=0.75, up_bias=0.55, planar=False, tip_taper=0.45):
    rng = np.random.default_rng(seed)
    cu = bpy.data.curves.new(name + "_cu", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 1.0
    cu.bevel_resolution = 1 if planar else 2
    cu.use_fill_caps = True
    cu.resolution_u = 2

    def jitter():
        j = rng.normal(size=3)
        if planar:
            j[1] *= 0.08
        return j

    def grow(p, d, L, r, lvl):
        pts = [p.copy()]
        cur, dirv = p.copy(), d.copy()
        nseg = 4
        for _ in range(nseg):
            dirv = dirv + jitter() * 0.22 + np.array([0, 0, up_bias * 0.25])
            dirv /= np.linalg.norm(dirv)
            cur = cur + dirv * L / nseg
            pts.append(cur.copy())
        r_end = r * (0.72 if lvl > 0 else tip_taper)
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for i, q in enumerate(pts):
            sp.points[i].co = (q[0], q[1], q[2], 1.0)
            sp.points[i].radius = r + (r_end - r) * i / nseg
        if lvl > 0:
            nc = rng.integers(children[0], children[1] + 1)
            for c in range(nc):
                k = rng.integers(2, nseg + 1)
                nd = dirv + jitter() * spread
                nd[2] = abs(nd[2]) * 0.7 + up_bias * 0.6
                nd /= np.linalg.norm(nd)
                grow(pts[k], nd, L * rng.uniform(0.62, 0.85), r * 0.72, lvl - 1)

    n_trunks = 1 if planar else int(rng.integers(3, 6))
    for _ in range(n_trunks):
        d = np.array([rng.normal() * 0.5, 0.0 if planar else rng.normal() * 0.5, 1.0])
        d /= np.linalg.norm(d)
        base = np.array([rng.normal() * 0.05, 0.0 if planar else rng.normal() * 0.05, -0.05])
        grow(base, d, length * rng.uniform(0.8, 1.2), radius, levels)
    tmp = add_object(name + "_tmp", cu, REEF)
    me = mesh_from_object(tmp, name)
    bpy.data.objects.remove(tmp)
    bpy.data.curves.remove(cu)
    set_smooth(me, True)
    return me


def table_mesh(name, seed):
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    rings, segs = 10, 44
    H = 0.35
    verts = [bm.verts.new((0, 0, H))]
    o = Vector((seed * 2.0, seed, 0))
    ring_vs = []
    for i in range(1, rings + 1):
        rr = i / rings
        row = []
        for j in range(segs):
            a = 2 * math.pi * j / segs
            edge = 1.0 + 0.12 * noise.noise(Vector((math.cos(a) * 2, math.sin(a) * 2, seed)) + o)
            r = rr * edge
            z = H + 0.06 * rr * rr + 0.015 * noise.noise(Vector((r * math.cos(a) * 4, r * math.sin(a) * 4, seed)))
            row.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
        ring_vs.append(row)
    for j in range(segs):
        bm.faces.new((verts[0], ring_vs[0][j], ring_vs[0][(j + 1) % segs]))
    for i in range(rings - 1):
        for j in range(segs):
            a, b = ring_vs[i][j], ring_vs[i][(j + 1) % segs]
            c, d = ring_vs[i + 1][(j + 1) % segs], ring_vs[i + 1][j]
            bm.faces.new((a, d, c, b))
    # 加厚
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    geom = bmesh.ops.extrude_face_region(bm, geom=list(bm.faces))
    for v in [e for e in geom["geom"] if isinstance(e, bmesh.types.BMVert)]:
        v.co.z -= 0.045
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # 柄
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=10, radius1=0.12, radius2=0.07,
                          depth=H, matrix=Matrix.Translation((0, 0, H / 2 - 0.05)))
    return bm_to_mesh(bm, name)


def tube_cluster_mesh(name, seed):
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    n = int(rng.integers(3, 7))
    for i in range(n):
        h = rng.uniform(0.25, 0.85)
        r = rng.uniform(0.045, 0.09)
        ang = rng.uniform(0, 2 * math.pi)
        off = rng.uniform(0.0, 0.14)
        lean = rng.uniform(0.0, 0.22)
        M = (Matrix.Translation((off * math.cos(ang), off * math.sin(ang), 0))
             @ Matrix.Rotation(lean, 4, Vector((-math.sin(ang), math.cos(ang), 0)))
             @ Matrix.Translation((0, 0, h / 2 - 0.05)))
        bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False, segments=14, radius1=r * 0.8,
                                    radius2=r * 1.15, depth=h, matrix=M)
    # 管壁厚度在对象上用 Solidify 修改器添加
    return bm_to_mesh(bm, name)


def anemone_mesh(name, seed):
    rng = np.random.default_rng(seed)
    cu = bpy.data.curves.new(name + "_cu", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 1.0
    cu.bevel_resolution = 1
    cu.use_fill_caps = True
    for i in range(110):
        a = rng.uniform(0, 2 * math.pi)
        r0 = math.sqrt(rng.uniform(0, 1)) * 0.16
        L = rng.uniform(0.12, 0.22)
        d = np.array([math.cos(a) * (0.3 + r0 * 3), math.sin(a) * (0.3 + r0 * 3), 1.0])
        d /= np.linalg.norm(d)
        sp = cu.splines.new("POLY")
        sp.points.add(4)
        p = np.array([r0 * math.cos(a), r0 * math.sin(a), 0.05])
        for k in range(5):
            q = p + d * L * k / 4 + np.array([0, 0, -0.03 * (k / 4) ** 2])
            sp.points[k].co = (*q, 1.0)
            sp.points[k].radius = 0.009 * (1 - 0.6 * k / 4)
    # 底座
    tmp = add_object(name + "_tmp", cu, REEF)
    me = mesh_from_object(tmp, name)
    bpy.data.objects.remove(tmp)
    bpy.data.curves.remove(cu)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=16, radius1=0.2, radius2=0.17,
                          depth=0.1, matrix=Matrix.Translation((0, 0, 0.0)))
    bm.to_mesh(me)
    bm.free()
    set_smooth(me, True)
    return me


def sway_group():
    """通用摆动几何节点：按高度比例摆动（海草 / 海葵 / 海扇共用）"""
    g = GN("Sway")
    H = g.param("Height", 1.0)
    A = g.param("Amplitude", 0.2)
    W = g.param("Speed", 1.0)
    pos = g.position()
    x, y, z = g.sep(pos)
    hf = g.math("DIVIDE", z, H, clamp=True)
    hf2 = g.math("POWER", hf, 1.6)
    t = g.math("MULTIPLY", g.seconds(), W)
    ph = g.math("ADD", g.math("MULTIPLY", x, 1.9), g.math("MULTIPLY", y, 1.3))
    s1 = g.math("SINE", g.math("ADD", g.math("MULTIPLY", t, 1.15), ph))
    s2 = g.math("SINE", g.math("ADD", g.math("MULTIPLY", t, 2.7), g.math("MULTIPLY", ph, 1.7)))
    s3 = g.math("SINE", g.math("ADD", g.math("MULTIPLY", t, 0.83), g.math("MULTIPLY", ph, 1.3)))
    dx = g.math("MULTIPLY", g.math("MULTIPLY", g.math("ADD", g.math("ADD", s1, 0.25), g.math("MULTIPLY", s2, 0.3)), hf2), A)
    dy = g.math("MULTIPLY", g.math("MULTIPLY", s3, hf2), g.math("MULTIPLY", A, 0.45))
    dz = g.math("MULTIPLY", g.math("MULTIPLY", hf2, A), -0.15)
    geo = g.set_position(g.geo_in(), offset=g.comb(dx, dy, dz))
    g.geo_out(geo)
    return g


def build_corals(rng, tl, sway):
    cams = np.concatenate([c["pos"] for c in tl["cams"]])
    turtle = tl["turtle"]["pos"]
    fish_lead = tl["fish"]["leader"]

    mats = dict(
        brain=mat_coral("BrainCoral", [(0.72, 0.62, 0.18), (0.45, 0.62, 0.25), (0.80, 0.45, 0.25)], "brain", 6.0, 0.12, 0.6),
        knob=mat_coral("KnobCoral", [(0.95, 0.40, 0.55), (0.92, 0.62, 0.22), (0.62, 0.30, 0.80)], "polyp", 40.0, 0.25, 0.3),
        branch=mat_coral("BranchCoral", [(0.55, 0.22, 0.62), (0.92, 0.45, 0.18), (0.88, 0.52, 0.62), (0.70, 0.72, 0.30), (0.25, 0.45, 0.75)],
                         "polyp", 70.0, 0.3, 0.25, tip_color=(0.95, 0.85, 0.75)),
        table=mat_coral("TableCoral", [(0.55, 0.42, 0.18), (0.38, 0.48, 0.25), (0.62, 0.36, 0.28)], "polyp", 55.0, 0.15, 0.25),
        tube=mat_coral("TubeSponge", [(0.62, 0.18, 0.58), (0.95, 0.72, 0.10), (0.95, 0.40, 0.10)], "noise", 20.0, 0.2, 0.5),
        fan=mat_coral("SeaFan", [(0.75, 0.12, 0.18), (0.55, 0.20, 0.60), (0.95, 0.60, 0.12)], "noise", 30.0, 0.4, 0.1),
        anem=mat_coral("Anemone", [(0.85, 0.50, 0.70), (0.55, 0.80, 0.45), (0.95, 0.65, 0.45)], "noise", 25.0, 0.5, 0.1,
                       tip_color=(1.0, 0.95, 0.95)),
    )
    V = dict(
        brain=[brain_mesh(f"Brain{i}", 11 + i) for i in range(3)],
        knob=[brain_mesh(f"Knob{i}", 21 + i, True) for i in range(2)],
        branch=[branch_curve_mesh(f"Staghorn{i}", 31 + i, levels=3) for i in range(4)],
        table=[table_mesh(f"Table{i}", 41 + i) for i in range(2)],
        tube=[tube_cluster_mesh(f"Tube{i}", 51 + i) for i in range(3)],
        fan=[branch_curve_mesh(f"Fan{i}", 61 + i, levels=5, length=0.34, radius=0.012, children=(2, 2),
                               spread=0.9, up_bias=0.35, planar=True, tip_taper=0.4) for i in range(3)],
        anem=[anemone_mesh(f"Anemone{i}", 71 + i) for i in range(2)],
    )
    for k, ms in V.items():
        for me in ms:
            me.materials.append(mats[k])

    # 尺寸范围、局部高度（用于避让检查）
    spec = dict(brain=((0.25, 0.65), 0.75), knob=((0.18, 0.4), 0.75), branch=((0.9, 1.7), 0.9),
                table=((0.4, 0.75), 0.4), tube=((0.8, 1.4), 0.85), fan=((0.8, 1.5), 1.3), anem=((0.8, 1.3), 0.3))

    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene

    def ground_hit(x, y):
        o = Vector((x, y, 8.0))
        for _ in range(6):
            ok, loc, nrm, _, ob, _ = scene.ray_cast(dg, o, Vector((0, 0, -1)))
            if not ok:
                return None
            if ob.name.startswith(("Rock", "Seabed")):
                return loc, nrm, ob.name.startswith("Rock")
            o = loc + Vector((0, 0, -0.01))
        return None

    def clear(x, y, top):
        d_cam = np.min(np.hypot(cams[:, 0] - x, cams[:, 1] - y))
        if d_cam < 1.0 and np.min(cams[np.hypot(cams[:, 0] - x, cams[:, 1] - y) < 1.0][:, 2]) < top + 0.5:
            return False
        near = np.hypot(turtle[:, 0] - x, turtle[:, 1] - y) < 1.2
        if near.any() and np.min(turtle[near, 2]) - 0.45 < top + 0.15:
            return False
        nearf = np.hypot(fish_lead[:, 0] - x, fish_lead[:, 1] - y) < 0.8
        if nearf.any() and np.min(fish_lead[nearf, 2]) - 0.5 < top:
            return False
        return True

    placed = []
    R = P.REEF_CENTER

    def place(kind, x, y, sway_amp=None):
        hit = ground_hit(x, y)
        if hit is None:
            return None
        loc, nrm, on_rock = hit
        (smin, smax), hgt = spec[kind]
        s = rng.uniform(smin, smax)
        if not clear(x, y, loc.z + s * hgt):
            return None
        for (px, py, pr) in placed:
            if math.hypot(px - x, py - y) < (pr + s * 0.35) * 0.6:
                return None
        me = V[kind][int(rng.integers(len(V[kind])))]
        ob = add_object(f"Coral_{kind}_{len(placed)}", me, REEF, location=(loc.x, loc.y, loc.z - 0.03))
        ob.scale = (s, s, s * rng.uniform(0.85, 1.15))
        tilt = Vector(nrm).lerp(Vector((0, 0, 1)), 0.5).normalized()
        ob.rotation_mode = "XYZ"
        ob.rotation_euler = (math.atan2(-tilt.y, tilt.z) * 0.6, math.atan2(tilt.x, tilt.z) * 0.6, rng.uniform(0, 6.28))
        if kind == "tube":
            sm = ob.modifiers.new("Solid", "SOLIDIFY")
            sm.thickness = 0.012
        if sway_amp is not None:
            add_gn_modifier(ob, sway, "Sway", Height=hgt * 1.0, Amplitude=sway_amp, Speed=rng.uniform(0.7, 1.1))
        placed.append((x, y, s * 0.4))
        return ob

    # 1) 礁丘上密集珊瑚
    kinds_reef = ["brain"] * 4 + ["knob"] * 6 + ["branch"] * 10 + ["tube"] * 3 + ["fan"] * 3 + ["anem"] * 4
    tries = 0
    count = 0
    while count < 150 and tries < 4000:
        tries += 1
        r = 2.9 * math.sqrt(rng.uniform(0, 1))
        a = rng.uniform(0, 2 * math.pi)
        kind = kinds_reef[int(rng.integers(len(kinds_reef)))]
        amp = 0.03 if kind == "fan" else (0.025 if kind == "anem" else None)
        if place(kind, R[0] + r * math.cos(a), R[1] + r * math.sin(a), amp):
            count += 1
    # 2) 周围沙地上的零散珊瑚丛
    kinds_out = ["brain"] * 4 + ["knob"] * 3 + ["branch"] * 4 + ["tube"] * 2 + ["fan"] * 2 + ["anem"] * 1
    tries = 0
    count2 = 0
    while count2 < 95 and tries < 3000:
        tries += 1
        r = 3.2 + 22.0 * rng.uniform(0, 1) ** 1.3
        a = rng.uniform(0, 2 * math.pi)
        kind = kinds_out[int(rng.integers(len(kinds_out)))]
        amp = 0.03 if kind == "fan" else None
        if place(kind, R[0] + r * math.cos(a), R[1] + r * math.sin(a), amp):
            count2 += 1
    print(f"[env] corals placed: reef={count} outer={count2}")
    return placed


# ============================================================================
# 漂浮颗粒（海雪）与气泡 —— 几何节点，按时间确定性运动，可任意帧单独渲染
# ============================================================================
def build_marine_snow(rng, count=22000):
    lo = np.array([-17.0, -8.0, 0.0])
    hi = np.array([12.0, 12.0, 9.0])
    pts = rng.uniform(lo, hi, size=(count, 3))
    me = mesh_from_arrays("SnowPts", pts, [], smooth=False)
    add_point_attr(me, "rnd", rng.uniform(0, 1, count))
    ob = add_object("MarineSnow", me, ENV)

    g = GN("MarineSnowGN")
    t = g.seconds()
    pos = g.position()
    nz = g.noise(pos, scale=0.25, detail=1.0, w=g.math("MULTIPLY", t, 0.04))
    drift = g.vmath("SUBTRACT", nz.outputs["Color"], (0.5, 0.5, 0.5))
    drift = g.vmath("MULTIPLY", drift, (2.2, 2.2, 1.2))
    rise = g.comb(g.math("MULTIPLY", t, 0.03), 0.0, g.math("MULTIPLY", t, 0.012))
    p2 = g.vmath("ADD", g.vmath("ADD", pos, drift), rise)
    wrap = g.t.nodes.new("ShaderNodeVectorMath")
    wrap.operation = "WRAP"
    g.link(p2, wrap.inputs[0])
    wrap.inputs[1].default_value = tuple(hi)
    wrap.inputs[2].default_value = tuple(lo)
    geo = g.set_position(g.geo_in(), position=wrap.outputs[0])
    ico = g.new("GeometryNodeMeshIcoSphere", Radius=0.0035, Subdivisions=1)
    sm = g.new("GeometryNodeSetMaterial")
    g.link(ico.outputs[0], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = mat_snow()
    inst = g.new("GeometryNodeInstanceOnPoints")
    g.link(geo, inst.inputs["Points"])
    g.link(sm.outputs[0], inst.inputs["Instance"])
    sc = g.math("ADD", 0.5, g.math("MULTIPLY", g.math("POWER", g.named_float("rnd"), 3.0), 2.2))
    g.link(sc, inst.inputs["Scale"])
    g.geo_out(inst.outputs[0])
    add_gn_modifier(ob, g, "Snow")
    return ob


def build_bubbles(rng, emitters):
    pts, spd, rnd = [], [], []
    for (x, y, z) in emitters:
        for i in range(55):
            pts.append((x + rng.normal() * 0.04, y + rng.normal() * 0.04, z + rng.uniform(0, P.SURFACE_Z - z)))
            spd.append(rng.uniform(0.28, 0.45))
            rnd.append(rng.uniform(0, 1))
    pts = np.array(pts)
    me = mesh_from_arrays("BubblePts", pts, [], smooth=False)
    add_point_attr(me, "spd", spd)
    add_point_attr(me, "rnd", rnd)
    ob = add_object("Bubbles", me, ENV)

    g = GN("BubblesGN")
    t = g.seconds()
    r = g.named_float("rnd")
    sp = g.named_float("spd")
    ph = g.math("MULTIPLY", r, 40.0)
    wob_x = g.math("MULTIPLY", g.math("SINE", g.math("ADD", g.math("MULTIPLY", t, 7.0), ph)), 0.02)
    wob_y = g.math("MULTIPLY", g.math("COSINE", g.math("ADD", g.math("MULTIPLY", t, 6.1), ph)), 0.02)
    off = g.comb(wob_x, wob_y, g.math("MULTIPLY", t, sp))
    p2 = g.vmath("ADD", g.position(), off)
    zmin = float(min(e[2] for e in emitters))
    wrap = g.t.nodes.new("ShaderNodeVectorMath")
    wrap.operation = "WRAP"
    g.link(p2, wrap.inputs[0])
    wrap.inputs[1].default_value = (1e4, 1e4, P.SURFACE_Z - 0.05)
    wrap.inputs[2].default_value = (-1e4, -1e4, zmin)
    geo = g.set_position(g.geo_in(), position=wrap.outputs[0])
    ico = g.new("GeometryNodeMeshIcoSphere", Radius=0.006, Subdivisions=2)
    sm = g.new("GeometryNodeSetMaterial")
    g.link(ico.outputs[0], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = mat_bubble()
    inst = g.new("GeometryNodeInstanceOnPoints")
    g.link(geo, inst.inputs["Points"])
    g.link(sm.outputs[0], inst.inputs["Instance"])
    # 越往上越大（水压降低）
    zf = g.math("DIVIDE", g.sep(g.position())[2], P.SURFACE_Z)
    g.link(g.math("MULTIPLY", g.math("ADD", 0.6, g.math("MULTIPLY", r, 1.0)), g.math("ADD", 0.8, zf)), inst.inputs["Scale"])
    g.geo_out(inst.outputs[0])
    add_gn_modifier(ob, g, "Bubbles")
    return ob


# ============================================================================
def build(tl, seed=3):
    rng = np.random.default_rng(seed)
    collection(ENV)
    collection(REEF)
    build_seabed()
    build_water()
    build_reef_mound(rng, mat_rock())
    sway = sway_group()
    build_corals(rng, tl, sway)
    build_marine_snow(rng)
    R = P.REEF_CENTER
    em = []
    for dx, dy in ((0.5, 0.3), (-0.9, -0.4), (1.3, 1.4)):
        x, y = R[0] + dx, R[1] + dy
        em.append((x, y, float(P.seabed_height(x, y)) + 1.1))
    build_bubbles(rng, em)
    return sway
