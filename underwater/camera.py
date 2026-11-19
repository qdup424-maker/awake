"""
camera.py — 相机模块
每个镜头一台相机（Cam1..Cam5），逐帧烘焙位置/朝向/焦距/对焦距离/光圈。
运动曲线来自 paths.py（已含缓入缓出与水下漂浮晃动）。
"""
import math

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

import paths as P
from util import add_object, bake, collection, eulers_from_matrices, look_matrix

COLL = "Cameras"


def build(tl):
    collection(COLL)
    cams = []
    for k, c in enumerate(tl["cams"]):
        cd = bpy.data.cameras.new(f"Cam{k+1}")
        cd.sensor_fit = "HORIZONTAL"
        cd.sensor_width = 36.0
        cd.clip_start = 0.03
        cd.clip_end = 250.0
        cd.dof.use_dof = True
        cd.dof.aperture_blades = 7
        cd.dof.aperture_rotation = math.radians(12)
        ob = add_object(f"Cam{k+1}", cd, COLL)
        ob.rotation_mode = "XYZ"

        mats = []
        for i in range(len(c["frames"])):
            M = look_matrix(c["fwd"][i])
            wp, wy, wr = np.radians(c["wobble_deg"][i])
            M = M @ Euler((wp, wy, wr), "XYZ").to_matrix()   # 局部晃动
            mats.append([list(r) for r in M])
        fr = c["frames"]
        bake(ob, "location", fr, c["pos"])
        bake(ob, "rotation_euler", fr, eulers_from_matrices(np.array(mats)))
        bake(cd, "lens", fr, c["lens"])
        bake(cd, "dof.focus_distance", fr, c["focus_dist"])
        bake(cd, "dof.aperture_fstop", fr, c["fstop"])
        cams.append(ob)
    bpy.context.scene.camera = cams[0]
    return cams
