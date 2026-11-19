"""
render.py — 渲染模块：Cycles 设置、质量预设、按镜头输出帧序列 / 关键帧截图
"""
import os
import time

import bpy

import paths as P

# 质量预设（分辨率、采样数）。CPU 渲染很慢，按需改这里。
PRESETS = {
    "stills":  dict(res=540,  samples=24,  denoise=True),
    "preview": dict(res=540,  samples=16,  denoise=True),
    "final":   dict(res=1080, samples=64,  denoise=True),
}

BASE_EXPOSURE = -0.2

# 每个镜头 2 张关键帧（帧号为全片绝对帧号）
KEYFRAMES = {
    1: [40, 215],
    2: [300, 480],
    3: [672, 688],
    4: [880, 1065],
    5: [1110, 1185],
}


def _render(**kw):
    """渲染；如果当前 Blender 没编译 OIDN（如部分 Linux 发行版自带的 Blender），自动关掉降噪重试"""
    try:
        bpy.ops.render.render(**kw)
    except RuntimeError as e:
        if "Denois" not in str(e) and "denois" not in str(e):
            raise
        print("[render] 此 Blender 不支持 OpenImageDenoise，关闭降噪后重试", flush=True)
        bpy.context.scene.cycles.use_denoising = False
        for vl in bpy.context.scene.view_layers:
            vl.cycles.use_denoising = False
        bpy.ops.render.render(**kw)


def setup(preset="preview", samples=None, res=None, threads=0):
    cfg = dict(PRESETS[preset])
    if samples:
        cfg["samples"] = samples
    if res:
        cfg["res"] = res
    sc = bpy.context.scene
    r = sc.render
    r.engine = "CYCLES"
    r.resolution_x = r.resolution_y = cfg["res"]
    r.resolution_percentage = 100
    r.fps = P.FPS
    sc.frame_start, sc.frame_end = 1, P.FRAMES
    r.use_persistent_data = True
    r.threads_mode = "FIXED" if threads else "AUTO"
    if threads:
        r.threads = threads
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.film_transparent = False

    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = cfg["samples"]
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.03
    cy.use_denoising = bool(cfg["denoise"])
    if cy.use_denoising:
        try:
            cy.denoiser = "OPENIMAGEDENOISE"
            cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        except Exception:
            print("[render] 此 Blender 没有 OpenImageDenoise，关闭降噪（建议提高采样数）")
            cy.use_denoising = False
    cy.max_bounces = 8
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 2
    cy.transmission_bounces = 6
    cy.volume_bounces = 0
    cy.transparent_max_bounces = 8
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 4.0
    cy.blur_glossy = 1.0
    try:
        cy.volume_step_rate = 4.0
        cy.volume_max_steps = 256
    except Exception:
        pass

    vs = sc.view_settings
    try:
        vs.view_transform = "AgX"
        vs.look = "AgX - Punchy"
    except Exception:
        try:
            vs.view_transform = "Filmic"
        except Exception:
            pass
    vs.exposure = BASE_EXPOSURE
    # 镜头1 仰拍太阳时降低曝光，随镜头下俯缓慢恢复（模拟摄影机自动曝光）
    import numpy as np
    from util import bake
    T = P.frame_time(np.arange(1, P.FRAMES + 1))
    ev = BASE_EXPOSURE - 0.8 * (1.0 - P.ease_window(T, 0.5, 6.5))
    bake(sc, "view_settings.exposure", np.arange(1, P.FRAMES + 1), ev)
    print(f"[render] preset={preset} res={cfg['res']} samples={cfg['samples']} denoise={cy.use_denoising}")
    return cfg


def _use_camera(k):
    sc = bpy.context.scene
    sc.camera = bpy.data.objects[f"Cam{k}"]


def render_stills(out_dir, shots=None, frames=None):
    os.makedirs(out_dir, exist_ok=True)
    sc = bpy.context.scene
    jobs = []
    for k, fl in KEYFRAMES.items():
        if shots and k not in shots:
            continue
        for f in (frames or fl):
            jobs.append((k, f))
    for k, f in jobs:
        _use_camera(k)
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out_dir, f"shot{k}_f{f:04d}.png")
        t = time.time()
        _render(write_still=True)
        print(f"[render] shot{k} frame {f} → {sc.render.filepath}  ({time.time() - t:.1f}s)", flush=True)


def render_shots(out_dir, shots=None, frame_range=None):
    sc = bpy.context.scene
    for k in range(1, len(P.SHOTS) + 1):
        if shots and k not in shots:
            continue
        a, b = P.shot_frame_range(k - 1)
        if frame_range:
            a, b = max(a, frame_range[0]), min(b, frame_range[1])
            if a > b:
                continue
        d = os.path.join(out_dir, f"shot{k}")
        os.makedirs(d, exist_ok=True)
        # 跳过已渲染的帧（便于中断后续渲）
        todo = [f for f in range(a, b + 1) if not os.path.exists(os.path.join(d, f"{f:04d}.png"))]
        if not todo:
            continue
        _use_camera(k)
        sc.render.use_overwrite = False
        sc.render.use_placeholder = True
        sc.frame_start, sc.frame_end = todo[0], b
        sc.render.filepath = os.path.join(d, "####")
        t = time.time()
        print(f"[render] shot{k} frames {todo[0]}-{b} → {d}", flush=True)
        _render(animation=True)
        print(f"[render] shot{k} done in {time.time() - t:.0f}s", flush=True)
