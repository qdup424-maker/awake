"""
main.py — 入口
用法（两种都可以）：
  blender -b -P main.py -- --preset stills
  python main.py --preset preview          # 需要 pip 安装的 bpy 模块

参数：
  --preset stills|preview|final   质量预设（见 render.py）
  --shots 1,3                     只渲染指定镜头
  --frames 300,480                stills 模式下指定帧
  --range 234,300                 动画模式下限制帧范围
  --samples N / --res N           覆盖预设
  --out DIR                       输出目录（默认 renders/<preset>）
  --save-blend FILE               保存 .blend 便于在 Blender 里查看
  --build-only                    只构建场景不渲染
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bpy  # noqa: E402

import camera  # noqa: E402
import creatures  # noqa: E402
import lighting  # noqa: E402
import paths as P  # noqa: E402
import render  # noqa: E402
import scene_env  # noqa: E402
import util  # noqa: E402


def parse():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="stills", choices=list(render.PRESETS))
    ap.add_argument("--shots", default="")
    ap.add_argument("--frames", default="")
    ap.add_argument("--range", default="")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-blend", default="")
    ap.add_argument("--build-only", action="store_true")
    return ap.parse_args(argv)


def build_scene():
    t0 = time.time()
    util.reset_scene()
    tl = P.build_timeline()
    sway = scene_env.build(tl)
    print(f"[main] environment built ({time.time() - t0:.1f}s)", flush=True)
    creatures.build(tl, sway)
    print(f"[main] creatures built ({time.time() - t0:.1f}s)", flush=True)
    lighting.build()
    camera.build(tl)
    print(f"[main] scene ready ({time.time() - t0:.1f}s)", flush=True)
    return tl


def main():
    a = parse()
    build_scene()
    render.setup(a.preset, a.samples or None, a.res or None, a.threads)
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save_blend))
    if a.build_only:
        return
    out = a.out or os.path.join(HERE, "renders", a.preset)
    shots = [int(s) for s in a.shots.split(",") if s] or None
    if a.preset == "stills":
        frames = [int(f) for f in a.frames.split(",") if f] or None
        render.render_stills(out, shots, frames)
    else:
        rng = tuple(int(x) for x in a.range.split(",")) if a.range else None
        render.render_shots(out, shots, rng)


if __name__ == "__main__":
    main()
