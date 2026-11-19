"""
encode.py — 用 ffmpeg 把 5 个镜头的帧序列合成 mp4（H.264）
  - 镜头之间 0.5 秒叠化（xfade，利用渲染时多出的 15 帧重叠）
  - 结尾 1.5 秒渐暗到黑
  - 轻微暗角
用法：python encode.py renders/preview underwater_preview.mp4 [--crf 18]
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("output")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--fade-out", type=float, default=1.5)
    a = ap.parse_args()

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats"]
    lens = []
    for k in range(len(P.SHOTS)):
        s, e = P.shot_frame_range(k)
        d = os.path.join(a.frames_dir, f"shot{k+1}")
        missing = [f for f in range(s, e + 1) if not os.path.exists(os.path.join(d, f"{f:04d}.png"))]
        if missing:
            sys.exit(f"镜头{k+1} 缺少 {len(missing)} 帧（例如 {missing[:5]}），请先渲染完整")
        cmd += ["-framerate", str(P.FPS), "-start_number", str(s), "-i", os.path.join(d, "%04d.png")]
        lens.append(e - s + 1)

    xd = P.XFADE_FRAMES / P.FPS
    parts, prev, acc = [], "[0:v]", lens[0]
    for k in range(1, len(lens)):
        off = (acc - P.XFADE_FRAMES) / P.FPS
        out = f"[x{k}]"
        parts.append(f"{prev}[{k}:v]xfade=transition=fade:duration={xd:.4f}:offset={off:.4f}{out}")
        prev = out
        acc = acc + lens[k] - P.XFADE_FRAMES
    total = acc / P.FPS
    parts.append(f"{prev}vignette=angle=PI/5.5,fade=t=out:st={total - a.fade_out:.4f}:d={a.fade_out},format=yuv420p[v]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-r", str(P.FPS),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-movflags", "+faststart", a.output]
    print(f"合成 {acc} 帧 = {total:.2f} 秒 → {a.output}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
