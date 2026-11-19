"""
plot_paths.py — 相机路径俯视示意图（直接读取 paths.py 的逐帧数据）
运行：python plot_paths.py [输出.png]
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Wedge, Circle, FancyArrowPatch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P

for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#e8590c", "#1971c2", "#2f9e44", "#9c36b5", "#c2255c"]


def main(out):
    tl = P.build_timeline()
    tt, cams, fish = tl["turtle"], tl["cams"], tl["fish"]
    T = tl["T"]

    fig = plt.figure(figsize=(17, 10.5), dpi=110)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.15, 1], hspace=0.42, wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    axz = fig.add_subplot(gs[0, 1])
    axp = fig.add_subplot(gs[1, 1])
    axt = fig.add_subplot(gs[2, 1])

    # 海底起伏
    xs = np.linspace(-15, 11, 200)
    ys = np.linspace(-6.5, 10, 140)
    X, Y = np.meshgrid(xs, ys)
    ax.contourf(X, Y, P.seabed_height(X, Y), levels=14, cmap="YlOrBr", alpha=0.28)

    # 珊瑚礁 / 海草
    rc = P.REEF_CENTER
    ax.add_patch(Circle(rc, 2.7, color="#f783ac", alpha=0.35, lw=0))
    ax.add_patch(Circle(rc, 1.5, color="#e64980", alpha=0.35, lw=0))
    ax.text(rc[0], rc[1] + 0.2, "珊瑚礁\n(礁石+珊瑚)", ha="center", va="center", fontsize=11, color="#a61e4d")
    for i, g in enumerate(tl["grass"]):
        ax.scatter(*g, s=260, marker="*", color="#2b8a3e", zorder=6, edgecolor="white")
        ax.text(g[0] + 0.25, g[1] - 0.45, f"海草{i+1}", fontsize=9, color="#2b8a3e")

    # 海龟路径
    ax.plot(tt["pos"][:, 0], tt["pos"][:, 1], color="#5c3d0e", lw=3.2, zorder=4, label="海龟路径")
    for t in range(0, 41, 4):
        i = min(int(t * P.FPS), P.FRAMES - 1)
        p = tt["pos"][i]
        ax.scatter(p[0], p[1], s=22, color="#5c3d0e", zorder=5)
        ax.text(p[0] + 0.12, p[1] + 0.22, f"{t}s", fontsize=8.5, color="#5c3d0e", zorder=7)

    # 鱼群路径（领头鱼）
    L = fish["leader"]
    ax.plot(L[:, 0], L[:, 1], color="#1098ad", lw=1.6, ls="--", zorder=3, label="鱼群(领头)路径")
    C = fish["info"]["C"]
    ax.scatter(C[0], C[1], s=90, marker="X", color="#0b7285", zorder=7)
    ax.annotate("鱼群贴近镜头3\n横穿 → 突然转向散开\n(t≈22.4–23.3s)", xy=(C[0], C[1]), xytext=(C[0] - 4.6, C[1] + 3.2),
                fontsize=9, color="#0b7285", arrowprops=dict(arrowstyle="->", color="#0b7285"))
    # 散开时刻各鱼的位置
    isc = int((P.FISH_SCATTER_T + 1.0) * P.FPS)
    ax.scatter(fish["pos"][:, isc, 0], fish["pos"][:, isc, 1], s=4, color="#0b7285", alpha=0.6, zorder=3)

    # 相机路径
    for k, c in enumerate(cams):
        s = P.SHOTS[k]
        col = COLORS[k]
        a0 = int(round(s["t0"] * P.FPS)) + 1
        m = (c["frames"] >= a0) & (c["frames"] <= int(round(s["t1"] * P.FPS)))
        pp = c["pos"][m]
        ax.plot(pp[:, 0], pp[:, 1], color=col, lw=2.6, zorder=8,
                label=f"镜头{k+1}  {s['t0']:.0f}–{s['t1']:.0f}s  {s['lens']:.0f}mm  {s['name']}")
        # 朝向箭头
        step = max(1, len(pp) // 7)
        fw = c["fwd"][m]
        for j in range(0, len(pp), step):
            h = fw[j, :2]
            hn = np.linalg.norm(h)
            L0 = 0.9 * hn + 0.15
            ax.add_patch(FancyArrowPatch(pp[j, :2], pp[j, :2] + h / max(hn, 1e-6) * L0,
                                         arrowstyle="-|>", mutation_scale=9, color=col, lw=1.0, zorder=9))
        # 起止 FOV 扇形
        for j, alpha in ((0, 0.10), (len(pp) - 1, 0.20)):
            h = fw[j, :2]
            yaw = math.degrees(math.atan2(h[1], h[0]))
            hf = math.degrees(math.atan(18.0 / c["lens"][m][j]))
            ax.add_patch(Wedge(pp[j, :2], 2.4, yaw - hf, yaw + hf, color=col, alpha=alpha, lw=0, zorder=2))
        ax.scatter(*pp[0, :2], s=60, color=col, marker="o", zorder=10, edgecolor="white")
        ax.scatter(*pp[-1, :2], s=70, color=col, marker="s", zorder=10, edgecolor="white")
        lab = pp[len(pp) // 2, :2]
        off = {0: (-1.6, -0.9), 1: (-0.4, -0.85), 2: (-1.4, 0.0), 3: (0.35, -0.55), 4: (-0.3, -1.0)}[k]
        ax.text(lab[0] + off[0], lab[1] + off[1], f"镜头{k+1}", fontsize=12, color=col, weight="bold", zorder=11)

    # 阳光方向
    ax.add_patch(FancyArrowPatch((10.2, 8.8), (8.4, 8.2), arrowstyle="-|>", mutation_scale=16, color="#f59f00", lw=2.5))
    ax.text(8.2, 9.15, "阳光来向（自 +X 斜射，\n与竖直约 25°）", fontsize=9, color="#e67700")

    ax.set_xlim(-15, 11)
    ax.set_ylim(-6.5, 10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("X (米)")
    ax.set_ylabel("Y (米)")
    ax.set_title("相机路径俯视图    ● 起点  ■ 终点  箭头=镜头朝向  扇形=水平视角（淡=起始，深=结束）", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    # 右侧：高度随时间
    axz.axhspan(-0.6, 0.5, color="#f1e3c4", alpha=0.7)
    axz.text(0.3, -0.35, "海底", fontsize=8, color="#8f6b2e")
    axz.plot(T, tt["pos"][:, 2], color="#5c3d0e", lw=2, label="海龟")
    axz.plot(T, fish["leader"][:, 2], color="#1098ad", lw=1.2, ls="--", label="鱼群")
    for k, c in enumerate(cams):
        axz.plot(P.frame_time(c["frames"]), c["pos"][:, 2], color=COLORS[k], lw=2)
    axz.set_ylabel("高度 z (米)\n水面在 z=15")
    axz.set_xlim(0, 40)
    axz.set_title("高度随时间（含叠化重叠段）", fontsize=11)
    axz.legend(fontsize=8, loc="upper left")
    axz.grid(alpha=0.25)

    # 俯仰角
    for k, c in enumerate(cams):
        fw = c["fwd"]
        pitch = np.degrees(np.arctan2(fw[:, 2], np.linalg.norm(fw[:, :2], axis=1)))
        axp.plot(P.frame_time(c["frames"]), pitch, color=COLORS[k], lw=2)
    axp.axhline(0, color="gray", lw=0.8)
    axp.set_ylabel("俯仰角 (°)\n+ 向上看")
    axp.set_xlim(0, 40)
    axp.set_title("镜头俯仰角", fontsize=11)
    axp.grid(alpha=0.25)

    # 相机速度（体现缓入缓出）
    for k, c in enumerate(cams):
        v = np.linalg.norm(np.gradient(c["pos"], axis=0), axis=1) * P.FPS
        axt.plot(P.frame_time(c["frames"]), v, color=COLORS[k], lw=2)
    axt.plot(T, tt["speed"], color="#5c3d0e", lw=1.2, ls=":", label="海龟速度")
    for s in P.SHOTS[1:]:
        axt.axvspan(s["t0"] - 0.25, s["t0"] + 0.25, color="gray", alpha=0.25)
    axt.set_ylabel("速度 (米/秒)")
    axt.set_xlabel("时间 (秒)   灰色竖条 = 0.5 秒叠化")
    axt.set_xlim(0, 40)
    axt.set_title("相机移动速度（缓入缓出 + 轻微漂浮晃动）", fontsize=11)
    axt.legend(fontsize=8, loc="upper left")
    axt.grid(alpha=0.25)

    fig.suptitle("热带珊瑚礁水下动画 · 40 秒 · 5 个镜头运镜示意", fontsize=16, y=0.985)
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "camera_paths_top.png")
