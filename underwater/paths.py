"""
paths.py — 纯数学时间线（不依赖 bpy）

所有运动（海龟、鱼群、5 个镜头的相机）都在这里用 numpy 计算成逐帧数组。
Blender 构建脚本和俯视示意图 (plot_paths.py) 共用同一份数据，保证示意图与成片一致。

坐标约定：米，Z 向上。海底平均高度 z≈0，水面 z=15（水深约 15 米）。
"""
import math
import numpy as np

# ----------------------------------------------------------------------------
# 全局时间设置
# ----------------------------------------------------------------------------
FPS = 30
FRAMES = 1200                      # 40 秒
SURFACE_Z = 15.0
REEF_CENTER = np.array([0.0, 4.0])
XFADE_FRAMES = 15                  # 镜头之间 0.5 秒叠化

# 镜头表：起止时间（秒）、焦距（mm）、光圈
SHOTS = [
    dict(id=1, t0=0.0,  t1=8.0,  lens=35.0, name="仰拍水面→下沉"),
    dict(id=2, t0=8.0,  t1=18.0, lens=50.0, name="侧面跟拍海龟"),
    dict(id=3, t0=18.0, t1=28.0, lens=24.0, name="环绕珊瑚礁 120°"),
    dict(id=4, t0=28.0, t1=36.0, lens=85.0, name="海龟头部特写推近"),
    dict(id=5, t0=36.0, t1=40.0, lens=28.0, name="上升拉远·渐黑"),
]

# 海龟模型上的关键点（海龟局部坐标：+X 朝前，+Y 朝左，+Z 朝上）
TURTLE_HEAD_LOCAL = np.array([0.66, 0.0, 0.035])
TURTLE_EYE_R_LOCAL = np.array([0.685, -0.058, 0.058])   # 右眼（相机特写这一侧）


def frame_time(f):
    """帧号(1 起) → 秒"""
    return (np.asarray(f, dtype=float) - 1.0) / FPS


def shot_frame_range(k):
    """第 k 个镜头（0 起）实际渲染的帧范围，含叠化用的前后余量。
    相邻镜头重叠 XFADE_FRAMES 帧：前 7 帧 + 后 8 帧。"""
    s = SHOTS[k]
    a = int(round(s["t0"] * FPS)) + 1
    b = int(round(s["t1"] * FPS))
    if k > 0:
        a -= XFADE_FRAMES // 2
    if k < len(SHOTS) - 1:
        b += XFADE_FRAMES - XFADE_FRAMES // 2
    return a, b


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def smootherstep(x):
    """缓入缓出（Perlin smootherstep，速度和加速度在两端都为 0）"""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def ease_window(t, t0, t1):
    return smootherstep((np.asarray(t, float) - t0) / (t1 - t0))


def normalize(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def rot2d(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class Spline:
    """非均匀 Catmull-Rom（Hermite，切线取相邻差分）。C1 连续，向量化求值。"""

    def __init__(self, times, points):
        self.t = np.asarray(times, float)
        self.p = np.asarray(points, float)
        n = len(self.t)
        m = np.zeros_like(self.p)
        for i in range(n):
            if i == 0:
                m[i] = (self.p[1] - self.p[0]) / (self.t[1] - self.t[0])
            elif i == n - 1:
                m[i] = (self.p[-1] - self.p[-2]) / (self.t[-1] - self.t[-2])
            else:
                m[i] = (self.p[i + 1] - self.p[i - 1]) / (self.t[i + 1] - self.t[i - 1])
        self.m = m

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        out = np.empty((len(t), self.p.shape[1]))
        idx = np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2)
        t0, t1 = self.t[idx], self.t[idx + 1]
        h = (t1 - t0)[:, None]
        u = ((t - t0) / (t1 - t0))[:, None]
        p0, p1, m0, m1 = self.p[idx], self.p[idx + 1], self.m[idx], self.m[idx + 1]
        u2, u3 = u * u, u * u * u
        out = ((2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * h * m0
               + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * h * m1)
        # 两端线性外推
        lo, hi = t < self.t[0], t > self.t[-1]
        out[lo] = self.p[0] + (t[lo] - self.t[0])[:, None] * self.m[0]
        out[hi] = self.p[-1] + (t[hi] - self.t[-1])[:, None] * self.m[-1]
        return out


def seabed_height(x, y):
    """起伏白沙海底的高度场（Blender 里的海底网格用同一个函数生成）"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return (0.30 * np.sin(0.23 * x + 0.5) * np.cos(0.19 * y - 0.3)
            + 0.22 * np.sin(0.11 * x - 0.17 * y + 1.1)
            + 0.10 * np.sin(0.61 * x + 0.43 * y)
            + 0.05 * np.cos(0.93 * x - 1.07 * y + 0.4))


def wobble(t, seed, amp):
    """水下手持感：3 个不可公约频率的正弦叠加（低频、平滑）"""
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    for _ in range(3):
        f = rng.uniform(0.07, 0.42)
        ph = rng.uniform(0, 2 * np.pi)
        out += np.sin(2 * np.pi * f * t + ph) * rng.uniform(0.6, 1.0)
    return out / 2.2 * amp


# ----------------------------------------------------------------------------
# 海龟
# ----------------------------------------------------------------------------
TURTLE_KEYS = [  # t,    x,     y,    z
    (0.0,  -11.0, -1.4, 2.5),
    (4.0,   -8.8, -0.9, 2.1),
    (8.0,   -6.6, -0.4, 1.9),
    (13.0,  -4.6,  0.0, 1.8),
    (18.0,  -2.6,  0.25, 1.85),
    (23.0,  -0.3,  0.9, 2.1),
    (28.0,   2.1,  1.9, 2.4),
    (32.0,   3.2,  2.5, 2.5),
    (36.0,   4.3,  2.95, 2.75),
    (38.0,   5.8,  3.3, 4.0),
    (40.0,   8.2,  3.8, 6.6),
]


def _rot_matrices(yaw, pitch, roll):
    """R = Rz(yaw) · Ry(-pitch) · Rx(roll)（pitch>0 抬头）→ (N,3,3)"""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(-pitch), np.sin(-pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    n = len(yaw)
    Rz = np.zeros((n, 3, 3)); Rz[:, 0, 0] = cy; Rz[:, 0, 1] = -sy; Rz[:, 1, 0] = sy; Rz[:, 1, 1] = cy; Rz[:, 2, 2] = 1
    Ry = np.zeros((n, 3, 3)); Ry[:, 0, 0] = cp; Ry[:, 0, 2] = sp; Ry[:, 2, 0] = -sp; Ry[:, 2, 2] = cp; Ry[:, 1, 1] = 1
    Rx = np.zeros((n, 3, 3)); Rx[:, 1, 1] = cr; Rx[:, 1, 2] = -sr; Rx[:, 2, 1] = sr; Rx[:, 2, 2] = cr; Rx[:, 0, 0] = 1
    return Rz @ Ry @ Rx


def turtle_track(T):
    sp = Spline([k[0] for k in TURTLE_KEYS], [k[1:] for k in TURTLE_KEYS])
    dt = 1.0 / FPS
    pos = sp(T)
    vel = (sp(T + dt) - sp(T - dt)) / (2 * dt)
    speed = np.linalg.norm(vel, axis=1)
    yaw = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    horiz = np.linalg.norm(vel[:, :2], axis=1)
    pitch = np.clip(np.arctan2(vel[:, 2], horiz) * 0.85, -0.5, 0.7)
    yaw_rate = np.gradient(yaw, dt)
    roll = np.clip(-yaw_rate * 0.8, -0.25, 0.25)
    # 划水频率：巡游 1/3.2 Hz，最后冲向光源时加快到 1/1.5 Hz
    freq = 1.0 / 3.2 + (1.0 / 1.5 - 1.0 / 3.2) * ease_window(T, 35.5, 38.0)
    stroke_phase = np.cumsum(freq) * dt * 2 * np.pi
    R = _rot_matrices(yaw, pitch, roll)
    head = pos + np.einsum("nij,j->ni", R, TURTLE_HEAD_LOCAL)
    eye_r = pos + np.einsum("nij,j->ni", R, TURTLE_EYE_R_LOCAL)
    return dict(pos=pos, vel=vel, speed=speed, yaw=yaw, pitch=pitch, roll=roll,
                R=R, head=head, eye_r=eye_r, stroke_phase=stroke_phase)


def _turtle_heading_h(tt, t):
    """平滑后的水平前进方向（单位 2D 向量）"""
    i0 = np.clip(np.searchsorted(tt["T"], t - 0.6), 0, len(tt["T"]) - 1)
    i1 = np.clip(np.searchsorted(tt["T"], t + 0.6), 0, len(tt["T"]) - 1)
    d = tt["pos"][i1, :2] - tt["pos"][i0, :2]
    return d / max(np.linalg.norm(d), 1e-6)


# ----------------------------------------------------------------------------
# 相机：每个镜头返回 pos / target / lens / fstop / focus_point / 晃动角
# ----------------------------------------------------------------------------
def _dir_from_yaw_pitch(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.stack([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)], axis=-1)


def camera_shot(k, T, tt):
    """k: 镜头序号（0 起），T: 秒数组，tt: turtle_track 结果（逐帧，与 FRAMES 对齐）"""
    s = SHOTS[k]
    n = len(T)
    idx = np.clip(np.round(T * FPS).astype(int), 0, FRAMES - 1)  # 对应 turtle 数组下标
    lens = np.full(n, s["lens"])
    fstop = np.full(n, 11.0)
    pos = np.zeros((n, 3))
    tgt = np.zeros((n, 3))
    focus = np.zeros((n, 3))

    if k == 0:
        # 镜头1：海底仰拍水面 → 缓慢向下俯仰 + 下沉
        e = ease_window(T, s["t0"], s["t1"])
        p0 = np.array([-12.6, -3.9, 4.4])
        p1 = np.array([-11.7, -3.3, 2.2])
        pos = p0 + (p1 - p0) * e[:, None]
        yaw = 18.0 + (30.0 - 18.0) * e
        pitch = 64.0 + (-5.0 - 64.0) * e
        tgt = pos + _dir_from_yaw_pitch(yaw, pitch) * 6.0
        focus = tgt
        fstop[:] = 8.0

    elif k == 1:
        # 镜头2：侧面横移跟拍，相机与海龟保持 3 米、同速
        e = ease_window(T, s["t0"], s["t1"])
        for i in range(n):
            tp = tt["pos"][idx[i]]
            h = _turtle_heading_h(tt, T[i])
            ang = -100.0 + 20.0 * e[i]          # 从略偏后方缓慢过渡到正侧面略偏前
            side = rot2d(h, ang)
            pos[i] = np.array([tp[0] + side[0] * 2.96, tp[1] + side[1] * 2.96, tp[2] - 0.5])
            tgt[i] = tp + np.array([h[0], h[1], 0.0]) * 0.18
            focus[i] = tt["eye_r"][idx[i]] * 0.5 + tp * 0.5   # 焦点锁在海龟靠近相机一侧
        fstop[:] = 2.8

    elif k == 2:
        # 镜头3：24mm 环绕珊瑚礁 120°
        e = ease_window(T, s["t0"], s["t1"])
        ang = np.radians(150.0 + 120.0 * e)
        r = 6.0 - 0.5 * np.sin(np.pi * e)
        pos[:, 0] = REEF_CENTER[0] + r * np.cos(ang)
        pos[:, 1] = REEF_CENTER[1] + r * np.sin(ang)
        pos[:, 2] = 1.35 + 0.9 * e
        tgt[:, 0], tgt[:, 1], tgt[:, 2] = REEF_CENTER[0], REEF_CENTER[1], 0.9
        focus = tgt.copy()
        fstop[:] = 8.0

    elif k == 3:
        # 镜头4：85mm 特写，缓慢推近海龟眼睛与头部
        e = ease_window(T, s["t0"], s["t1"])
        for i in range(n):
            hd = tt["head"][idx[i]]
            eye = tt["eye_r"][idx[i]]
            h = _turtle_heading_h(tt, T[i])
            side = rot2d(h, -58.0 + 8.0 * e[i])       # 前右侧
            d = 2.7 + (1.15 - 2.7) * e[i]
            pos[i] = np.array([hd[0] + side[0] * d, hd[1] + side[1] * d,
                               hd[2] + 0.06 + 0.20 * (1 - e[i])])
            tgt[i] = eye * 0.6 + hd * 0.4
            focus[i] = eye
        fstop[:] = 4.0

    elif k == 4:
        # 镜头5：快速上升并拉远，海龟游向远处光源
        e = ease_window(T, s["t0"], s["t0"] + 3.4)
        i36 = int(round(36.0 * FPS))
        p36 = tt["pos"][i36]
        h = _turtle_heading_h(tt, 36.0)
        side = rot2d(h, -90.0)
        start = np.array([p36[0] - h[0] * 3.2 + side[0] * 1.6, p36[1] - h[1] * 3.2 + side[1] * 1.6, p36[2] + 0.1])
        end = np.array([p36[0] - h[0] * 6.5 + side[0] * 2.8, p36[1] - h[1] * 6.5 + side[1] * 2.8, p36[2] + 3.6])
        pos = start + (end - start) * e[:, None]
        tp = tt["pos"][idx]
        light = tp + np.array([h[0] * 4.0, h[1] * 4.0, 3.5])
        w = (0.15 + 0.35 * e)[:, None]
        tgt = tp * (1 - w) + light * w
        focus = tp
        fstop[:] = 5.6
        lens = 28.0 + (24.0 - 28.0) * e        # 拉远时略微变广

    # 水下手持晃动：位置 + 角度（长焦减小角度幅度）
    amp_rot = {24: 0.55, 28: 0.5, 35: 0.45, 50: 0.3, 85: 0.16}.get(int(s["lens"]), 0.4)
    amp_pos = 0.03 if s["lens"] < 60 else 0.012
    base = 100 * (k + 1)
    pos = pos + np.stack([wobble(T, base + 1, amp_pos), wobble(T, base + 2, amp_pos),
                          wobble(T, base + 3, amp_pos * 0.8)], axis=1)
    rot = np.stack([wobble(T, base + 4, amp_rot), wobble(T, base + 5, amp_rot),
                    wobble(T, base + 6, amp_rot * 1.4)], axis=1)  # pitch / yaw / roll（度）

    fwd = normalize(tgt - pos)
    focus_dist = np.maximum(np.einsum("ni,ni->n", focus - pos, fwd), 0.3)
    return dict(T=T, pos=pos, target=tgt, lens=lens, fstop=fstop,
                focus_dist=focus_dist, wobble_deg=rot, fwd=fwd)


# ----------------------------------------------------------------------------
# 鱼群
# ----------------------------------------------------------------------------
FISH_COUNT = 64
FISH_CROSS_T = 22.4            # 鱼群从镜头3 前方穿过的时刻
FISH_SCATTER_T = FISH_CROSS_T + 0.4


def _cam_at(k, t, tt):
    c = camera_shot(k, np.array([t]), tt)
    return c["pos"][0], c["fwd"][0]


def fish_leader_spline(tt):
    cpos, cfwd = _cam_at(2, FISH_CROSS_T, tt)
    fh = normalize(np.array([cfwd[0], cfwd[1], 0.0]))
    right = np.array([fh[1], -fh[0], 0.0])
    up = np.array([0, 0, 1.0])
    C = cpos + fh * 0.95 - up * 0.12          # 镜头前 0.95 米处穿过
    R3 = np.array([REEF_CENTER[0], REEF_CENTER[1], 0.0])

    keys = []
    # 1) 开场到穿越前：顺时针绕礁巡游（半径 3.4，z≈2.7）
    B = C + right * 4.2 + fh * 1.6
    thB = math.atan2(B[1] - R3[1], B[0] - R3[0])
    w = 0.16
    t_end_circle = FISH_CROSS_T - 6.0
    for t in np.arange(-1.0, t_end_circle + 1e-6, 1.5):
        th = thB + 0.55 + w * (t_end_circle - t)
        keys.append((t, R3[0] + 3.4 * math.cos(th), R3[1] + 3.4 * math.sin(th), 2.7 + 0.25 * math.sin(0.4 * t)))
    # 2) 加速冲向镜头，从画面右侧横穿到左侧
    keys.append((FISH_CROSS_T - 3.4, *(B + up * 0.3)))
    keys.append((FISH_CROSS_T - 1.3, *(C + right * 1.5)))
    keys.append((FISH_CROSS_T, *C))
    keys.append((FISH_SCATTER_T - 0.02, *(C - right * 0.42)))
    # 3) 突然转向（背离镜头、斜向上）
    E = C - right * 0.75 + fh * 0.9 + up * 0.5
    keys.append((FISH_SCATTER_T + 0.55, *E))
    F = E + fh * 2.2 + right * 0.8 + up * 0.3
    keys.append((FISH_SCATTER_T + 2.6, *F))
    # 4) 重新集结，回到绕礁巡游
    thF = math.atan2(F[1] - R3[1], F[0] - R3[0])
    for j, t in enumerate(np.arange(FISH_SCATTER_T + 5.0, 42.0, 1.5)):
        th = thF - 0.6 - w * (t - (FISH_SCATTER_T + 5.0))
        keys.append((t, R3[0] + 3.5 * math.cos(th), R3[1] + 3.5 * math.sin(th), 2.8 + 0.25 * math.sin(0.4 * t)))
    keys.sort(key=lambda k: k[0])
    return Spline([k[0] for k in keys], [k[1:] for k in keys]), dict(C=C, right=right, fwd=fh, cam=cpos)


def fish_school(T, tt, seed=7):
    lead, info = fish_leader_spline(tt)
    rng = np.random.default_rng(seed)
    N = FISH_COUNT
    dt = 1.0 / FPS
    lag = rng.uniform(0.0, 0.9, N)
    off = rng.normal(size=(N, 3)) * np.array([0.32, 0.32, 0.18])
    sc_dir = normalize(off / np.array([0.32, 0.32, 0.18]) + rng.normal(size=(N, 3)) * 0.6
                       + info["fwd"] * 0.25)
    sc_dir[:, 2] *= 0.6
    sc_amp = rng.uniform(1.0, 2.4, N)
    size = rng.uniform(0.85, 1.2, N)

    def S(t):
        x = np.maximum(t - FISH_SCATTER_T, 0.0)
        return (1 - np.exp(-x / 0.42)) * np.exp(-x / 3.2)

    def pos_at(t):  # t: (F,) → (N,F,3)
        P = np.zeros((N, len(t), 3))
        for i in range(N):
            P[i] = lead(t - lag[i]) + off[i]
            P[i] += sc_dir[i] * (sc_amp[i] * S(t))[:, None]
            for a in range(3):
                P[i, :, a] += wobble(t, 1000 + i * 3 + a, 0.07)
        return P

    P = pos_at(T)
    V = (pos_at(T + dt) - pos_at(T - dt)) / (2 * dt)
    speed = np.linalg.norm(V, axis=2)
    fwd = normalize(V)
    # 限制俯仰
    hz = np.linalg.norm(fwd[..., :2], axis=2, keepdims=True)
    fwd[..., 2] = np.clip(fwd[..., 2], -0.6, 0.6)
    fwd = normalize(fwd)
    freq = (3.0 + 2.5 * np.clip(speed - 0.7, 0, 4)) * rng.uniform(0.9, 1.1, N)[:, None]
    tail_phase = np.cumsum(freq, axis=1) * dt * 2 * np.pi + rng.uniform(0, 6.28, N)[:, None]
    return dict(pos=P, fwd=fwd, speed=speed, tail_phase=tail_phase, size=size,
                leader=lead(T), info=info)


# ----------------------------------------------------------------------------
# 海草丛位置（两丛放在镜头2 相机与海龟之间做前景虚化，一丛在礁石西南侧）
# ----------------------------------------------------------------------------
def seagrass_spots(tl):
    c2 = tl["cams"][1]
    spots = []
    for t, d in ((10.4, 1.05), (15.2, 0.9)):
        i = int(np.argmin(np.abs(c2["T"] - t)))
        p = c2["pos"][i]
        tp = tl["turtle"]["pos"][int(round(t * FPS))]
        v = normalize((tp - p)[:2])
        spots.append(np.array([p[0] + v[0] * d, p[1] + v[1] * d]))
    spots.append(np.array([-2.5, 1.7]))
    return spots


# ----------------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------------
_CACHE = {}


def build_timeline():
    if "tl" in _CACHE:
        return _CACHE["tl"]
    T = frame_time(np.arange(1, FRAMES + 1))
    tt = turtle_track(T)
    tt["T"] = T
    cams = []
    for k in range(len(SHOTS)):
        a, b = shot_frame_range(k)
        Tk = frame_time(np.arange(a, b + 1))
        c = camera_shot(k, Tk, tt)
        c["frames"] = np.arange(a, b + 1)
        cams.append(c)
    fish = fish_school(T, tt)
    tl = dict(T=T, turtle=tt, cams=cams, fish=fish)
    tl["grass"] = seagrass_spots(tl)
    _CACHE["tl"] = tl
    return tl


if __name__ == "__main__":
    tl = build_timeline()
    tt = tl["turtle"]
    print("turtle speed min/max", tt["speed"].min().round(2), tt["speed"].max().round(2))
    for k, c in enumerate(tl["cams"]):
        g = seabed_height(c["pos"][:, 0], c["pos"][:, 1])
        print(f"shot{k+1} frames {c['frames'][0]}-{c['frames'][-1]}  min clearance {np.min(c['pos'][:,2]-g):.2f}"
              f"  focus {c['focus_dist'].min():.2f}-{c['focus_dist'].max():.2f}")
    f = tl["fish"]
    print("fish speed p50/p99/max", np.percentile(f["speed"], 50).round(2), np.percentile(f["speed"], 99).round(2), f["speed"].max().round(2))
    c3 = tl["cams"][2]
    a = c3["frames"][0]
    d = np.linalg.norm(f["pos"][:, a - 1:a - 1 + len(c3["frames"])] - c3["pos"][None], axis=2)
    print("fish min distance to cam3", d.min().round(2))
    c2 = tl["cams"][1]
    i0 = c2["frames"][0] - 1
    dd = np.linalg.norm(c2["pos"] - tt["pos"][i0:i0 + len(c2["frames"])], axis=1)
    print("shot2 cam-turtle dist", dd.min().round(2), dd.max().round(2))
    print("grass", [g.round(2) for g in tl["grass"]])
