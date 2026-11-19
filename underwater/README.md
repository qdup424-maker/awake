# 热带珊瑚礁 · 水下动画（Blender / bpy 程序化生成）

40 秒，1080×1080，30fps，共 1200 帧。Cycles 渲染。所有模型、材质、动画都由代码生成，不导入任何外部素材。

## 模块

| 文件 | 内容 |
|---|---|
| `paths.py` | **纯数学时间线**（不依赖 bpy）：海龟路径、64 条鱼的鱼群运动、5 个镜头的相机轨迹（缓入缓出 + 水下漂浮晃动）、海底高度场。示意图和 Blender 用的是同一份数据 |
| `scene_env.py` | 场景：起伏白沙海底、海水体积雾、波动反光的水面、礁石、6 类珊瑚（脑珊瑚、瘤状珊瑚、鹿角珊瑚、管状海绵、海扇、海葵；桌面珊瑚的生成函数保留但默认不放置）、漂浮颗粒、上升气泡 |
| `creatures.py` | 生物：海龟（龟壳/头/眼/四肢，前肢扑动 + 攻角 + 前后扫，后肢舵式摆动，身体起伏）、鱼群（几何节点尾部摆动）、3 丛海草（几何节点随水流摆动） |
| `lighting.py` | 灯光：太阳（与竖直约 25°）、焦散投影板（只对阴影可见，产生海底晃动光斑和体积光柱）、环境天空 |
| `camera.py` | 相机：每个镜头一台相机，逐帧烘焙位置、朝向、焦距、对焦距离、光圈 |
| `render.py` | 渲染：Cycles 设置、质量预设、关键帧截图、按镜头渲染帧序列（可断点续渲） |
| `main.py` | 入口 |
| `encode.py` | ffmpeg 合成：镜头间 0.5 秒叠化、结尾渐黑、轻微暗角，输出 H.264 mp4 |
| `plot_paths.py` | 生成相机路径俯视示意图 |
| `util.py` | 材质节点 / 几何节点 / 关键帧烘焙工具（兼容 Blender 4.0 ~ 5.x） |

## 运行

```bash
# 关键帧截图（每个镜头 2 张）
blender -b -P main.py -- --preset stills

# 540x540 全片预览 → 合成 mp4
blender -b -P main.py -- --preset preview
python3 encode.py renders/preview underwater_preview.mp4

# 1080x1080 最终版
blender -b -P main.py -- --preset final
python3 encode.py renders/final underwater_final.mp4

# 或者一步到位
./run.sh preview
```

常用参数：`--shots 2,4`（只渲染某几个镜头）、`--range 234,300`（限制帧范围）、`--samples 32`、`--res 720`、`--save-blend scene.blend`（保存工程，在 Blender 里打开查看）、`--build-only`。

已渲染的帧会自动跳过，中断后重新运行同一条命令即可续渲。

也可以不用 Blender 程序，直接用 pip 安装的 `bpy` 模块运行：`python main.py --preset preview`（需要 Python 版本与 bpy 匹配，例如 bpy 5.1 对应 Python 3.13）。

## 镜头

| 镜头 | 时间 | 焦距 | 内容 |
|---|---|---|---|
| 1 | 0–8 s | 35mm | 海底仰拍水面，缓慢向下俯仰（+64° → −5°）并下沉 2.2 m，光柱从画面上方划过；开头曝光 −0.8 档，随下俯恢复 |
| 2 | 8–18 s | 50mm | 侧面横移跟拍海龟，距离 2.96–3.03 m、同速；f/2.8，焦点锁定海龟；两丛海草作为前景虚化掠过 |
| 3 | 18–28 s | 24mm | 环绕珊瑚礁 120°；22.4 s 鱼群从镜头前约 1 m 处横穿，22.8 s 突然转向散开 |
| 4 | 28–36 s | 85mm | f/4 特写，从 2.7 m 缓慢推近到 1.15 m，焦点在海龟右眼 |
| 5 | 36–40 s | 28→24mm | 快速上升并拉远，海龟加速游向光亮处，最后 1.5 秒渐暗到黑 |

镜头之间的 0.5 秒叠化：每个镜头渲染时前后多渲 7–8 帧，合成时用 ffmpeg `xfade` 重叠 15 帧，总长正好 1200 帧。

## 调参入口

- 水的颜色 / 能见度：`scene_env.py` → `mat_water_volume()`（散射 Density 越大越雾，吸收 Color 决定色调）
- 阳光强度 / 角度：`lighting.py` 顶部 `SUN_*`
- 焦散与光柱图案：`lighting.py` → `mat_caustic_gobo()`
- 曝光：`render.py` → `BASE_EXPOSURE`
- 运镜：`paths.py` → `camera_shot()`；海龟路线 `TURTLE_KEYS`；鱼群穿越时刻 `FISH_CROSS_T`
- 质量：`render.py` → `PRESETS`
