#!/usr/bin/env bash
# 一键运行：./run.sh stills|preview|final [main.py 的其他参数]
#   stills  : 每个镜头 2 张关键帧截图（540px，低采样）
#   preview : 540x540 全片预览，渲染完自动合成 underwater_preview.mp4
#   final   : 1080x1080 最终版，渲染完自动合成 underwater_final.mp4
# 默认用 PATH 里的 blender（blender -b -P main.py）；
# 也可以 BLENDER=/path/to/blender ./run.sh ... 指定，或 BPY_PYTHON=python ./run.sh ... 用 pip 版 bpy。
set -euo pipefail
cd "$(dirname "$0")"
PRESET="${1:-stills}"
shift || true

if [[ -n "${BPY_PYTHON:-}" ]]; then
  "$BPY_PYTHON" main.py --preset "$PRESET" "$@"
else
  "${BLENDER:-blender}" -b -P main.py -- --preset "$PRESET" "$@"
fi

if [[ "$PRESET" != "stills" ]]; then
  "${BPY_PYTHON:-python3}" encode.py "renders/$PRESET" "underwater_${PRESET}.mp4"
fi
