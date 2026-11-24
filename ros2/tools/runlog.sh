#!/usr/bin/env bash
# ROS2 部署记录器：把你跑的每条命令 + 完整输出 + 退出码自动落盘。
#
# 为什么需要它
#   "预判版"教程升级成"实录版"，靠的不是事后回忆，而是当时的原始输出。
#   手动复制粘贴会丢三样东西：完整命令、时间戳、退出码——而这三样正是排错的关键。
#
# 用法（先 source 一次）
#   source /mnt/d/VC\ Code/YOLO_Road_Detection/ros2/tools/runlog.sh
#   log A1-compile colcon build --packages-select yolo_trt_ros --symlink-install
#   log A1-check  ros2 topic hz /yolo_trt_node/detections
#
# 产物位置（Windows 侧我可以直接读）
#   D:\VC Code\YOLO_Road_Detection\ros2\runlogs\2026-11-24\
#       A1-compile_143512.log    ← 含命令、环境、完整 stdout/stderr、退出码
#       INDEX.md                 ← 本次会话跑了哪些命令的清单

# 项目根目录：脚本所在 ros2/ 目录（Windows D 盘 -> WSL /mnt/d）
RUNLOG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUNLOG_DIR="$RUNLOG_ROOT/runlogs/$(date +%Y-%m-%d)"
mkdir -p "$RUNLOG_DIR"

# 索引文件：每次 log 追加一行，方便我一眼看清你跑了哪些步骤
RUNLOG_INDEX="$RUNLOG_DIR/INDEX.md"
if [ ! -f "$RUNLOG_INDEX" ]; then
  {
    echo "# ROS2 部署实测记录 · $(date +%Y-%m-%d)"
    echo
    echo "> 自动生成。每行对应一份 .log，小T 会直接读这个目录。"
    echo
    echo "| 时间 | 标签 | 命令 | 退出码 | 日志 |"
    echo "|---|---|---|---|---|"
  } > "$RUNLOG_INDEX"
fi

log() {
  local tag="$1"; shift
  if [ -z "$tag" ] || [ $# -eq 0 ]; then
    echo "用法: log <标签> <命令...>" >&2
    return 2
  fi

  local ts="$(date +%H%M%S)"
  local full="$(date +'%Y-%m-%d %H:%M:%S')"
  local file="$RUNLOG_DIR/${tag}_${ts}.log"

  {
    echo "============================================================"
    echo "# 标签: $tag"
    echo "# 时间: $full"
    echo "# 目录: $(pwd)"
    echo "# 命令: $*"
    echo "# ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<未设>}  RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<默认>}"
    echo "============================================================"
  } > "$file"

  echo "→ 记录到: $file"
  echo "──────── 以下输出同时显示在终端并写入日志 ────────"
  # 关键：用 tee 而不是 >>，让命令输出既能当场看到，也能落盘。
  # 之前纯重定向导致用户"盲跑"，只能事后靠读日志判断成功与否。
  # ${PIPESTATUS[0]} 取的是被包裹命令的退出码（不是 tee 的）。
  "$@" 2>&1 | tee -a "$file"
  local rc=${PIPESTATUS[0]}

  echo "" >> "$file"
  echo "============================================================" >> "$file"
  echo "# 退出码: $rc" >> "$file"
  echo "============================================================" >> "$file"

  printf '| %s | %s | `%s` | %s | [%s](%s) |\n' \
    "$(date +%H:%M:%S)" "$tag" "$*" "$rc" "$(basename "$file")" "$(basename "$file")" >> "$RUNLOG_INDEX"

  echo "← 退出码 $rc，已写入索引"
  return $rc
}

echo "runlog 已加载，日志目录：$RUNLOG_DIR"
echo "用法：log <标签> <命令...>"
