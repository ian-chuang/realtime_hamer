# Optional: source before running demos. scripts/demo_video.py also preloads
# these via realtime_hamer.trt_runtime.preload_gpu_libs().
# Usage: source scripts/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="$ROOT/.venv/lib/python3.11/site-packages"
NVIDIA_LIBS="$(find "$SITE/nvidia" -type d -name lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="\
${NVIDIA_LIBS}\
$SITE/tensorrt_libs:\
$SITE/tensorrt_lean_libs:\
$SITE/tensorrt_dispatch_libs:\
$SITE/torch/lib:\
/usr/local/cuda/lib64:\
${LD_LIBRARY_PATH:-}"
