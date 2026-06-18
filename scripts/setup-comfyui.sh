#!/usr/bin/env bash
set -euo pipefail

# Provision headless ComfyUI as Calque's Generator stage (WSL2 + CUDA).
# Idempotent: safe to re-run. Logs each phase with markers.
#
#   bash scripts/setup-comfyui.sh
#
# Models are NOT copied — ComfyUI reads the existing library at
# /mnt/f/Software/models via extra_model_paths.yaml. The two SD1.5 ControlNets
# we need in ComfyUI single-file format are fetched into models/controlnet.

COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
MODELS_ROOT="${MODELS_ROOT:-/mnt/f/Software/models}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "PHASE clone"
if [[ ! -d "$COMFY_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
  git -C "$COMFY_DIR" pull --ff-only || true
fi

echo "PHASE venv"
cd "$COMFY_DIR"
[[ -d venv ]] || python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

echo "PHASE torch"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  && echo "  torch+cuda already present" \
  || pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

echo "PHASE requirements"
pip install --quiet -r requirements.txt
# ComfyUI's requirements may pull a torchaudio built for a different CUDA than
# our torch wheel (causes 'libcudart.so.13 not found'). Re-pin to match torch.
pip install --quiet torchaudio==2.6.0 --index-url "$TORCH_INDEX"

echo "PHASE model-paths"
cat > "$COMFY_DIR/extra_model_paths.yaml" <<YAML
calque:
  base_path: ${MODELS_ROOT}
  checkpoints: checkpoints
  loras: loras
  ipadapter: ip-adapter
  controlnet: ${COMFY_DIR}/models/controlnet
  upscale_models: ${COMFY_DIR}/models/upscale_models
YAML

echo "PHASE controlnets"
CN_DIR="$COMFY_DIR/models/controlnet"
mkdir -p "$CN_DIR"
BASE="https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main"
for f in control_v11p_sd15_lineart_fp16.safetensors control_v11f1p_sd15_depth_fp16.safetensors; do
  if [[ ! -s "$CN_DIR/$f" ]]; then
    echo "  downloading $f"
    curl -fL --retry 3 -o "$CN_DIR/$f" "$BASE/$f"
  else
    echo "  have $f"
  fi
done

echo "PHASE verify"
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo "INSTALL_DONE"
