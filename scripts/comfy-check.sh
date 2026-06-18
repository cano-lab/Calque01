#!/usr/bin/env bash
# Wait for ComfyUI to be up, then report the models/nodes Calque needs.
set -u
HOST=http://127.0.0.1:8188

for i in $(seq 1 40); do
  if curl -sf "$HOST/system_stats" >/dev/null 2>&1; then echo "UP (after $((i*2))s)"; break; fi
  sleep 2
done

q() { # $1 = node class, $2 = input field
  curl -s "$HOST/object_info/$1" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(d['$1']['input']['required']['$2'][0])
except Exception as e:
    print('  (node $1 not found:', e, ')')
"
}

has() { # $1 = node class
  curl -s "$HOST/object_info/$1" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('YES' if '$1' in d and d['$1'] else 'NO')
"
}

echo "=== checkpoints ==="; q CheckpointLoaderSimple ckpt_name
echo "=== controlnets ==="; q ControlNetLoader control_net_name
echo "=== loras ==="; q LoraLoader lora_name
echo "=== upscale_models ==="; q UpscaleModelLoader model_name
echo "=== DifferentialDiffusion node: $(has DifferentialDiffusion)"
echo "=== InpaintModelConditioning node: $(has InpaintModelConditioning)"
echo "=== ImageCompositeMasked node: $(has ImageCompositeMasked)"
echo "=== SAMLoader node (impact pack): $(has SAMLoader)"
