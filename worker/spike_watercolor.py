"""Watercolor via diffusion LoRA (style in generation, not filter)."""
import io, sys, time
import cv2, numpy as np
from PIL import Image
from calque.generator import ComfyClient

SRC = sys.argv[1]
MAXW = 1024

def to_png(img): b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()

def main():
    c = ComfyClient(); assert c.is_up(), "ComfyUI down"
    img = Image.open(SRC).convert("RGB")
    w, h = img.size
    if w > MAXW: img = img.resize((MAXW, round(h*MAXW/w)), Image.LANCZOS); w, h = img.size
    img = img.resize((w - w % 8, h - h % 8)); w, h = img.size

    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    e = cv2.dilate(cv2.Canny(g, 60, 160), np.ones((3,3), np.uint8))
    soft = Image.fromarray(255 - cv2.GaussianBlur(e, (0,0), 2.0)).convert("RGB")
    free = Image.new("RGB", (w, h), (0,0,0))

    rn = c.upload_image(to_png(img), "wc_render.png")
    sn = c.upload_image(to_png(soft), "wc_soft.png")
    ln = c.upload_image(to_png(free), "wc_lock.png")
    img.save("wc_in.png")

    wf = c.load_workflow("structure_style", {
        "CKPT": "counterfeitV30_v30.safetensors",
        "LORA": "SoraWaterColor.safetensors", "LORA_STR": 1.0,
        "RENDER": rn, "SMAP": sn, "LOCKMAP": ln,
        "POS": "SoraWaterColor, watercolor painting, loose washes, wet-on-wet, "
               "soft edges, paper texture, interior room",
        "NEG": "photo, 3d render, cgi, sharp, hard edges, lowres, watermark, text",
        "RES": max(w, h), "SEED": 7, "STEPS": 28, "CFG": 7.0, "DENOISE": 0.85,
        "CN_DEPTH": 0.45, "CN_LINEART": 0.35,
    })
    print(f"{w}x{h} submitting watercolor LoRA...")
    t = time.time(); res = c.run(wf); print(f"done {time.time()-t:.1f}s")
    Image.open(io.BytesIO(res.images[0])).save("wc_out.png")
    print("-> wc_out.png")

if __name__ == "__main__":
    main()
