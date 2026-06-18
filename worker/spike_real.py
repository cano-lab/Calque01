"""
Spike on a real architectural render.

Auto-derive the precision map from structure edges so hard geometry (edges,
fixtures) stays anchored while flat surfaces (walls, floor, ceiling) are free
to be reinterpreted. No hard pixel-lock here (lock map = all free) — we want to
see the generator restyle the whole room first.
"""
import io, sys, time
import cv2
import numpy as np
from PIL import Image

from calque.generator import ComfyClient
from calque.postprocess.true_watercolor import TrueWatercolorEffect

SRC = sys.argv[1] if len(sys.argv) > 1 else "studio.png"
MAXW = 1024  # SD1.5 comfort on 8GB; we upscale later


def to_png(img): b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()


def fit(img):
    w, h = img.size
    if w <= MAXW: return img
    return img.resize((MAXW, round(h * MAXW / w)), Image.LANCZOS)


def edge_strength_map(img):
    """white(free) on flat areas, darker(anchor) near strong edges."""
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(g, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    soft = 255 - cv2.GaussianBlur(edges, (0, 0), 2.0)  # anchor near edges
    return Image.fromarray(soft).convert("RGB")


def main():
    c = ComfyClient(); assert c.is_up(), "ComfyUI down"
    render = fit(Image.open(SRC).convert("RGB"))
    w, h = render.size
    render = render.resize((w - w % 8, h - h % 8))  # VAE-safe dims
    w, h = render.size
    soft = edge_strength_map(render)
    free = Image.new("RGB", (w, h), (0, 0, 0))  # lock nothing this pass

    rn = c.upload_image(to_png(render), "real_render.png")
    sn = c.upload_image(to_png(soft), "real_soft.png")
    ln = c.upload_image(to_png(free), "real_lock.png")
    render.save("real_in.png"); soft.save("real_soft_map.png")

    wf = c.load_workflow("structure_control", {
        "CKPT": "counterfeitV30_v30.safetensors",
        "RENDER": rn, "SMAP": sn, "LOCKMAP": ln,
        "POS": "professional interior photograph, cozy office studio, warm "
               "natural light, realistic materials, wood, carpet, detailed, "
               "architectural photography",
        "NEG": "cartoon, flat shading, lowres, blurry, distorted, watermark, text",
        "RES": max(w, h), "SEED": 7, "STEPS": 28, "CFG": 6.5, "DENOISE": 0.75,
        "CN_DEPTH": 0.7, "CN_LINEART": 0.55,
    })
    print(f"input {w}x{h}, submitting...")
    t = time.time(); res = c.run(wf); dt = time.time() - t
    assert res.images, "no image"
    gen = Image.open(io.BytesIO(res.images[0])).convert("RGB")
    gen.save("real_generated.png")
    print(f"generation: {dt:.1f}s -> real_generated.png ({gen.size})")
    TrueWatercolorEffect().apply(gen).save("real_generated_watercolor.png")
    print("calque watercolor -> real_generated_watercolor.png")


if __name__ == "__main__":
    main()
