"""
Spike: prove the Generator -> Calque chain end to end on the 3070.

  render + precision map  --(ComfyUI: depth+lineart CN + differential diffusion
                             + exact composite)-->  generated image
                          --(Calque watercolor finish)-->  final

Run from worker/ with ComfyUI up on :8188.
"""
import io
import time
from PIL import Image, ImageDraw

from calque.generator import ComfyClient
from calque.postprocess.true_watercolor import TrueWatercolorEffect

W, H = 640, 480
# Building geometry (must match test_render.png) -> the regions we LOCK.
BUILDINGS = [
    ("rect", (80, 220, 260, 420)),
    ("rect", (300, 150, 520, 420)),
    ("poly", [(80, 220), (170, 160), (260, 220)]),
]


def build_maps():
    """soft strength map (white=regenerate, black=anchor) + hard lock (white=lock)."""
    soft = Image.new("L", (W, H), 255)   # default: free / regenerate
    lock = Image.new("L", (W, H), 0)     # default: not locked
    ds, dl = ImageDraw.Draw(soft), ImageDraw.Draw(lock)
    for kind, geo in BUILDINGS:
        if kind == "rect":
            ds.rectangle(geo, fill=0)     # anchor the building during diffusion
            dl.rectangle(geo, fill=255)   # and lock those pixels bit-exact
        else:
            ds.polygon(geo, fill=0)
            dl.polygon(geo, fill=255)
    return soft.convert("RGB"), lock.convert("RGB")


def to_png(img: Image.Image) -> bytes:
    b = io.BytesIO(); img.save(b, format="PNG"); return b.getvalue()


def main():
    c = ComfyClient()
    assert c.is_up(), "ComfyUI not reachable on :8188"

    render = Image.open("test_render.png").convert("RGB")
    soft, lock = build_maps()
    soft.save("map_soft.png"); lock.save("map_lock.png")

    render_name = c.upload_image(to_png(render), "calque_render.png")
    soft_name = c.upload_image(to_png(soft), "calque_soft.png")
    lock_name = c.upload_image(to_png(lock), "calque_lock.png")

    wf = c.load_workflow("structure_control", {
        "CKPT": "counterfeitV30_v30.safetensors",
        "RENDER": render_name, "SMAP": soft_name, "LOCKMAP": lock_name,
        "POS": "architectural visualization, warm golden hour light, detailed "
               "facade, lush landscaping, professional render",
        "NEG": "flat, dull, lowres, blurry, distorted, watermark",
        "RES": 640, "SEED": 42, "STEPS": 25, "CFG": 6.5, "DENOISE": 1.0,
        "CN_DEPTH": 0.6, "CN_LINEART": 0.5,
    })

    print("submitting...")
    t = time.time()
    res = c.run(wf)
    dt = time.time() - t
    assert res.images, "no image returned"
    gen = Image.open(io.BytesIO(res.images[0])).convert("RGB")
    gen.save("out_generated.png")
    print(f"generation: {dt:.1f}s -> out_generated.png  ({gen.size})")

    t = time.time()
    TrueWatercolorEffect().apply(gen).save("out_generated_watercolor.png")
    print(f"calque watercolor finish: {time.time()-t:.1f}s -> out_generated_watercolor.png")


if __name__ == "__main__":
    main()
