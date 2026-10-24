"""Image-space proxies for LIBERO-Plus perturbation categories on the two stored frames.

camera   : random affine (translate / rotate / scale), applied to both views; ROI boxes are transformed
lighting : brightness, contrast, gamma, colour-temperature shifts
noise    : additive Gaussian sensor noise
Writes PNGs and manifest_plus.json (clean records first) next to the base manifest.
"""
import json, os, sys, math
import numpy as np
from PIL import Image, ImageEnhance
from roi_stats import ROIS

BASE = "./scripts/attn_vis"
man = json.load(open(f"{BASE}/manifest.json"))
outdir = f"{BASE}/plus"; os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(0)
records = []


def tag_of(r):
    return f"{r['suite']}_t{r['task_index']}_e{r['episode_index']}_f{r['frame_index']}"


def affine(img, tx, ty, rot, scale):
    w, h = img.size
    # PIL uses inverse mapping: output(x,y) = input(a x + b y + c, d x + e y + f)
    th = math.radians(rot); s = 1.0 / scale
    a, b = s * math.cos(th), s * math.sin(th)
    cx, cy = w / 2, h / 2
    c = cx - a * cx - b * cy - tx; f = cy + b * cx - a * cy - ty
    return img.transform((w, h), Image.AFFINE, (a, b, c, -b, a, f), resample=Image.BILINEAR, fillcolor=None)


def affine_box(box, tx, ty, rot, scale, w=512, h=512):
    th = math.radians(rot); cx, cy = w / 2, h / 2
    pts = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
    out = []
    for x, y in pts:  # forward map: rotate+scale about centre, then translate
        dx, dy = x - cx, y - cy
        xx = cx + scale * (math.cos(th) * dx - math.sin(th) * dy) + tx
        yy = cy + scale * (math.sin(th) * dx + math.cos(th) * dy) + ty
        out.append((xx, yy))
    xs, ys = [p[0] for p in out], [p[1] for p in out]
    return (max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys)))


def lighting(img, kind, v):
    if kind == "brightness": return ImageEnhance.Brightness(img).enhance(v)
    if kind == "contrast": return ImageEnhance.Contrast(img).enhance(v)
    if kind == "gamma":
        a = np.asarray(img).astype(np.float32) / 255.0
        return Image.fromarray((255 * np.power(a, v)).clip(0, 255).astype(np.uint8))
    if kind == "temperature":  # v>0 warm, v<0 cool
        a = np.asarray(img).astype(np.float32); a[..., 0] += 40 * v; a[..., 2] -= 40 * v
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))
    raise ValueError(kind)


def noise(img, sigma, seed):
    r = np.random.default_rng(seed); a = np.asarray(img).astype(np.float32)
    return Image.fromarray((a + r.normal(0, sigma, a.shape)).clip(0, 255).astype(np.uint8))


for r in man:
    base = tag_of(r); rois = ROIS[base]
    rec = dict(r); rec["tag"] = base + "_clean"; rec["category"] = "clean"; rec["strength"] = 0; rec["rois"] = rois
    records.append(rec)
    agent = Image.open(r["image"]).convert("RGB"); wrist = Image.open(r["wrist_image"]).convert("RGB")
    variants = []
    # camera: three strengths x two seeds
    for k, (tmax, rmax, smax) in enumerate([(20, 3, 0.05), (40, 6, 0.10), (60, 9, 0.15)], start=1):
        for seed in range(2):
            g = np.random.default_rng(100 * k + seed)
            p = dict(tx=float(g.uniform(-tmax, tmax)), ty=float(g.uniform(-tmax, tmax)), rot=float(g.uniform(-rmax, rmax)), scale=float(g.uniform(1 - smax, 1 + smax)))
            q = dict(tx=float(g.uniform(-tmax, tmax)), ty=float(g.uniform(-tmax, tmax)), rot=float(g.uniform(-rmax, rmax)), scale=float(g.uniform(1 - smax, 1 + smax)))
            variants.append(("camera", k, f"cam{k}s{seed}", affine(agent, **p), affine(wrist, **q), {n: affine_box(b, **p) for n, b in rois.items()}))
    for k, (kind, v) in enumerate([("brightness", 0.6), ("brightness", 1.5), ("contrast", 0.6), ("gamma", 0.6), ("gamma", 1.6), ("temperature", 1.0)], start=1):
        variants.append(("lighting", k, f"light_{kind}{v}", lighting(agent, kind, v), lighting(wrist, kind, v), rois))
    for k, (sig, seed) in enumerate([(8, 0), (8, 1), (16, 0), (16, 1), (32, 0), (32, 1)], start=1):
        variants.append(("noise", int(sig), f"noise{sig}s{seed}", noise(agent, sig, seed), noise(wrist, sig, 10 + seed), rois))
    for cat, strength, name, a_img, w_img, boxes in variants:
        tag = f"{base}_{name}"
        pa = f"{outdir}/{tag}_image.png"; pw = f"{outdir}/{tag}_wrist_image.png"
        a_img.save(pa); w_img.save(pw)
        rec = dict(r); rec.update({"tag": tag, "category": cat, "strength": strength, "image": pa, "wrist_image": pw, "rois": {n: [float(x) for x in b] for n, b in boxes.items()}})
        records.append(rec)
json.dump(records, open(f"{BASE}/manifest_plus.json", "w"), indent=1)
print(len(records), "records;", sum(r["category"] == "clean" for r in records), "clean")
