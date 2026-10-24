"""Figure 4 cells as the attention each cell receives above the policy's own average over the 100 captured frames.

excess = max(A - mean_frames(A), 0), in units of the uniform density 1/196 (each frame's map first normalized
within its view).  Attention that every frame gives to the same positions, such as the Yak18 sink columns,
cancels; nothing is removed and no cell is masked.  Policy cells: jet overlay, 99th-percentile clip.  The
difference cell is the signed surplus of the Yak18 JAM minus that of the native policy on a fixed scale.
Policy cells use a compressive colour scale (surplus ** GAMMA after the 99th-percentile clip) so weaker regions stay visible.
"""
import sys, os
sys.path.insert(0, './scripts/attn_vis')
import numpy as np
from PIL import Image
from matplotlib import cm
from make_figure import overlay
ROOT, TAG, OUT = sys.argv[1:4]; VMAX = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0; T0 = 0.5  # differences below half a uniform density stay grey
GAMMA = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
FAMS = ['native', 'yak', 'dino', 'dinol', 'qwen']; SZ = 448
def lm(f, t):
    d = np.load(os.path.join(ROOT, f, t + '.npz'), allow_pickle=True); keep = np.asarray(d['layers']) < 100
    m = d['maps'][keep].mean(axis=(0, 1)); return d, m / m.sum(axis=(1, 2), keepdims=True)
tags = sorted(x[:-4] for x in os.listdir(os.path.join(ROOT, 'native')) if x.endswith('.npz'))
maps = {f: {t: lm(f, t)[1] for t in tags} for f in FAMS}
mean = {f: np.mean([maps[f][t] for t in tags], axis=0) for f in FAMS}
d0, _ = lm('native', TAG)
def surplus(f, t):                       # signed, in multiples of the uniform density
    m = maps[f][t]; return (m - mean[f]) * m[0].size
# scale for the difference panel: spread of (JAM - native) surplus over all 100 frames
alld = np.concatenate([(surplus('yak', t) - surplus('native', t)).ravel() for t in tags])
print('difference of surpluses over 100 frames: 95th pct |d| = %.2f, 99th = %.2f, 99.9th = %.2f, max = %.2f'
      % tuple(np.percentile(np.abs(alld), [95, 99, 99.9, 100])))
os.makedirs(OUT, exist_ok=True); rows = []
for v, vn in enumerate(('agent', 'wrist')):
    img = np.asarray(d0[vn]); big = np.asarray(Image.fromarray(img).resize((SZ, SZ), Image.BICUBIC))
    Image.fromarray(big).save(f'{OUT}/M_{vn}_image.png'); cells = [big]
    for f in FAMS:
        e = np.maximum(surplus(f, TAG)[v], 0); e = np.clip(e / (np.percentile(e, 99) + 1e-9), 0, 1) ** GAMMA
        c = (overlay(big, e, 0.55, clip_pct=100.0) * 255).round().astype(np.uint8); Image.fromarray(c).save(f'{OUT}/M_{vn}_{f}.png'); cells.append(c)
    d = surplus('yak', TAG)[v] - surplus('native', TAG)[v]
    print('%s: surplus max native %.2f yak %.2f | diff min %.2f max %.2f' % (vn, surplus('native', TAG)[v].max(), surplus('yak', TAG)[v].max(), d.min(), d.max()))
    heat = np.asarray(Image.fromarray(d.astype(np.float32)).resize((SZ, SZ), Image.BICUBIC))
    gray = np.asarray(Image.fromarray(big).convert('L')).astype(np.float32) / 255.0
    base = 0.75 * np.stack([gray] * 3, -1) + 0.15
    col = cm.get_cmap('RdBu_r')(np.clip((heat + VMAX) / (2 * VMAX), 0, 1))[..., :3]; a = 0.85 * np.clip((np.abs(heat) - T0) / (VMAX - T0), 0, 1) ** 0.6
    c = (np.clip((1 - a[..., None]) * base + a[..., None] * col, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(c).save(f'{OUT}/M_{vn}_diff.png'); cells.insert(3, c)
    rows.append(np.concatenate(cells, axis=1))
Image.fromarray(np.concatenate(rows, axis=0)).save(f'{OUT}/montage.png')
print('cells written to', OUT, '| montage columns: image | native | yak | diff | dino | dinol | qwen')
