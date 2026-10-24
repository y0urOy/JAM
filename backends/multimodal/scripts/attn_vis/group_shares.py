"""Mean share of the action expert's attention on text / image / action keys, over the captured frames.

Reads the `group_mass` array [layers, steps, 4] each capture writes (text, visual, action, other), keeps the
recorded layers, normalizes each layer-step row to 1, and averages.  Families are read from one or more capture
directories, so a re-captured family can be merged with an earlier run.
    python group_shares.py out.json  dir1:fam1,fam2  dir2:fam3,...
"""
import sys, os, glob, json
import numpy as np
OUT = sys.argv[1]; res = {}
for spec in sys.argv[2:]:
    root, fams = spec.split(':'); 
    for fam in fams.split(','):
        vals, shown = [], None
        for f in sorted(glob.glob(os.path.join(root, fam, '*.npz'))):
            d = np.load(f, allow_pickle=True)
            if 'group_mass' not in d.files: print('%-7s %s has no group_mass' % (fam, os.path.basename(f))); break
            g = np.asarray(d['group_mass'], dtype=np.float64)
            keep = np.asarray(d['layers']) < 100
            g = g[keep]
            if not np.isfinite(g).all(): print('%-7s %s has non-finite group_mass' % (fam, os.path.basename(f))); break
            g = g / np.clip(g.sum(axis=-1, keepdims=True), 1e-9, None)
            v = g.mean(axis=(0, 1)); vals.append(v)
            if 'e272_f80' in f: shown = v
        if not vals: continue
        a = np.stack(vals)
        res[fam] = dict(mean=[round(float(x), 3) for x in a.mean(0)], std=[round(float(x), 3) for x in a.std(0)],
                        shown=[round(float(x), 3) for x in (shown if shown is not None else a.mean(0))], n=len(vals))
        print('%-7s n=%3d  text %.3f  image %.3f  action %.3f  other %.3f  (shown frame: %s)'
              % (fam, len(vals), *a.mean(0), ' '.join('%.3f' % x for x in res[fam]['shown'])))
json.dump(res, open(OUT, 'w'), indent=1); print('wrote', OUT)
