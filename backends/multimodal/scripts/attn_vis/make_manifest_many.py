"""A larger, mechanically chosen frame set for the attention analysis.

For every task of LIBERO-Goal and LIBERO-Long (libero_10), the first episode of that task in the dataset, and
five frames at 10/30/50/70/90% of its length.  Each record carries the agent and wrist frames (decoded with
PyAV, as-is orientation), the 8-dim proprioceptive state, the instruction, and the index of the frame 16 steps
ahead (clipped to the episode) for later future-change statistics.

Usage: python make_manifest_many.py --out ./artifacts/jam_multimodal/attn_vis/many_frames
"""
import argparse, json, os
import numpy as np, av, pyarrow.parquet as pq
from PIL import Image

DR = './data/libero_mujoco3.3.2'
SUITES = ['libero_goal_no_noops_lerobot', 'libero_10_no_noops_lerobot']
PHASES = [0.1, 0.3, 0.5, 0.7, 0.9]; AHEAD = 16

def decode(path, wanted):
    out = {}; wanted = set(wanted)
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in wanted: out[i] = fr.to_ndarray(format='rgb24')
            if len(out) == len(wanted): break
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    fdir = os.path.join(a.out, 'frames'); os.makedirs(fdir, exist_ok=True); records = []
    for suite in SUITES:
        root = f'{DR}/{suite}'; info = json.load(open(f'{root}/meta/info.json'))
        tasks = {t['task_index']: t['task'] for t in map(json.loads, open(f'{root}/meta/tasks.jsonl'))}
        eps = [json.loads(l) for l in open(f'{root}/meta/episodes.jsonl')]
        first = {}
        for e in eps:                                   # first episode of each task
            t = e['tasks'][0]
            if t not in first: first[t] = e
        for ti in sorted(tasks):
            e = first[tasks[ti]]; ep = e['episode_index']; n = e['length']; chunk = ep // info['chunks_size']
            tab = pq.read_table(root + '/' + info['data_path'].format(episode_chunk=chunk, episode_index=ep)).to_pandas()
            assert int(tab['task_index'].iloc[0]) == ti
            frames = sorted({min(n - 1, int(round(p * (n - 1)))) for p in PHASES})
            vids = {k: root + '/' + info['video_path'].format(episode_chunk=chunk, video_key=k, episode_index=ep) for k in ('observation.images.image', 'observation.images.wrist_image')}
            imgs = {k: decode(v, frames) for k, v in vids.items()}
            for fi in frames:
                tag = f"{suite.split('_no_noops')[0]}_t{ti}_e{ep}_f{fi}"
                pa = os.path.join(fdir, tag + '_image.png'); pw = os.path.join(fdir, tag + '_wrist_image.png')
                Image.fromarray(imgs['observation.images.image'][fi]).save(pa); Image.fromarray(imgs['observation.images.wrist_image'][fi]).save(pw)
                records.append({'suite': suite, 'task_index': ti, 'instruction': tasks[ti], 'episode_index': ep, 'frame_index': fi, 'episode_length': n,
                                'phase': round(fi / (n - 1), 3), 'future_frame_index': min(n - 1, fi + AHEAD),
                                'state': [float(x) for x in np.asarray(tab['observation.state'].iloc[fi])],
                                'image': pa, 'image_shape': [512, 512, 3], 'wrist_image': pw, 'wrist_image_shape': [512, 512, 3]})
            print(suite, ti, tasks[ti][:40], 'episode', ep, 'len', n, 'frames', frames)
    json.dump(records, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)
    print('wrote', len(records), 'records to', os.path.join(a.out, 'manifest.json'))

if __name__ == '__main__':
    main()
