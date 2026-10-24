"""Every frame (or every STRIDE-th frame) of one LIBERO episode, in the record format of make_manifest_many.py,
for an attention video along a rollout.

Default: LIBERO-Long episode 272 ("put the white mug on the left plate and put the yellow and white mug on the
right plate"), the episode behind the frame shown in Figure 5 of the paper.

Usage: python make_manifest_episode.py --out ./artifacts/jam_multimodal/attn_vis/episode_frames [--suite libero_10]
       [--episode 272] [--stride 1]
"""
import argparse, json, os
import numpy as np, av, pyarrow.parquet as pq
from PIL import Image

DR = './data/libero_mujoco3.3.2'
AHEAD = 16


def decode_all(path):
    with av.open(path) as c:
        return [fr.to_ndarray(format='rgb24') for fr in c.decode(video=0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--suite', default='libero_10')
    ap.add_argument('--episode', type=int, default=272)
    ap.add_argument('--stride', type=int, default=1)
    a = ap.parse_args()
    suite = f'{a.suite}_no_noops_lerobot'
    root = f'{DR}/{suite}'
    info = json.load(open(f'{root}/meta/info.json'))
    tasks = {t['task_index']: t['task'] for t in map(json.loads, open(f'{root}/meta/tasks.jsonl'))}
    ep_meta = next(e for e in map(json.loads, open(f'{root}/meta/episodes.jsonl')) if e['episode_index'] == a.episode)
    ep, n = a.episode, int(ep_meta['length'])
    chunk = ep // info['chunks_size']
    tab = pq.read_table(root + '/' + info['data_path'].format(episode_chunk=chunk, episode_index=ep)).to_pandas()
    ti = int(tab['task_index'].iloc[0])
    vids = {k: root + '/' + info['video_path'].format(episode_chunk=chunk, video_key=k, episode_index=ep)
            for k in ('observation.images.image', 'observation.images.wrist_image')}
    imgs = {k: decode_all(v) for k, v in vids.items()}
    assert len(imgs['observation.images.image']) >= n and len(tab) == n, (len(imgs['observation.images.image']), len(tab), n)
    fdir = os.path.join(a.out, 'frames'); os.makedirs(fdir, exist_ok=True)
    records = []
    for fi in range(0, n, a.stride):
        tag = f"{a.suite}_t{ti}_e{ep}_f{fi}"
        pa = os.path.join(fdir, tag + '_image.png'); pw = os.path.join(fdir, tag + '_wrist_image.png')
        Image.fromarray(imgs['observation.images.image'][fi]).save(pa)
        Image.fromarray(imgs['observation.images.wrist_image'][fi]).save(pw)
        records.append({'suite': suite, 'task_index': ti, 'instruction': tasks[ti], 'episode_index': ep, 'frame_index': fi,
                        'episode_length': n, 'phase': round(fi / (n - 1), 3), 'future_frame_index': min(n - 1, fi + AHEAD),
                        'state': [float(x) for x in np.asarray(tab['observation.state'].iloc[fi])],
                        'image': pa, 'image_shape': list(imgs['observation.images.image'][fi].shape),
                        'wrist_image': pw, 'wrist_image_shape': list(imgs['observation.images.wrist_image'][fi].shape),
                        'tag': tag})
    json.dump(records, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)
    print(f"{suite} task {ti} '{tasks[ti]}' episode {ep} length {n} stride {a.stride}: wrote {len(records)} records to {a.out}/manifest.json")


if __name__ == '__main__':
    main()
