import os
import pandas
import numpy as np

folders = [f for f in os.listdir('assets/loop_edges1-5') if f.startswith('result_')]
print(folders)

kitti_root = 'C:/Users/Tymon/Documents/Projects/kitti'
euroc_root = 'C:/Users/Tymon/Documents/Projects/euroc'

name2data = {
    # 'result_euroc-MH01': 'MH_01_easy/mav0/cam0',
    # 'result_kitti00': '2011_10_03/2011_10_03_drive_0027_sync/image_02',
    'result_kitti05': '2011_09_30/2011_09_30_drive_0018_sync/image_02'
}

def kitti_ts2file(datadir, ts):
    df = pandas.read_csv(f'{kitti_root}/{datadir}/timestamps.txt', header=None, names=['timestamp'], parse_dates=['timestamp'])
    timestamps = df.values.reshape(-1)
    seconds_from_start = (timestamps - timestamps[0]) / np.timedelta64(1, 's')

    ts = ts.reshape(-1)
    index = np.array([np.where(np.abs(seconds_from_start - t) < 0.01)[0][0] for t in ts])
    index = index.reshape(-1, 2)
    
    pairs = []
    for i in index:
        pairs.append(f'{kitti_root}/{datadir}/data/{i[0]:0>10d}.png {kitti_root}/{datadir}/data/{i[1]:0>10d}.png')
    return np.array(pairs)


def euroc_ts2file(datadir, ts):
    df = pandas.read_csv(f'{euroc_root}/{datadir}/data.csv')
    timestamps = df.values[:, 0] / 1e9
    filenames = df.values[:, 1].astype(str)

    ts = ts.reshape(-1)
    index = np.array([np.where(np.abs(timestamps - t) < 0.01)[0][0] for t in ts])

    names = filenames[index].reshape(-1, 2)

    pairs = []
    for n in names:
        pairs.append(f'{euroc_root}/{datadir}/data/{n[0]} {euroc_root}/{datadir}/data/{n[1]}')
    return np.array(pairs)

for folder in folders:
    if folder not in name2data:
        continue
    datadir = name2data[folder]

    res = np.loadtxt(f'assets/loop_edges1-5/{folder}/loop_final.txt', delimiter=' ')
    if len(res.shape) == 1:
        res = res[np.newaxis, :]
    ts = res[:, 2:]

    if 'euroc' in folder:
        pairs = euroc_ts2file(datadir, ts)
        full_datadir = f'{euroc_root}/{datadir}/data'
    if 'kitti' in folder:
        pairs = kitti_ts2file(datadir, ts)
        full_datadir = f'{kitti_root}/{datadir}/data'

    pairfile = f'assets/loop_edges1-5/{folder}/loop_pairs.txt'
    np.savetxt(pairfile, pairs, fmt='%s')

    outputdir = f'assets/loop_edges1-5/{folder}/superglue'
    cmd = f'python match_pairs.py --input_dir {full_datadir} --input_pairs {pairfile} --output_dir {outputdir} --viz --resize 640 --superglue indoor --max_keypoints 1024 --nms_radius 4'
    print(cmd)
    os.system(cmd)
