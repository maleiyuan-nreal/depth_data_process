import time
import logging
import multiprocessing as mp
import argparse
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import h5py


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth


i, j, h, w = 130, 10, 240, 1200


def func_callback(args, pbar):
    pbar.update()
    for t in args[0]:
        hist_dict[t] += 1

def func_core(task_info):
    global hist_dict
    depth = task_info[0]
    depth = depth[i:i + h, j:j + w].astype("float32")
    depth = np.array(Image.fromarray(depth).resize(
        (768, 224), resample=Image.NEAREST))
    depth_np_flatten = depth.reshape(-1, 1)
    res = list(map(lambda x: int(x*10), depth_np_flatten))
    return res
    

def main(args):
    time_start = time.time()

    logging.info(f"n_proc: {args.n_proc}")

    kitti_train_path = "/home/lyma/DepthEstimation/data/kitti/train/*/*.h5"
    kitti_val_path = "/home/lyma/DepthEstimation/data/kitti/val/*/*.h5"

    kitti_data = glob.glob(kitti_train_path)

    pbar = tqdm(total=len(kitti_data))

    pool = mp.Pool(args.n_proc)
    call_back = lambda *args: func_callback(args, pbar)
    for item in tqdm(kitti_data):
        rgb, depth = h5_loader(item)

        task_info = [depth]
        # func_core(task_info)
        pool.apply_async(func_core, (task_info, ),
                         callback=call_back)

    pool.close()
    pool.join()

    np.save("hist_dict_train.npy", hist_dict)

    time_end = time.time()
    time_cost = time_end - time_start
    time_cost_min = time_cost / 60
    time_cost_hour = time_cost_min / 60

    logging.info("")
    logging.info("done")
    logging.info(
        f"time_cost: {time_cost} s == {time_cost_min} m == {time_cost_hour} h")
    logging.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--n_proc',
                        default=20,
                        type=int,
                        help='mp process number'
                        )

    args = parser.parse_args()

    log_filename = "log/depth_dist.log"
    logging.basicConfig(filename=log_filename, format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)

    hist_dict = defaultdict(int)
    main(args)
