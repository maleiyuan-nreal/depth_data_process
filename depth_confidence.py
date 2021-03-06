"""
Auther: lyma@nreal.ai
Date: 2022/06/07

Description: NDS data for depth estimation pipeline
    feature 1: data process by line
    feature 2: one dateset for one NDS file
    feature 3: using multi process

"""
import os
import time
import logging
import multiprocessing as mp
import argparse


import cv2
import numpy as np
from tqdm import tqdm

from bfuncs import check_and_make_dir
from bfuncs.files import load_json_items, save_json_items
from utils import transform_rgb


def func_callback(args, pbar, left_err_map, right_err_map, nds_info):
    pbar.update()
    # error_str = ["{:.4f}".format(e) for e in args[0]["ratio"]]
    # result_file.write("\n".join(error_str))
    # result_file.write("\n")
    mask_info = args[0]["mask"]
    left_nds_item = args[0]["left_nds_item"]
    right_nds_item = args[0]["right_nds_item"]
    for mask_item in mask_info:
        cv2.imwrite(mask_item['path'], mask_item['mask'])
        if mask_item['camera_name'] == "left":
            left_err_map += mask_item['mask']
        if mask_item['camera_name'] == "right":
            right_err_map += mask_item['mask']
    nds_info.append(left_nds_item)
    nds_info.append(right_nds_item)


def func_core(task_info):
    args, data_path, left_nds_item, right_nds_item, nreal_glass_info = task_info
    left_image_path = os.path.join(data_path, left_nds_item['image_path'])
    left_depth_path = os.path.join(data_path, left_nds_item['depth_path'])

    left_image = cv2.imread(left_image_path)
    left_depth = cv2.imread(left_depth_path, cv2.IMREAD_UNCHANGED)
    left_depth = left_depth.astype(np.float32) / 1000

    right_image_path = os.path.join(data_path, right_nds_item['image_path'])
    right_depth_path = os.path.join(data_path, right_nds_item['depth_path'])

    right_image = cv2.imread(right_image_path)
    right_depth = cv2.imread(right_depth_path, cv2.IMREAD_UNCHANGED)
    right_depth = right_depth.astype(np.float32) / 1000

    left_rgb_trans = transform_rgb.project_rgb(
        left_depth, right_image, nreal_glass_info, need_inverse=True)
    right_rgb_trans = transform_rgb.project_rgb(
        right_depth, left_image, nreal_glass_info)

    mask_unvisible_left = left_rgb_trans == 0
    mask_left = abs(left_image.astype("float32") -
                    left_rgb_trans.astype("float32"))
    mask_left[mask_unvisible_left] = 0
    mask_left[mask_left > args.max_threshold] = args.max_threshold
    mask_left = mask_left[..., 0]

    mask_unvisible_right = right_rgb_trans == 0
    mask_right = abs(right_image.astype("float32") -
                     right_rgb_trans.astype("float32"))
    mask_right[mask_unvisible_right] = 0
    mask_right[mask_right > args.max_threshold] = args.max_threshold
    mask_right = mask_right[..., 0]

    chosen_left_mask = mask_left > args.min_threshold
    chosen_right_mask = mask_right > args.min_threshold
    left_error_pixel_num = sum(sum(chosen_left_mask))
    left_error_ratio = left_error_pixel_num / \
        (mask_left.shape[0] * mask_left.shape[1])

    right_error_pixel_num = sum(sum(chosen_right_mask))
    right_error_ratio = right_error_pixel_num / \
        (mask_right.shape[0] * mask_right.shape[1])
    left_mask_path = left_image_path.replace(
        "left_image", "left_mask").replace(".jpg", ".png")
    right_mask_path = right_image_path.replace(
        "right_image", "right_mask").replace(".jpg", ".png")
    check_and_make_dir(os.path.dirname(left_mask_path))
    check_and_make_dir(os.path.dirname(right_mask_path))
    mask_info = [
        {"path": left_mask_path, "mask": chosen_left_mask.astype(
            "uint8"), "camera_name": "left"},
        {"path": right_mask_path, "mask": chosen_right_mask.astype(
            "uint8"), "camera_name": "right"}
    ]

    left_nds_item['mask_path'] = os.path.relpath(left_mask_path, data_path)
    right_nds_item['mask_path'] = os.path.relpath(right_mask_path, data_path)
    ratio_info = [left_error_ratio, right_error_ratio]
    return {"ratio": ratio_info, "mask": mask_info, "left_nds_item": left_nds_item, "right_nds_item": right_nds_item}


def main(args):
    time_start = time.time()

    logging.info(f"n_proc: {args.n_proc}")

    root_dir = "/data/lyma/data_depth/nreal_light"
    glass_info_path = os.path.join(
        root_dir, "nreal_glasses/glasses_info_20220415_json")
    data_path = os.path.join(
        root_dir, "training_data/nreal_light_v4.0.0_ssl_2s/train_labeled_by_ssl_20220630")
    mini_nds_path = os.path.join(data_path, "annotation_add_nreal_name.nds")
    mini_nds_info = load_json_items(mini_nds_path)
    left_list = [
        nds_item for nds_item in mini_nds_info if nds_item['camera_name'] == "left"]
    right_list = [
        nds_item for nds_item in mini_nds_info if nds_item['camera_name'] == "right"]

    pbar = tqdm(total=len(left_list))

    pool = mp.Pool(args.n_proc)
    left_err_map = np.zeros((480, 640))
    right_err_map = np.zeros((480, 640))
    nds_info = list()
    call_back = lambda *args: func_callback(args, pbar, left_err_map, right_err_map, nds_info)
    for left_nds_item, right_nds_item in zip(left_list, right_list):
        assert left_nds_item['nreal_name'] == right_nds_item['nreal_name']
        assert left_nds_item['camera_name'] == "left"
        assert right_nds_item['camera_name'] == "right"

        left_image_name = os.path.basename(left_nds_item['image_path'])
        right_image_name = os.path.basename(right_nds_item['image_path'])
        assert left_image_name.replace("l", "r") == right_image_name

        nreal_name = left_nds_item['nreal_name']
        nreal_glass_path = os.path.join(glass_info_path, nreal_name+".json")
        nreal_glass_info = load_json_items(nreal_glass_path)[0]

        task_info = [args, data_path, left_nds_item,
                     right_nds_item, nreal_glass_info]
        pool.apply_async(func_core, (task_info, ),
                         callback=call_back)

    pool.close()
    pool.join()

    cv2.imwrite(
        "result/left_err_map_{}.png".format(args.min_threshold), left_err_map)
    cv2.imwrite(
        "result/right_err_map_{}.png".format(args.min_threshold), right_err_map)

    nds_path = os.path.join(
        data_path, "annotation_add_nreal_name_add_mask.nds")
    save_json_items(nds_path, nds_info)

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

    parser.add_argument('--min_threshold',
                        default=40,
                        type=int,
                        help='min_threshold'
                        )
    parser.add_argument('--max_threshold',
                        default=80,
                        type=int,
                        help='max_threshold'
                        )

    args = parser.parse_args()

    log_filename = "log/confidence_{}.log".format(args.min_threshold)
    # result_filename = "result/confidence_{}.txt".format(args.min_threshold)
    # result_file = open(result_filename, 'a+')
    logging.basicConfig(filename=log_filename, format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)

    main(args)

    # result_file.close()
