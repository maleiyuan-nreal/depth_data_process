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


from bfuncs import (
    check_and_make_dir, get_file_name
)

from config.data_config import init_obj
from utils.process_segmentaion import process_segmentation
from utils.process_depth import process_depth
from utils.process_ori_image import process_ori_image
from utils.format_nds import format_nds


def collect_result(args, pbar, nds_data):
    pbar.update()
    nds_data.append(args[0])


def func_core(task_info):
    path_dict, image_id, obj = task_info
    image = cv2.imread(path_dict["ori_images_path"])

    for data_type in obj.DATA_TYPE_LIST:
        if data_type == "depths":
            process_depth(args, path_dict["ori_depths_path"], os.path.join(
                obj.NAME, path_dict["output_depths_path"]))
        elif data_type == "segmentations":
            process_segmentation(args, path_dict["ori_segmentations_path"], os.path.join(
                obj.NAME, path_dict["output_segmentations_path"]))
        elif data_type == "images":
            process_ori_image(args, path_dict["ori_images_path"], os.path.join(
                obj.NAME, path_dict["output_images_path"]))
        else:
            raise NotImplementedError

    nds_data_item = format_nds(image_id, image.shape, path_dict)
    return nds_data_item


def main(args):
    time_start = time.time()

    logging.info(f"n_proc: {args.n_proc}")

    check_and_make_dir(args.output_path)
    logging.info(f"image_output_dir: {args.output_path}")

    obj_dict = init_obj(args.output_path)

    if args.dataset != "ALL":
        assert args.dataset in list(obj_dict.keys())
        obj_dict[args.dataset].process(args, func_core, collect_result)
    elif args.dataset == "ALL":
        for dataset_name in list(obj_dict.keys()):
            obj_dict[dataset_name].process(
                obj_dict[args.dataset], args, func_core, collect_result)

    else:
        raise NotImplementedError

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

    parser.add_argument('-o', '--output_path',
                        default='output',
                        help='folder with output images and nds file'
                        )

    parser.add_argument('-n', '--n_proc',
                        default=30,
                        type=int,
                        help='mp process number'
                        )

    parser.add_argument('-d', '--dataset',
                        default="ALL",
                        help='choose dataset name to process, default way is process all dataset'
                        )

    args = parser.parse_args()

    log_filename = "log/track_{}.log".format(args.dataset)
    logging.basicConfig(filename=log_filename, format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)

    main(args)
