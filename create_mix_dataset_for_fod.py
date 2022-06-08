"""
Auther: lyma@nreal.ai
Date: 2022/06/07

Description: NDS data for depth estimation pipeline
    feature 1: data process by line
    feature 2: one dateset for one NDS file
    feature 3: using multi process

"""
import os
import copy
import time
import shutil
import json
from tqdm import tqdm
import logging
import multiprocessing as mp
import glob


import cv2
import numpy as np


from bfuncs import (
    check_and_make_dir, check_and_make_dir,
    get_file_name
)
from process import process_inria, process_nyuv2, process_posetrack
from config.data_config import IMAGE_OUTPUT_DIR
from utils.process_segmentaion import process_segmentation
from utils.process_depth import process_depth
from utils.process_ori_image import process_ori_image
from utils.format_nds import format_nds


def write_nds_file(args, pbar, nds_path):
    pbar.update()
    write_file_obj = open(nds_path, "a+")
    write_file_obj.write(json.dumps(args[0][0]))
    write_file_obj.write("\n")
    write_file_obj.close()


def func_core(task_info):
    ori_image_path, image_id, obj = task_info

    nds_data = list()
    nds_data_item = dict()
    image = cv2.imread(ori_image_path)
    image_name_only = get_file_name(ori_image_path)

    ori_depth_path = os.path.join(obj.INPUT_DIR, obj.NAME,  "depths", image_name_only)
    ori_seg_path = os.path.join(obj.INPUT_DIR, obj.NAME,  "segmentations", image_name_only.split(".")[0]+".png")
    # 保存相对路径即可
    output_image_path = os.path.join(obj.NAME,  "images", image_name_only)
    ouput_depth_path = os.path.join(obj.NAME, "depths", image_name_only.split(".")[0]+".png")
    output_seg_path = os.path.join(obj.NAME, "segmentations", image_name_only.split(".")[0]+".png")

    process_depth(ori_depth_path, ouput_depth_path)
    process_segmentation(ori_seg_path, output_seg_path)
    process_ori_image(ori_image_path, output_image_path)

    nds_info = [image_id, output_image_path, ouput_depth_path, output_seg_path, image.shape]
    nds_data_item = format_nds(nds_info)
    nds_data.append(nds_data_item)
    return nds_data


def main():
    time_start = time.time()

    n_proc = 20
    logging.info(f"n_proc: {n_proc}")

    check_and_make_dir(IMAGE_OUTPUT_DIR)
    logging.info(f"image_output_dir: {IMAGE_OUTPUT_DIR}")

    process_inria.process(n_proc, func_core, write_nds_file)
    process_nyuv2.process(n_proc, func_core, write_nds_file)
    process_posetrack.process(n_proc, func_core, write_nds_file)

    time_end = time.time()
    time_cost = time_end - time_start
    time_cost_min = time_cost / 60
    time_cost_hour = time_cost_min / 60

    logging.info("")
    logging.info("done")
    logging.info(f"time_cost: {time_cost} s == {time_cost_min} m == {time_cost_hour} h")
    logging.info("")


if __name__ == "__main__":
    logging.basicConfig(filename="log/track.log", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)

    main()
