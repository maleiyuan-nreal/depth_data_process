"""
tartanair数据只用了左目+Easy
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm

import cv2
import numpy as np
from bfuncs import (
    check_and_make_dir, save_json_items
)
from process.common_process import Base_Data
from config import nreal_param


class TartanAir(Base_Data):
    def __init__(self, output_path, transform_flag) -> None:
        self.NAME = "tartanair"
        self.INPUT_DIR = "/data/lyma/"
        if transform_flag:
            self.NDS_FILE_NAME = os.path.join(
                output_path, self.NAME, "annotation_to_nreal.nds")
            self.OUTPUT_DIR = "data_2_nreal" if transform_flag else "data"
        else:
            self.NDS_FILE_NAME = os.path.join(
                output_path, self.NAME, "annotation.nds")
            self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["image_left", "depth_left"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "npy"
        self.IMAGE_SUFFIX = "png"
        self.K = np.float32([[320,  0., 320],
                            [0.,  320, 240],
                            [0.,  0.,  1.]])

    def process(self, args, func_callback):
        self.common_process(args)
        p = os.path.join(self.INPUT_DIR, self.NAME+"/*/Easy")
        dir_list = glob.glob(p)
        nds_file_list = list()
        sample_num = 0
        if args.transform:
            map1_x, map1_y = cv2.initUndistortRectifyMap(cameraMatrix=self.K, distCoeffs=None, R=None, newCameraMatrix=nreal_param.nreal_left_K, size=(
                nreal_param.nreal_col, nreal_param.nreal_row), m1type=cv2.CV_32FC1)
        for dir_num, d in enumerate(dir_list):
            for sub_d in glob.glob(d+"/P*"):
                img_list = glob.glob(sub_d+"/image_left/*")
                rel_sub_d = os.path.relpath(sub_d, os.path.join(
                    self.INPUT_DIR, self.NAME))
                sub_nds_file = os.path.join(
                    args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d, "annotation.nds")
                logging.info(f"sub_nds_file: {sub_nds_file}")

                pbar = tqdm(total=len(img_list))
                pbar.set_description(
                    "Creating {} nds dataset: ".format(rel_sub_d))

                for dirs in self.DATA_TYPE_LIST:
                    check_and_make_dir(os.path.join(
                        args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d, dirs))

                pool = mp.Pool(args.n_proc)
                nds_data = list()
                call_back = lambda *args: func_callback(args, pbar, nds_data)
                for _, ori_image_path in enumerate(img_list):
                    path_dict = self.get_path(ori_image_path, rel_sub_d)
                    if args.transform:
                        task_info = [args, map1_x,
                                     map1_y, path_dict, sample_num]
                        # nds_data_item = self.tartanair_data_2_nreal_core(task_info)
                        pool.apply_async(self.tartanair_data_2_nreal_core, (task_info, ),
                                         callback=call_back)
                    else:
                        task_info = [args, path_dict, sample_num]
                        # nds_data_item = self.func_core(task_info)
                        pool.apply_async(self.func_core, (task_info, ),
                                         callback=call_back)
                    sample_num += 1

                pool.close()
                pool.join()

                nds_data.sort(key=lambda x: x['image_id'])
                save_json_items(sub_nds_file, nds_data)
                nds_file_list.append(sub_nds_file)
            assert len(img_list) == len(nds_data)
            logging.info(
                "Total dirs {}, currently {}/{}".format(len(dir_list), dir_num+1, len(dir_list)))

        self.mergeFiles(nds_file_list)
        logging.info("sub_nds_file merged!")

    def tartanair_data_2_nreal_core(self, task_info):
        args,  map1_x, map1_y, path_dict, image_id = task_info
        image = cv2.imread(path_dict["ori_images_path"])
        ori_depth_path = path_dict["ori_depths_path"]
        output_depth = np.load(ori_depth_path)
        output_depth = np.clip(output_depth, 0, 50)
        output_depth = (output_depth * 1000).astype("uint16")

        data_2_nreal_info = [args, image, output_depth,
                             map1_x, map1_y, path_dict, image_id]
        nds_data_item = self.data_2_nreal_core(data_2_nreal_info)
        return nds_data_item
