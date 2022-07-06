"""
blendedmvs数据只用了high_res
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm
import shutil


import numpy as np
import cv2

from bfuncs import (
    check_and_make_dir, save_json_items,
)
from utils.process_depth import load_pfm, load_exr
from utils.format_nds import format_nds
from process.common_process import Base_Data
from config import nreal_param


class BlendedMVS(Base_Data):
    def __init__(self, output_path, transform_flag) -> None:
        super().__init__()
        self.NAME = "BlendedMVS"
        self.INPUT_DIR = "/home/lyma/SHARE_DATA/datadepth"
        if transform_flag:
            self.NDS_FILE_NAME = os.path.join(
                output_path, self.NAME, "annotation_to_nreal.nds")
            self.OUTPUT_DIR = "data_2_nreal"
        else:
            self.NDS_FILE_NAME = os.path.join(
                output_path, self.NAME, "annotation.nds")
            self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["blended_images", "rendered_depth_maps"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.IMAGE_SUFFIX = "jpg"
        self.DPETH_SUFFIX = "pfm"

    def mask_depth_image(self, depth_image, min_depth, max_depth):
        """ mask out-of-range pixel to zero """
        ret, depth_image = cv2.threshold(
            depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
        ret, depth_image = cv2.threshold(
            depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
        return depth_image
    
    
    def blendedmvs_depth_read(self, ori_depth_path, depth_start, depth_end):
        output_depth = load_pfm(open(ori_depth_path, 'rb'))
        depth_image = self.mask_depth_image(
            output_depth, depth_start, depth_end)
        depth_image = (depth_image*1000).astype("uint16")
        return depth_image
    
    
    def get_intrinsic(self, cam_file, interval_scale=1):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = cam_file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

        return cam

    def process(self, args, func_callback):
        self.common_process(args)
        p = os.path.join(self.INPUT_DIR, self.NAME+"/high_res/*")
        dir_list = glob.glob(p)
        sample_num = 0
        nds_file_list = list()
        for dir_num, d in enumerate(dir_list):
            for sub_d in glob.glob(d+"/*"):
                img_list = glob.glob(sub_d+"/blended_images/*")

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
                    cam_txt_path = ori_image_path.replace(
                        "blended_images", "cams").split(".")[0]+"_cam.txt"
                    cams = self.get_intrinsic(open(cam_txt_path, "r"))
                    depth_start = cams[1, 3, 0] + cams[1, 3, 1]
                    depth_end = cams[1, 3, 3] - cams[1, 3, 1]
                    depth_end = depth_end if depth_end < 50 else 50
                    blendedmvs_K = cams[1, :3, :3]
                    if args.transform:
                        map1_x, map1_y = cv2.initUndistortRectifyMap(cameraMatrix=blendedmvs_K, distCoeffs=None, R=None, newCameraMatrix=nreal_param.nreal_left_K, size=(
                            nreal_param.nreal_col, nreal_param.nreal_row), m1type=cv2.CV_32FC1)
                        task_info = [args, map1_x, map1_y, path_dict,
                                     depth_start, depth_end, sample_num]
                        # nds_data_item = self.blendedmvs_data_2_nreal_core(task_info)
                        pool.apply_async(self.blendedmvs_data_2_nreal_core, (task_info, ),
                                         callback=call_back)
                    else:
                        task_info = [args, path_dict,
                                     depth_start, depth_end, sample_num]
                        # nds_data_item = self.blendedmvs_func_core(task_info)
                        pool.apply_async(self.blendedmvs_func_core, (task_info, ),
                                         callback=call_back)

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


    def blendedmvs_func_core(self, task_info):
        args, path_dict, depth_start, depth_end, image_id = task_info
        depth_image = self.blendedmvs_depth_read(
            path_dict["ori_depths_path"], depth_start, depth_end)
        cv2.imwrite(os.path.join(args.output_path, self.NAME,
                    path_dict["output_depths_path"]), depth_image)

        shutil.copy(path_dict["ori_images_path"], os.path.join(
            args.output_path, self.NAME, path_dict["output_images_path"]))
        
        nds_data_item = format_nds(image_id, depth_image.shape, path_dict)
        return nds_data_item
    

    def blendedmvs_data_2_nreal_core(self, task_info):
        args,  map1_x, map1_y, path_dict, depth_start, depth_end, image_id = task_info
        image = cv2.imread(path_dict["ori_images_path"])
        ori_depth_path = path_dict["ori_depths_path"]
        output_depth = self.blendedmvs_depth_read(ori_depth_path, depth_start, depth_end)
        
        
        data_2_nreal_info = [args, image, output_depth,
                             map1_x, map1_y, path_dict, image_id]
        nds_data_item = self.data_2_nreal_core(data_2_nreal_info)
        return nds_data_item
