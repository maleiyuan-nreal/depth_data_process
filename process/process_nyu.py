import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np


from bfuncs import (
    check_and_make_dir, save_json_items
)
from process.common_process import Base_Data, get_path_by_depth


class NYU(Base_Data):
    def __init__(self, output_path) -> None:
        super().__init__()
        self.NAME = "nyu_v2"
        self.INPUT_DIR = "/data/depth/datasets/public/mono_extracted/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["images", "depths"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "png"
        self.IMAGE_SUFFIX = "png"

    def process(self, args, func_callback):
        self.common_process(args)
        
        nds_file_list = list()
        sample_num = 0
        
        K = {"fx": 518.85790117450188,
            "fy": 519.46961112127485,
            "cx": 325.58244941119034,
            "cy": 253.73616633400465}
        for train_val in ["training", "testing"]:
            train_val_dir = os.path.join(self.INPUT_DIR, self.NAME, train_val)
           
            depth_dir = os.path.join(self.INPUT_DIR, self.NAME, train_val, "depths")

            if train_val == "training":
                scene_list = glob.glob(train_val_dir+"/images/*")
                
                for scene_dir in scene_list:
                    scene_name = scene_dir.split("/")[-1]
                    
                    pool = mp.Pool(args.n_proc)
                    nds_data = list()
                    
                    image_list = glob.glob(scene_dir+"/*")
                    pbar = tqdm(total=len(image_list))
                    pbar.set_description(
                        "Creating {} {} {} nds dataset: ".format(self.NAME, train_val, scene_name))
                    call_back = lambda *args: func_callback(args, pbar, nds_data)
                    sub_nds_file = os.path.join(
                        args.output_path, self.NAME, "data", train_val, "images", scene_name, "annotation.nds")
                    logging.info(f"sub_nds_file: {sub_nds_file}")
                
                    rel_sub_d = os.path.relpath(scene_dir, os.path.join(
                            self.INPUT_DIR, self.NAME))
                    check_and_make_dir(os.path.join(
                        args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d))
                    check_and_make_dir(os.path.join(
                        args.output_path, self.NAME, self.OUTPUT_DIR, train_val, "depths", scene_name))
                
        
                    for image_path in image_list:
                        image_name = image_path.split("/")[-1]
                        info_dict = dict()
                        info_dict["ori_images_path"] = image_path
                        depth_path = os.path.join(depth_dir, scene_name, image_name)
                        info_dict["ori_depths_path"] = depth_path
                        info_dict["output_images_path"] = os.path.join(self.OUTPUT_DIR, rel_sub_d, image_name)
                        info_dict["output_depths_path"] = os.path.join(self.OUTPUT_DIR, train_val, "depths", scene_name, image_name)
                        info_dict.update(K)

                        task_info = [args, info_dict, sample_num]
                        sample_num += 1
                        # nds_data_item = self.func_core(task_info)
                        # call_back(nds_data_item, pbar, nds_data)
                        pool.apply_async(self.func_core, (task_info, ), callback=call_back)
                
                
                    pool.close()
                    pool.join()

            elif train_val == "testing":
                nds_data = list()
                image_dir = os.path.join(train_val_dir, "images")
                image_list = glob.glob(image_dir+"/*")
                
                pool = mp.Pool(args.n_proc)
                pbar = tqdm(total=len(image_list))
                pbar.set_description(
                    "Creating {} {} nds dataset: ".format(self.NAME, train_val))
                call_back = lambda *args: func_callback(args, pbar, nds_data)
                sub_nds_file = os.path.join(
                        args.output_path, self.NAME, "data", train_val, "images", "annotation.nds")
                logging.info(f"sub_nds_file: {sub_nds_file}")
                for image_path in image_list:
                    rel_sub_d = os.path.relpath(image_dir, os.path.join(
                            self.INPUT_DIR, self.NAME))
                    check_and_make_dir(os.path.join(
                        args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d))
                    check_and_make_dir(os.path.join(
                        args.output_path, self.NAME, self.OUTPUT_DIR, train_val, "depths"))
                    image_name = image_path.split("/")[-1]
                    info_dict = dict()
                    info_dict["ori_images_path"] = image_path
                    depth_path = os.path.join(depth_dir, image_name)
                    info_dict["ori_depths_path"] = depth_path
                    info_dict["output_images_path"] = os.path.join(self.OUTPUT_DIR, rel_sub_d, image_name)
                    info_dict["output_depths_path"] = os.path.join(self.OUTPUT_DIR, train_val, "depths", image_name)
                    info_dict.update(K)
                    task_info = [args, info_dict, sample_num]
                    sample_num += 1
                    # nds_data_item = self.func_core(task_info)
                    # call_back(nds_data_item, pbar, nds_data)
                    pool.apply_async(self.func_core, (task_info, ), callback=call_back)
                
                
                pool.close()
                pool.join()

        
            nds_data.sort(key=lambda x: x['image_id'])
            save_json_items(sub_nds_file, nds_data)
            nds_file_list.append(sub_nds_file)
                

        self.mergeFiles(nds_file_list)
        logging.info("sub_nds_file merged!")