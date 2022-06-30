"""
blendedmvs数据只用了high_res
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


import numpy as np


from bfuncs import (
    check_and_make_dir, save_json_items,
)
from process.common_process import Base_Data


class BlendedMVS(Base_Data):
    def __init__(self, output_path) -> None:
        super().__init__()
        self.NAME = "BlendedMVS"
        self.INPUT_DIR = "/home/lyma/SHARE_DATA/datadepth"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data_2_nreal"
        self.SUB_INPUT_DIR = ["blended_images", "rendered_depth_maps"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.IMAGE_SUFFIX = "jpg"
        self.DPETH_SUFFIX = "pfm"

    def get_intrinsic(self, ori_image_path):
        cam_txt_path = ori_image_path.replace("blended_images","cams").split(".")[0]+"_cam.txt"
        intrinsic = list()
        with open(cam_txt_path, "r") as f:
            info = f.readlines()
        for idx, item in enumerate(info[6:10]):
            item = item.strip()
            if idx == 0:
                assert item == "intrinsic"
            else:
                intrinsic.append([float(i) for i in item.split(" ")])
        intrinsic = np.array(intrinsic)
        assert intrinsic.shape == (3,3)
        return intrinsic
        
    
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
                    intrinsic = self.get_intrinsic(ori_image_path)
                    task_info = [args, intrinsic, path_dict, sample_num]
                    sample_num += 1
                    # self.data_2_nreal_core(args, intrinsic, path_dict, sample_num)
                    # nds_data_item = func_core(task_info)
                    # call_back(args, pbar, nds_data_item)
                    pool.apply_async(self.data_2_nreal_core, (task_info, ),
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
