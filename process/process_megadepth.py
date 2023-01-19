"""
megadepth有俩种不同的depth, dense depth map & ordinal depth; 其中ordinal depth中包含-1,
这里只处理了dense depth map
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm
import numpy as np


from bfuncs import (
    check_and_make_dir, save_json_items
)
from process.common_process import Base_Data, get_path_by_depth


class MegaDepth(Base_Data):
    def __init__(self, output_path) -> None:
        super().__init__()
        self.NAME = "MegaDepth_v1"
        self.INPUT_DIR = "/data/depth/datasets/public/mono_extracted/MegaDepth"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["imgs", "depths"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "h5"
        self.IMAGE_SUFFIX = "jpg"

    def process(self, args, func_core, func_callback):
        self.common_process(args)

        orientation_list = ["landscape", "portrait"]
        whole_img_list = list()
        whole_depth_list = list()
        for orientation in orientation_list:
            logging.info("{} loaded!".format(orientation))
            dir_load_all_img = self.INPUT_DIR + \
                "/train_val_list/" + orientation + "/imgs_MD.p"
            dir_load_all_target = self.INPUT_DIR + \
                "/train_val_list/" + orientation + "/targets_MD.p"

            img_list = np.load(dir_load_all_img, allow_pickle=True)
            target_list = np.load(dir_load_all_target, allow_pickle=True)
            whole_img_list.extend(img_list)
            whole_depth_list.extend(target_list)

        pbar = tqdm(total=len(whole_depth_list))
        pbar.set_description(
            "Creating {} nds dataset: ".format(self.NAME))

        sample_num = 0
        missing_depth_num = 0

        pool = mp.Pool(args.n_proc)
        nds_data = list()
        call_back = lambda *args: func_callback(args, pbar, nds_data)
        for img_path, target_path in zip(whole_img_list, whole_depth_list):
            rel_sub_d = os.path.dirname(os.path.dirname(target_path))
            for dirs in self.DATA_TYPE_LIST:
                check_and_make_dir(os.path.join(
                    args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d, dirs))
            ori_depth_path = os.path.join(
                self.INPUT_DIR, self.NAME, target_path)
            path_dict = get_path_by_depth(self, ori_depth_path)
            if not path_dict:
                sample_num += 1
                missing_depth_num += 1
                continue
            task_info = [path_dict, sample_num, self]
            sample_num += 1
            # nds_data_item = func_core(task_info)
            # call_back(nds_data_item, pbar, nds_data)
            # print(nds_data)
            pool.apply_async(func_core, (task_info, ), callback=call_back)
        pool.close()
        pool.join()

        if missing_depth_num > 0:
            assert missing_depth_num + len(nds_data) == len(whole_depth_list)
            logging.info("processed done! Total map {}, Missing {} dense map".format(
                len(whole_depth_list), missing_depth_num))
        nds_data.sort(key=lambda x: x['image_id'])
        save_json_items(self.NDS_FILE_NAME, nds_data)
