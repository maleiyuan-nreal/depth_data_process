"""
train+val 俩个nds file

"""

import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, check_and_make_dir,
    save_json_items
)
from process.common_process import Base_Data, get_path
from utils.merge_nds import mergeFiles


class HRWSI(Base_Data):
    def __init__(self, output_path) -> None:
        self.NAME = "HR-WSI"
        self.INPUT_DIR = "/data/lyma/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["imgs", "gts"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "png"
        self.IMAGE_SUFFIX = "jpg"

    def process(self, args, func_core, func_callback):
        self.common_process(args)
        p = os.path.join(self.INPUT_DIR, self.NAME)
        sample_num = 0
        nds_file_list = list()
        for split_type in ["train", "val"]:
            img_list = glob.glob(p+"/"+split_type+"/" +
                                 self.SUB_INPUT_DIR[0]+"/*.jpg")
            pbar = tqdm(total=len(img_list))
            pbar.set_description(
                "Creating {} {} nds dataset: ".format(self.NAME, split_type))

            sub_nds_file = os.path.join(
                args.output_path, self.NAME, self.OUTPUT_DIR, split_type, "annotation.nds")

            for dirs in self.DATA_TYPE_LIST:
                check_and_make_dir(os.path.join(
                    args.output_path, self.NAME, self.OUTPUT_DIR, split_type, dirs))

            pool = mp.Pool(args.n_proc)
            nds_data = list()
            call_back = lambda *args: func_callback(args, pbar, nds_data)
            for _, ori_image_path in enumerate(img_list):
                path_dict = get_path(self, ori_image_path, split_type)
                task_info = [path_dict, sample_num, self]
                sample_num += 1
                # nds_data_item = func_core(task_info)
                # call_back(args, pbar, nds_data_item)
                pool.apply_async(func_core, (task_info, ), callback=call_back)

            pool.close()
            pool.join()

            nds_data.sort(key=lambda x: x['image_id'])
            save_json_items(sub_nds_file, nds_data)
            nds_file_list.append(sub_nds_file)

        self.mergeFiles(nds_file_list)
        logging.info("sub_nds_file merged!")
