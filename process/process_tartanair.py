"""
tartanair数据只用了左目+Easy
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from bfuncs import (
    check_and_make_dir, save_json_items
)
from process.common_process import Base_Data, get_path, get_path_by_depth
from utils.merge_nds import mergeFiles


class TartanAir(Base_Data):
    def __init__(self, output_path) -> None:
        self.NAME = "tartanair"
        self.INPUT_DIR = "/data/lyma/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["image_left", "depth_left"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "npy"
        self.IMAGE_SUFFIX = "png"

    def process(self, args, func_core, func_callback):
        self.common_process(args)
        p = os.path.join(self.INPUT_DIR, self.NAME+"/*/Easy")
        dir_list = glob.glob(p)
        nds_file_list = list()
        sample_num = 0
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
                    path_dict = get_path(self, ori_image_path, rel_sub_d)
                    task_info = [path_dict, sample_num, self]
                    sample_num += 1
                    # print(path_dict)
                    # nds_data_item = func_core(task_info)
                    # call_back(args, pbar, nds_data_item)
                    pool.apply_async(func_core, (task_info, ),
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
