"""
blendedmvs数据只用了high_res
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, get_file_name, save_json_items
)
from process.common_process import Base_Data, get_path, get_path_by_depth
from utils.merge_nds import mergeFiles
from utils.process_depth import exr2hdr


class IRS(Base_Data):
    def __init__(self, output_path) -> None:
        self.NAME = "IRSDataset"
        self.INPUT_DIR = "/data/lyma/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["", ""]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "exr"
        self.IMAGE_SUFFIX = "png"

    def process(self, args, func_core, func_callback):
        self.common_process(args)
        p = os.path.join(self.INPUT_DIR, self.NAME)
        txt_path = os.path.join(p, "list")

        nds_file_list = list()
        sample_num = 0
        file_list = glob.glob(txt_path+"/*")
        for dir_num, txt_list_path in enumerate(file_list):
            txt_list_name = os.path.basename(txt_list_path).split(".")[0]
            scene_name, sub_d = txt_list_name.split("_")
            rel_sub_d = os.path.join(scene_name, sub_d)
            for dirs in self.DATA_TYPE_LIST:
                check_and_make_dir(os.path.join(
                    args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d, dirs))

            sub_nds_file = os.path.join(
                args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d, "annotation.nds")
            logging.info(f"sub_nds_file: {sub_nds_file}")

            with open(txt_list_path, "r") as f:
                samples = f.readlines()
                pbar = tqdm(total=len(samples))
                pbar.set_description(
                    "Creating {} nds dataset: ".format(rel_sub_d))
                call_back = lambda *args: func_callback(args, pbar, nds_data)
                pool = mp.Pool(args.n_proc)
                nds_data = list()
                missing_depth_num = 0
                for line in samples:
                    path_dict = dict()
                    left_rgb_path, _, depth_path = line.strip().split(" ")

                    image_path_rel_sub_d = os.path.relpath(
                        left_rgb_path, os.path.join(self.NAME, scene_name))
                    depth_path_rel_sub_d = os.path.relpath(
                        depth_path, os.path.join(self.NAME, scene_name))
                    path_dict["ori_images_path"] = os.path.join(
                        self.INPUT_DIR, left_rgb_path)
                    path_dict["output_images_path"] = os.path.join(
                        self.OUTPUT_DIR, rel_sub_d, "images", "_".join(image_path_rel_sub_d.split("/")))
                    path_dict["ori_depths_path"] = os.path.join(
                        self.INPUT_DIR, depth_path)
                    depth_name_only = "_".join(depth_path_rel_sub_d.split("/"))
                    path_dict["output_depths_path"] = os.path.join(
                        self.OUTPUT_DIR, rel_sub_d, "depths", depth_name_only.split(".")[0]+".png")

                    # 有的depth的channel是Y, 过滤掉
                    hdr = exr2hdr(path_dict["ori_depths_path"])
                    if hdr is None:
                        sample_num += 1
                        missing_depth_num += 1
                        continue
                    task_info = [path_dict, sample_num, self]
                    sample_num += 1
                    # nds_data_item = func_core(task_info)
                    # call_back(args, pbar, nds_data_item)

                    pool.apply_async(func_core, (task_info, ),
                                     callback=call_back)

                pool.close()
                pool.join()

                nds_data.sort(key=lambda x: x['image_id'])
                save_json_items(sub_nds_file, nds_data)
                nds_file_list.append(sub_nds_file)
                if missing_depth_num > 0:
                    assert missing_depth_num + len(nds_data) == len(samples)
                    logging.info("{} processed done! Total map {}, Missing {} dense map".format(
                        sub_d, len(samples), missing_depth_num))
            logging.info(
                "Total dirs {}, currently {}/{}".format(len(file_list), dir_num+1, len(file_list)))

        self.mergeFiles(nds_file_list)
        logging.info("sub_nds_file merged!")
