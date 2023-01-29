"""
apollospace用的是stereo
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm
import shutil

from bfuncs import (
    check_and_make_dir, save_json_items, get_file_name
)
from process.common_process import Base_Data
from utils.format_nds import format_nds

class ApolloScapeScene(Base_Data):
    def __init__(self, output_path) -> None:
        self.NAME = "apolloscape_scene_parsing"
        self.INPUT_DIR = "/data/depth/datasets/public/mono_extracted"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["ColorImage", "Depth"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.IMAGE_SUFFIX = "jpg"
        self.DPETH_SUFFIX = "png"

    def process(self, args, func_callback):
        self.common_process(args)
        p = os.path.join(self.INPUT_DIR, self.NAME+"/*")
        dir_list = glob.glob(p)
        nds_file_list = list()
        
        sample_num = 0
    
        for dir_num, sub_dir in enumerate(dir_list):
            depth_dir = os.path.join(sub_dir, "Depth")
            image_dir = os.path.join(sub_dir, "ColorImage")
            
            rel_sub_d = os.path.relpath(sub_dir, os.path.join(
                self.INPUT_DIR, self.NAME))
            sub_nds_file = os.path.join(
                args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d, "annotation.nds")
            logging.info(f"sub_nds_file: {sub_nds_file}")
            for dirs in self.DATA_TYPE_LIST:
                check_and_make_dir(os.path.join(
                    args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d, dirs))
            
            nds_data = list()
            for image_list_dir in glob.glob(image_dir+"/*"):
                record_name = image_list_dir.split("/")[-1]
                img_list = glob.glob(image_list_dir + "/Camera 5/*") + glob.glob(image_list_dir + "/Camera 6/*")

                pbar = tqdm(total=len(img_list))
                pbar.set_description("Creating {} {} nds dataset: ".format(rel_sub_d, record_name))
                
                pool = mp.Pool(args.n_proc)
                call_back = lambda *args: func_callback(args, pbar, nds_data)
                
                for _, ori_image_path in enumerate(img_list):   
                    
                    camera_name = ori_image_path.split("/")[-2]
                    if camera_name == "Camera 5":
                        K = {"fx": 2304.54786556982,
                            "fy": 2305.875668062,
                            "cx": 1686.23787612802,
                            "cy": 1354.98486439791}
                    if camera_name == "Camera 6":
                        K = {"fx": 2300.39065314361,
                            "fy": 2301.31478860597,
                            "cx": 1713.21615190657,
                            "cy": 1342.91100799715}

                    image_name_only = get_file_name(ori_image_path)
                    depth_path = os.path.join(depth_dir, record_name, camera_name, image_name_only.split(".")[0]+".png")
                    if not os.path.exists(depth_path):
                        continue
                    
                    info_dict = dict()
                    info_dict["ori_images_path"] = ori_image_path
                    check_and_make_dir(os.path.join(
                            args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d, "images", record_name, camera_name))
                    info_dict["output_images_path"] = os.path.join(self.OUTPUT_DIR, rel_sub_d, "images", record_name, camera_name, image_name_only.split(".")[0]+".png")
                    info_dict["ori_depths_path"] = depth_path
                    check_and_make_dir(os.path.join(
                            args.output_path, self.NAME, self.OUTPUT_DIR, rel_sub_d, "depths", record_name, camera_name))
                    info_dict["output_depths_path"] = os.path.join(self.OUTPUT_DIR, rel_sub_d, "depths", record_name, camera_name, image_name_only.split(".")[0]+".png")
                    info_dict.update(K)
                    task_info = [args, info_dict, sample_num]
                    sample_num += 1
                    # nds_data_item = self.func_core(task_info)
                    # call_back(args, pbar, nds_data)
                    pool.apply_async(self.func_core, (task_info, ), callback=call_back)

                pool.close()
                pool.join()

            nds_data.sort(key=lambda x: x['image_id'])
            save_json_items(sub_nds_file, nds_data)
            nds_file_list.append(sub_nds_file)
            logging.info(
                "Total dirs {}, currently {}/{}".format(len(dir_list), dir_num+1, len(dir_list)))

        self.mergeFiles(nds_file_list)
        logging.info("sub_nds_file merged!")