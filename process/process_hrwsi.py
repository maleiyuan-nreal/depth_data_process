"""
train+val 俩个nds file

"""

import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import HRWSI
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, check_and_make_dir,
    save_json_items
)
from process.common_process import common_process, get_path
from utils.merge_nds import mergeFiles


def process(args, func_core, func_callback):
    hrwsi_obj = HRWSI(args.output_path)
    
    p = os.path.join(hrwsi_obj.INPUT_DIR, hrwsi_obj.NAME)
    sample_num = 0
    nds_file_list = list()
    for split_type in ["train", "val"]:
        img_list = glob.glob(p+"/"+split_type+"/"+hrwsi_obj.SUB_INPUT_DIR[0]+"/*.jpg")
        pbar = common_process(hrwsi_obj, args)
        pbar = tqdm(total=len(img_list))
        pbar.set_description("Creating {} {} nds dataset: ".format(hrwsi_obj.NAME, split_type))
        
        sub_nds_file = os.path.join(
                args.output_path, hrwsi_obj.NAME, hrwsi_obj.OUTPUT_DIR, split_type, "annotation.nds")
        if os.path.exists(sub_nds_file):
            os.remove(sub_nds_file)
        for dirs in hrwsi_obj.DATA_TYPE_LIST:
            check_and_make_dir(os.path.join(args.output_path, hrwsi_obj.NAME, hrwsi_obj.OUTPUT_DIR, split_type, dirs))
            
        pool = mp.Pool(args.n_proc)
        nds_data = list()
        call_back = lambda *args: func_callback(args, pbar, nds_data)
        for _, ori_image_path in enumerate(img_list):
            sample_num += 1
            path_dict = get_path(hrwsi_obj, ori_image_path, split_type)
            task_info = [path_dict, sample_num, hrwsi_obj]
            # nds_data_item = func_core(task_info)
            # call_back(args, pbar, nds_data_item)
            pool.apply_async(func_core, (task_info, ), callback=call_back)

        pool.close()
        pool.join()
        
        nds_data.sort(key=lambda x:x['image_id'])
        save_json_items(sub_nds_file, nds_data)
        nds_file_list.append(sub_nds_file)
    
    mergeFiles(hrwsi_obj, nds_file_list)
    logging.info("sub_nds_file merged!")