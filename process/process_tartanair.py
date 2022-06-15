"""
tartanair数据只用了左目+Easy
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import TartanAir
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, get_file_name, save_json_items
)
from process.common_process import common_process, get_path, get_path_by_depth
from utils.merge_nds import mergeFiles


def process(args, func_core, func_callback):
    tartanair_obj = TartanAir(args.output_path)
    common_process(tartanair_obj, args)
    p = os.path.join(tartanair_obj.INPUT_DIR, tartanair_obj.NAME+"/*/Easy")
    dir_list = glob.glob(p)
    nds_file_list = list()
    for dir_num, d in enumerate(dir_list):
        for sub_d in glob.glob(d+"/P*"):
            img_list = glob.glob(sub_d+"/image_left/*")

            rel_sub_d = os.path.relpath(sub_d, os.path.join(
                tartanair_obj.INPUT_DIR, tartanair_obj.NAME))
            sub_nds_file = os.path.join(
                args.output_path, tartanair_obj.NAME, tartanair_obj.OUTPUT_DIR, rel_sub_d, "annotation.nds")
            logging.info(f"sub_nds_file: {sub_nds_file}")

            pbar = tqdm(total=len(img_list))
            pbar.set_description("Creating {} nds dataset: ".format(rel_sub_d))

            for dirs in tartanair_obj.DATA_TYPE_LIST:
                check_and_make_dir(os.path.join(
                    args.output_path, tartanair_obj.NAME, tartanair_obj.OUTPUT_DIR, rel_sub_d, dirs))

            pool = mp.Pool(args.n_proc)
            nds_data = list()
            call_back = lambda *args: func_callback(args, pbar, nds_data)
            for image_id, ori_image_path in enumerate(img_list):
                path_dict = get_path(tartanair_obj, ori_image_path, rel_sub_d)
                task_info = [path_dict, image_id, tartanair_obj]
                # print(path_dict)
                # nds_data_item = func_core(task_info)
                # call_back(args, pbar, nds_data_item)
                pool.apply_async(func_core, (task_info, ), callback=call_back)

            pool.close()
            pool.join()

            nds_data.sort(key=lambda x: x['image_id'])
            save_json_items(sub_nds_file, nds_data)
            nds_file_list.append(sub_nds_file)
        assert len(img_list) == len(nds_data)
        logging.info(
            "Total dirs {}, currently {}/{}".format(len(dir_list), dir_num+1, len(dir_list)))

    mergeFiles(tartanair_obj, nds_file_list)
