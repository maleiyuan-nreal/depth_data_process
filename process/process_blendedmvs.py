"""
blendedmvs数据只用了high_res
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import BlendedMVS
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, get_file_name, save_json_items
)
from process.common_process import common_process, get_path, get_path_by_depth
from utils.merge_nds import mergeFiles


def process(args, func_core, func_callback):
    blendedmvs_obj = BlendedMVS(args.output_path)
    common_process(blendedmvs_obj, args)
    p = os.path.join(blendedmvs_obj.INPUT_DIR, blendedmvs_obj.NAME+"/*")
    dir_list = glob.glob(p)
    sample_num = 0
    nds_file_list = list()
    for dir_num, d in enumerate(dir_list):
        for sub_d in glob.glob(d+"/*"):
            img_list = glob.glob(sub_d+"/blended_images/*")
            
            rel_sub_d = os.path.relpath(sub_d, os.path.join(
                blendedmvs_obj.INPUT_DIR, blendedmvs_obj.NAME))
            sub_nds_file = os.path.join(
                args.output_path, blendedmvs_obj.NAME, blendedmvs_obj.OUTPUT_DIR, rel_sub_d, "annotation.nds")
            logging.info(f"sub_nds_file: {sub_nds_file}")

            pbar = tqdm(total=len(img_list))
            pbar.set_description("Creating {} nds dataset: ".format(rel_sub_d))

            for dirs in blendedmvs_obj.DATA_TYPE_LIST:
                check_and_make_dir(os.path.join(
                    args.output_path, blendedmvs_obj.NAME, blendedmvs_obj.OUTPUT_DIR, rel_sub_d, dirs))

            pool = mp.Pool(args.n_proc)
            nds_data = list()
            call_back = lambda *args: func_callback(args, pbar, nds_data)
            for _, ori_image_path in enumerate(img_list):
                sample_num += 1
                path_dict = get_path(blendedmvs_obj, ori_image_path, rel_sub_d)
                task_info = [path_dict, sample_num, blendedmvs_obj]
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

    mergeFiles(blendedmvs_obj, nds_file_list)
    logging.info("sub_nds_file merged!")