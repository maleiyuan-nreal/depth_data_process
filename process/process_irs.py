"""
blendedmvs数据只用了high_res
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import IRS
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, get_file_name, save_json_items
)
from process.common_process import common_process, get_path, get_path_by_depth
from utils.merge_nds import mergeFiles
from utils.process_depth import exr2hdr


def process(args, func_core, func_callback):
    irs_obj = IRS(args.output_path)
    common_process(irs_obj, args)
    p = os.path.join(irs_obj.INPUT_DIR, irs_obj.NAME)
    txt_path = os.path.join(p, "list")
   
    nds_file_list = list()
    sample_num = 0 
    file_list = glob.glob(txt_path+"/*")
    for dir_num, txt_list_path in enumerate(file_list):
        if txt_list_path in ["/data/lyma/IRSDataset/list/IRSDataset_TRAIN.list", "/data/lyma/IRSDataset/list/IRSDataset_TEST.list"]:
            continue
        txt_list_name = os.path.basename(txt_list_path).split(".")[0]
        scene_name, sub_d = txt_list_name.split("_")
        rel_sub_d = os.path.join(scene_name, sub_d)
        for dirs in irs_obj.DATA_TYPE_LIST:
            check_and_make_dir(os.path.join(
                args.output_path, irs_obj.NAME, irs_obj.OUTPUT_DIR, rel_sub_d, dirs))
        
        sub_nds_file = os.path.join(
                args.output_path, irs_obj.NAME, irs_obj.OUTPUT_DIR, rel_sub_d, "annotation.nds")
        logging.info(f"sub_nds_file: {sub_nds_file}")
     
        with open(txt_list_path, "r") as f:
            samples = f.readlines()
            pbar = tqdm(total=len(samples))
            pbar.set_description("Creating {} nds dataset: ".format(rel_sub_d))
            call_back = lambda *args: func_callback(args, pbar, nds_data)
            pool = mp.Pool(args.n_proc)
            nds_data = list()
            missing_depth_num = 0
            for line in samples:
                sample_num += 1
                path_dict = dict()
                left_rgb_path, _, depth_path = line.strip().split(" ")
                
                image_path_rel_sub_d = os.path.relpath(left_rgb_path, os.path.join(irs_obj.NAME, scene_name))
                depth_path_rel_sub_d = os.path.relpath(depth_path, os.path.join(irs_obj.NAME, scene_name))
                path_dict["ori_images_path"] = os.path.join(irs_obj.INPUT_DIR, left_rgb_path)
                path_dict["output_images_path"] = os.path.join(irs_obj.OUTPUT_DIR, rel_sub_d, "images", "_".join(image_path_rel_sub_d.split("/")))
                path_dict["ori_depths_path"] = os.path.join(irs_obj.INPUT_DIR, depth_path)
                depth_name_only = "_".join(depth_path_rel_sub_d.split("/"))
                path_dict["output_depths_path"] = os.path.join(irs_obj.OUTPUT_DIR, rel_sub_d, "depths", depth_name_only.split(".")[0]+".png")
                
                # 有的depth的channel是Y, 过滤掉
                hdr = exr2hdr(path_dict["ori_depths_path"])
                if hdr is None:
                    missing_depth_num += 1
                    continue
                task_info = [path_dict, sample_num, irs_obj]
                # nds_data_item = func_core(task_info)
                # call_back(args, pbar, nds_data_item)

                pool.apply_async(func_core, (task_info, ), callback=call_back)

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

    mergeFiles(irs_obj, nds_file_list)
    logging.info("sub_nds_file merged!")