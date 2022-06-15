"""
megadepth有俩种不同的depth, dense depth map & ordinal depth; 其中ordinal depth中包含-1,
这里只处理了dense depth map
"""
import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import MegaDepth
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, get_file_name, save_json_items
)
from process.common_process import common_process, get_path, get_path_by_depth
from utils.merge_nds import mergeFiles


def process(args, func_core, func_callback):
    megadepth_obj = MegaDepth(args.output_path)
    common_process(megadepth_obj, args)
    p = os.path.join(megadepth_obj.INPUT_DIR, megadepth_obj.NAME+"/*")
    sample_num = 0

    nds_file_list = list()
    for dir_num, d in enumerate(glob.glob(p)):
        for sub_d in glob.glob(d+"/*"):
            depth_list = glob.glob(sub_d+"/depths/*.h5")
            rel_sub_d = os.path.relpath(sub_d, os.path.join(
                megadepth_obj.INPUT_DIR, megadepth_obj.NAME))
            sub_nds_file = os.path.join(
                args.output_path, megadepth_obj.NAME, megadepth_obj.OUTPUT_DIR, rel_sub_d, "annotation.nds")
            if os.path.exists(sub_nds_file):
                os.remove(sub_nds_file)
            logging.info(f"sub_nds_file: {sub_nds_file}")

            pbar = tqdm(total=len(depth_list))
            pbar.set_description("Creating {} nds dataset: ".format(rel_sub_d))

            for dirs in megadepth_obj.DATA_TYPE_LIST:
                check_and_make_dir(os.path.join(
                    args.output_path, megadepth_obj.NAME, megadepth_obj.OUTPUT_DIR, rel_sub_d, dirs))

            pool = mp.Pool(args.n_proc)
            nds_data = list()
            call_back = lambda *args: func_callback(args, pbar, nds_data)
            missing_depth_num = 0
            for _, ori_depth_path in enumerate(depth_list):
                path_dict = get_path_by_depth(megadepth_obj, ori_depth_path)
                if not path_dict:
                    missing_depth_num += 1
                    continue
                sample_num += 1
                task_info = [path_dict, sample_num, megadepth_obj]
                # nds_data_item = func_core(task_info)
                # call_back(args, pbar, nds_data_item)
                pool.apply_async(func_core, (task_info, ), callback=call_back)

            pool.close()
            pool.join()

            nds_data.sort(key=lambda x: x['image_id'])
            save_json_items(sub_nds_file, nds_data)
            nds_file_list.append(sub_nds_file)

            if missing_depth_num > 0:
                assert missing_depth_num + len(nds_data) == len(depth_list)
                logging.info("{} processed done! Total map {}, Missing {} dense map".format(
                    sub_d, len(depth_list), missing_depth_num))

        logging.info(
            "Total dirs {}, currently {}/{}".format(len(glob.glob(p)), dir_num+1, len(glob.glob(p))))

    mergeFiles(megadepth_obj, nds_file_list)
