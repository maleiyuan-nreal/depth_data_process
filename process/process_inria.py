import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import INRIA
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, get_file_name, save_json_items
)
from process.common_process import common_process, get_path


def process(args, func_core, func_callback):
    inria_obj = INRIA(args.output_path)
    
    p = os.path.join(inria_obj.INPUT_DIR, inria_obj.NAME)
    img_list = glob.glob(p+"/"+inria_obj.SUB_INPUT_DIR[0]+"/*.jpg")
    common_process(inria_obj, args)
    pbar = tqdm(total=len(img_list))
    pbar.set_description("Creating {} nds dataset: ".format(inria_obj.NAME))
    for dirs in inria_obj.DATA_TYPE_LIST:
        check_and_make_dir(os.path.join(args.output_path, inria_obj.NAME, inria_obj.OUTPUT_DIR, dirs))
        
    
    pool = mp.Pool(args.n_proc)
    nds_data = list()
    call_back = lambda *args: func_callback(args, pbar, nds_data)
    for image_id, ori_image_path in enumerate(img_list):
        path_dict = get_path(inria_obj, ori_image_path)
        task_info = [path_dict, image_id, inria_obj]
        pool.apply_async(func_core, (task_info, ), callback=call_back)

    pool.close()
    pool.join()

    nds_data.sort(key=lambda x:x['image_id'])
    save_json_items(inria_obj.NDS_FILE_NAME, nds_data)