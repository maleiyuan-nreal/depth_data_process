import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import INRIA
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, get_file_name
)
from process.common_process import common_process, get_path


def process(args, func_core, func_callback):
    inria_obj = INRIA(args.output_path)
    
    p = os.path.join(inria_obj.INPUT_DIR, inria_obj.NAME)
    img_list = glob.glob(p+"/"+inria_obj.SUB_INPUT_DIR[0]+"/*.jpg")
    pbar = common_process(inria_obj, args, len(img_list))
    
    image_id = 0
    pool = mp.Pool(args.n_proc)
    call_back = lambda *args: func_callback(args, pbar, inria_obj.NDS_FILE_NAME)
    for ori_image_path in img_list:
        image_id += 1
        path_dict = get_path(inria_obj, ori_image_path)
        task_info = [path_dict, image_id, inria_obj]
        pool.apply_async(func_core, (task_info, ), callback=call_back)

    pool.close()
    pool.join()
