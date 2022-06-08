import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import NYUV2
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, check_and_make_dir,
    get_file_name
)
from process.common_process import common_process, get_path


def process(args, func_core, func_callback):
    nyuv2_obj = NYUV2(args.output_path)
    
    p = os.path.join(nyuv2_obj.INPUT_DIR, nyuv2_obj.NAME)
    img_list = glob.glob(p+"/"+nyuv2_obj.SUB_INPUT_DIR[0]+"/*.jpg")
    pbar = common_process(nyuv2_obj, args, len(img_list))
    
    image_id = 0
    pool = mp.Pool(args.n_proc)

    call_back = lambda *args: func_callback(args, pbar, nyuv2_obj.NDS_FILE_NAME)
    for ori_image_path in img_list:
        image_id += 1
        path_dict = get_path(nyuv2_obj, ori_image_path)
        task_info = [path_dict, image_id, nyuv2_obj]
        pool.apply_async(func_core, (task_info, ), callback=call_back)

    pool.close()
    pool.join()
