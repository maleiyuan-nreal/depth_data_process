import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import ReDWeb
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, check_and_make_dir,
    save_json_items
)
from process.common_process import common_process, get_path


def process(args, func_core, func_callback):
    rw_obj = ReDWeb(args.output_path)
    
    p = os.path.join(rw_obj.INPUT_DIR, rw_obj.NAME)
    img_list = glob.glob(p+"/"+rw_obj.SUB_INPUT_DIR[0]+"/*.jpg")
    pbar = common_process(rw_obj, args, len(img_list))
    
    pool = mp.Pool(args.n_proc)
    nds_data = list()
    call_back = lambda *args: func_callback(args, pbar, nds_data)
    for image_id, ori_image_path in enumerate(img_list):
        path_dict = get_path(rw_obj, ori_image_path)
        task_info = [path_dict, image_id, rw_obj]
        pool.apply_async(func_core, (task_info, ), callback=call_back)

    pool.close()
    pool.join()
    
    nds_data.sort(key=lambda x:x['image_id'])
    save_json_items(rw_obj.NDS_FILE_NAME, nds_data)
