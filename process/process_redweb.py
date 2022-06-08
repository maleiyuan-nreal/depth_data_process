import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import ReDWeb
from process.common_process import common_process, get_path


def process(args, func_core, func_callback):
    rw_obj = ReDWeb(args.output_path)
    
    p = os.path.join(rw_obj.INPUT_DIR, rw_obj.NAME)
    img_list = glob.glob(p+"/"+rw_obj.SUB_INPUT_DIR[0]+"/*.jpg")
    # print(p+"/"+rw_obj.SUB_INPUT_DIR[0]+"/*.jpg", len(img_list))
    pbar = common_process(rw_obj, args, len(img_list))
    
    image_id = 0
    pool = mp.Pool(args.n_proc)
    call_back = lambda *args: func_callback(args, pbar, rw_obj.NDS_FILE_NAME)
    for ori_image_path in img_list:
        image_id += 1
        path_dict = get_path(rw_obj, ori_image_path)
        task_info = [path_dict, image_id]
        pool.apply_async(func_core, (task_info, ), callback=call_back)

    pool.close()
    pool.join()
