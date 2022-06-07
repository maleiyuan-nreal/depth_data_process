import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from config.data_config import NYUV2, IMAGE_OUTPUT_DIR
from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, check_and_make_dir,
    get_file_name
)


nyuv2_obj = NYUV2()


def process(n_proc, func_core, func_callback):

    dataset_name = nyuv2_obj.NAME
    check_and_make_dir(os.path.join(IMAGE_OUTPUT_DIR, dataset_name))

    image_id = 0
    pool = mp.Pool(n_proc)

    p = os.path.join(nyuv2_obj.INPUT_DIR, dataset_name)
    img_list = glob.glob(p+"/images/*.jpg")

    pbar = tqdm(total=len(img_list))
    pbar.set_description("Creating {} nds dataset: ".format(dataset_name))

    logging.info(dataset_name+" is processing")
    nds_path = nyuv2_obj.NDS_FILE_NAME
    if os.path.exists(nds_path):
        os.remove(nds_path)
    call_back = lambda *args: func_callback(args, pbar, nds_path)
    logging.info(f"nds_path: {nds_path}")
    for ori_image_path in img_list:
        image_id += 1
        task_info = [ori_image_path, image_id, nyuv2_obj]
        pool.apply_async(func_core, (task_info, ), callback=call_back)

    pool.close()
    pool.join()
