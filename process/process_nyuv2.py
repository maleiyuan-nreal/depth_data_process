import os
import glob
import logging
import multiprocessing as mp
from tqdm import tqdm


from bfuncs import (
    check_and_make_dir, check_and_make_dir,
    save_json_items
)
from process.common_process import Base_Data


class NYUV2(Base_Data):
    def __init__(self, output_path) -> None:
        self.NAME = "nyuv2"
        self.INPUT_DIR = "/home/lyma/FOD/FocusOnDepth/datasets"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["images", "depths", "segmentations"]
        self.DATA_TYPE_LIST = ["images", "depths", "segmentations"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "jpg"
        self.IMAGE_SUFFIX = "jpg"

    def process(self, args, func_callback):

        p = os.path.join(self.INPUT_DIR, self.NAME)
        img_list = glob.glob(p+"/"+self.SUB_INPUT_DIR[0]+"/*.jpg")
        self.common_process(args)
        pbar = tqdm(total=len(img_list))
        pbar.set_description("Creating {} nds dataset: ".format(self.NAME))
        for dirs in self.DATA_TYPE_LIST:
            check_and_make_dir(os.path.join(
                args.output_path, self.NAME, self.OUTPUT_DIR, dirs))

        pool = mp.Pool(args.n_proc)
        nds_data = list()
        call_back = lambda *args: func_callback(args, pbar, nds_data)
        for image_id, ori_image_path in enumerate(img_list):
            path_dict = self.get_path(ori_image_path)
            task_info = [args, path_dict, image_id]
            # print(path_dict)
            # nds_data_item = func_core(task_info)
            # call_back(args, pbar, nds_data_item)
            pool.apply_async(self.func_core, (task_info, ), callback=call_back)

        pool.close()
        pool.join()

        nds_data.sort(key=lambda x: x['image_id'])
        save_json_items(self.NDS_FILE_NAME, nds_data)
