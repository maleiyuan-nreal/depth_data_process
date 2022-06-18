import os
from tqdm import tqdm
import logging
import h5py


import numpy as np


from bfuncs import (
    check_and_make_dir, get_file_name, save_json_items, load_json_items
)


class Base_Data(object):
    def __init__(self):
        self.NAME = ""
        self.NDS_FILE_NAME = ""

    def common_process(self, args):
        """
        func1: check dir exists
        func2: set pbar
        func3: delete nds_file is exists
        """
        dataset_name = self.NAME
        check_and_make_dir(os.path.join(args.output_path, dataset_name))
        check_and_make_dir(os.path.join(
            args.output_path, dataset_name, self.OUTPUT_DIR))

        logging.info(self.NAME+" is processing")

        nds_path = self.NDS_FILE_NAME
        if os.path.exists(nds_path):
            os.remove(nds_path)
        logging.info(f"nds_path: {self.NDS_FILE_NAME}")

    def mergeFiles(self, fileList: list):

        res_items = []

        for filePath in fileList:
            res_items.extend(load_json_items(filePath))

        save_json_items(self.NDS_FILE_NAME, res_items)


def get_path(obj, ori_image_path, sub_folder=None):
    """
    inria、nyuv2、posetrack 需要将depth的jpg图转换为png图
    HR-WSI 有train、val俩个sub_folder
    
    OUTPUT:
        输出的路径为保存相对路径
        ori_data_type
        output_data_type
    
    """
    path_dict = dict()
    path_dict["ori_images_path"] = ori_image_path
    image_name_only = get_file_name(ori_image_path)
    if sub_folder:
        output_root = os.path.join(obj.OUTPUT_DIR, sub_folder)
        input_root = os.path.join(obj.INPUT_DIR, obj.NAME, sub_folder)
    else:
        output_root = obj.OUTPUT_DIR
        input_root = os.path.join(obj.INPUT_DIR, obj.NAME)

    for input_dir, data_type in zip(obj.SUB_INPUT_DIR, obj.DATA_TYPE_LIST):
        if data_type == "images":
            path_dict["output_"+data_type +
                      "_path"] = os.path.join(output_root, data_type, image_name_only)
        elif data_type == "depths":
            if obj.DPETH_SUFFIX == "jpg":
                path_dict["ori_"+data_type +
                          "_path"] = os.path.join(input_root, input_dir, image_name_only)
            elif obj.DPETH_SUFFIX == "png":
                path_dict["ori_"+data_type+"_path"] = os.path.join(
                    input_root, input_dir, image_name_only.split(".")[0]+".png")
            elif obj.DPETH_SUFFIX == "npy":
                # tartanair
                path_dict["ori_"+data_type+"_path"] = os.path.join(
                    input_root, input_dir, image_name_only.split(".")[0]+"_depth.npy")
            elif obj.DPETH_SUFFIX == "pfm":
                # blendedmvs
                path_dict["ori_"+data_type+"_path"] = os.path.join(
                    input_root, input_dir, image_name_only.split(".")[0]+".pfm")
            else:
                raise NotImplementedError
            path_dict["output_"+data_type+"_path"] = os.path.join(
                output_root, data_type, image_name_only.split(".")[0]+".png")
        elif data_type == "segmentations":
            path_dict["ori_"+data_type+"_path"] = os.path.join(
                input_root, input_dir, image_name_only.split(".")[0]+".png")
            path_dict["output_"+data_type+"_path"] = os.path.join(
                output_root, data_type, image_name_only.split(".")[0]+".png")
        else:
            raise Exception("Unknown data type: " + data_type)

    return path_dict


def get_path_by_depth(obj, ori_depth_path):
    """
    megadepth 需要根据depth进行过滤
    
    
    OUTPUT:
        输出的路径为保存相对路径
    
    """
    path_dict = dict()

    hdf5_file_read = h5py.File(ori_depth_path, 'r')
    depth = hdf5_file_read.get('/depth')
    depth = np.array(depth)
    hdf5_file_read.close()

    if depth.min() < 0:
        return path_dict
    root_dir = os.path.join(obj.INPUT_DIR, obj.NAME)
    path_dict["ori_depths_path"] = ori_depth_path
    image_name_only = get_file_name(ori_depth_path)
    sub_d = os.path.relpath(ori_depth_path, root_dir)
    sub_root = os.path.dirname(os.path.dirname(sub_d))
    for input_dir, data_type in zip(obj.SUB_INPUT_DIR, obj.DATA_TYPE_LIST):
        if data_type == "images":
            suffix = ""
            img_path_1 = os.path.join(
                root_dir,  sub_root, "imgs", image_name_only.split(".")[0] + ".jpg")
            img_path_2 = os.path.join(
                root_dir,  sub_root, "imgs", image_name_only.split(".")[0] + ".JPG")
            img_path_3 = os.path.join(
                root_dir,  sub_root, "imgs", image_name_only.split(".")[0] + ".png")
            if os.path.exists(img_path_1):
                path_dict["ori_"+data_type+"_path"] = img_path_1
                suffix = ".jpg"
            elif os.path.exists(img_path_2):
                path_dict["ori_"+data_type+"_path"] = img_path_2
                suffix = ".JPG"
            elif os.path.exists(img_path_3):
                path_dict["ori_"+data_type+"_path"] = img_path_3
                suffix = ".png"
            else:
                print(ori_depth_path, image_name_only)
                print(img_path_1, os.path.exists(img_path_1))
                print(img_path_2, os.path.exists(img_path_2))
                print(img_path_3, os.path.exists(img_path_3))
                raise Exception("Unknown data suffix!")
            path_dict["output_"+data_type+"_path"] = os.path.join(
                obj.OUTPUT_DIR, sub_root, data_type, image_name_only.split(".")[0] + suffix)
        elif data_type == "depths":
            path_dict["output_"+data_type+"_path"] = os.path.join(
                obj.OUTPUT_DIR, sub_root, data_type, image_name_only.split(".")[0]+".png")
        else:
            raise Exception("Unknown data type: " + data_type)

    return path_dict


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname("__file__"))
    from config.data_config import MegaDepth
    megadepth_obj = MegaDepth(".")
    path_dict = get_path_by_depth(
        megadepth_obj, "/home/lyma/SHARE_DATA/datadepth/MegaDepth/MegaDepth_v1/5017/dense0/depths/P1010480.h5")
    print(path_dict)
