import os
from tqdm import tqdm
import logging


from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, get_file_name
)


def common_process(obj, args, total):
    """
    func1: check dir exists
    func2: set pbar
    func3: delete nds_file is exists
    """
    dataset_name = obj.NAME
    check_and_make_dir(os.path.join(args.output_path, dataset_name))
    check_and_make_dir(os.path.join(args.output_path, obj.OUTPUT_DIR))
    for dirs in obj.DATA_TYPE_LIST:
        check_and_make_dir(os.path.join(args.output_path, dataset_name, "data", dirs))
        
    logging.info(obj.NAME+" is processing")
    
    pbar = tqdm(total=total)
    pbar.set_description("Creating {} nds dataset: ".format(dataset_name))
    
    
    nds_path = obj.NDS_FILE_NAME
    if os.path.exists(nds_path):
        os.remove(nds_path)
    logging.info(f"nds_path: {obj.NDS_FILE_NAME}")
    
    return pbar

def get_path(obj, ori_image_path):
    """
    inria、nyuv2、posetrack 需要将depth的jpg图转换为png图
    
    
    OUTPUT:
        输出的路径为保存相对路径
    
    """
    path_dict = dict()
    path_dict["ori_images_path"] = ori_image_path
    image_name_only = get_file_name(ori_image_path)
    for input_dir, data_type in zip(obj.SUB_INPUT_DIR, obj.DATA_TYPE_LIST):
        if data_type == "images":
            path_dict["output_"+data_type+"_path"] = os.path.join(obj.OUTPUT_DIR, data_type, image_name_only)
        elif data_type == "depths":
            if obj.DPETH_SUFFIX == "jpg":
                path_dict["ori_"+data_type+"_path"] = os.path.join(obj.INPUT_DIR, obj.NAME, input_dir, image_name_only)
            elif obj.DPETH_SUFFIX == "png":
                path_dict["ori_"+data_type+"_path"] = os.path.join(obj.INPUT_DIR, obj.NAME, input_dir, image_name_only.split(".")[0]+".png")
            else:
                raise NotImplementedError 
            path_dict["output_"+data_type+"_path"] = os.path.join(obj.OUTPUT_DIR, data_type, image_name_only.split(".")[0]+".png")
        elif data_type == "segmentations":
            path_dict["ori_"+data_type+"_path"] = os.path.join(obj.INPUT_DIR, obj.NAME, input_dir, image_name_only.split(".")[0]+".png")
            path_dict["output_"+data_type+"_path"] = os.path.join(obj.OUTPUT_DIR, data_type, image_name_only.split(".")[0]+".png")
        else:
            raise Exception("Unknown data type: " + data_type) 
        
    return path_dict