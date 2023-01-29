import copy


def format_nds(image_id, image_shape, info_dict):
    height, weight, *tmp = image_shape
    nds_data_item = dict()
    nds_data_item["image_id"] = "{:0>8d}".format(image_id)
    nds_data_item["image_path"] = info_dict["output_images_path"]

    nds_data_item["depth_path"] = info_dict["output_depths_path"]
    nds_data_item["segmentation_path"] = info_dict["output_segmentations_path"] if "output_segmentations_path" in info_dict else ""
    nds_data_item["image_height"] = height
    nds_data_item["image_width"] = weight
    extra_info = dict()
    if "fx" in info_dict:
        extra_info = {"fx": info_dict["fx"],
                    "fy": info_dict["fy"],
                    "cx": info_dict["cx"],
                    "cy": info_dict["cy"]}
    nds_data_item["extra_info"] = extra_info
    return copy.deepcopy(nds_data_item)



if __name__ == "__main__":
    nds_info = [1, (384, 384, 3)]
    path_dict = {'ori_images_path': '/home/lyma/FOD/datasets/inria/images/154.jpg', 'output_images_path': 'inria/images/154.jpg', 
                 'ori_depths_path': '/home/lyma/FOD/datasets/inria/depths/154.jpg', 'output_depths_path': 'inria/depths/154.png', 
                 'ori_segmentations_path': '/home/lyma/FOD/datasets/inria/segmentations/154.png', 'output_segmentations_path': 'inria/segmentations/154.png'}
    format_nds(nds_info, path_dict)