import copy


def format_nds(nds_info):
    image_id, output_image_path, ouput_depth_path, output_seg_path, image_shape = nds_info
    height, weight, *tmp = image_shape
    nds_data_item = dict()
    nds_data_item["image_id"] = "{:0>8d}".format(image_id)
    nds_data_item["image_path"] = output_image_path

    nds_data_item["depth_path"] = ouput_depth_path
    nds_data_item["segmentation_path"] = output_seg_path
    nds_data_item["image_height"] = height
    nds_data_item["image_width"] = weight
    nds_data_item["extra_info"] = dict()
    return copy.deepcopy(nds_data_item)
