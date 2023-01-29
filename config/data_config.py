"""
Midas data config


!!!!!!IMPORTANT!!!!!!
DATA_TYPE_LIST 最多支持三个: "images", "depths", "segmentations"
DATA_TYPE_LIST 与SUB_INPUT_DIR 一一对应


INPUT目录结构
    INPUT_DIR
    -- NAME
    ---- ...(一个数据集文件下可能有多个子文件, 这里的处理逻辑由各自的process_NAME.py进行)
    ------ SUB_INPUT_DIR[0]
    ------ SUB_INPUT_DIR[1]
    ------ SUB_INPUT_DIR[2]
    ...
    
OUTPUT目录结构 
    IMAGE_OUTPUT_DIR
    -- NAME
    ---- annotation.nds
    ---- data
    ----- ...(一个数据集文件下可能有多个子文件, 这里的处理逻辑由各自的process_NAME.py进行)
    -------- DATA_TYPE_LIST[0] (images)
    -------- DATA_TYPE_LIST[1] (depths)
    -------- DATA_TYPE_LIST[2] (segmentations)
    ...
    
"""

from process.process_megadepth import MegaDepth
from process.process_apolloscape import ApolloScape
from process.process_apolloscape_scene_parsing import ApolloScapeScene
from process.process_blendedmvs import BlendedMVS
from process.process_hrwsi import HRWSI
from process.process_inria import INRIA
from process.process_irs import IRS
from process.process_nyuv2 import NYUV2
from process.process_posetrack import POSETRCK
from process.process_redweb import ReDWeb
from process.process_tartanair import TartanAir


def init_obj(args):
    output_path = args.output_path
    transform_flag = args.transform
    obj_dict = dict()

    inria_obj = INRIA(output_path)
    nyuv2_obj = NYUV2(output_path)
    posetrack_obj = POSETRCK(output_path)

    megadepth_obj = MegaDepth(output_path)
    rw_obj = ReDWeb(output_path)
    blendedmvs_obj = BlendedMVS(output_path, transform_flag)
    apollospace_obj = ApolloScape(output_path)
    apollospace_scene_obj = ApolloScapeScene(output_path)
    hrwsi_obj = HRWSI(output_path)
    tartanair_obj = TartanAir(output_path, transform_flag)
    irs_obj = IRS(output_path)

    for obj in [blendedmvs_obj, apollospace_obj, apollospace_scene_obj, tartanair_obj, hrwsi_obj,
                inria_obj, nyuv2_obj, irs_obj, posetrack_obj, megadepth_obj, rw_obj]:
        obj_dict[obj.NAME] = obj

    return obj_dict


if __name__ == "__main__":
    inria_obj = INRIA(".")
