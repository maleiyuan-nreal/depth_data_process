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
import os


class INRIA():
    def __init__(self, output_path) -> None:

        self.NAME = "inria"
        self.INPUT_DIR = "/home/lyma/FOD/FocusOnDepth/datasets"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["images", "depths", "segmentations"]
        self.DATA_TYPE_LIST = ["images", "depths", "segmentations"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "jpg"


class NYUV2():
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


class POSETRCK():
    def __init__(self, output_path) -> None:
        self.NAME = "posetrack"
        self.INPUT_DIR = "/home/lyma/FOD/FocusOnDepth/datasets"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["images", "depths", "segmentations"]
        self.DATA_TYPE_LIST = ["images", "depths", "segmentations"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "jpg"


class ReDWeb():
    def __init__(self, output_path) -> None:
        self.NAME = "ReDWeb_V1"
        self.INPUT_DIR = "/home/lyma/SHARE_DATA/datadepth/reDWeb/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.NDS_TRAIN_FILE_NAME = os.path.join(
            output_path, self.NAME, "train_annotation.nds")
        self.NDS_VAL_FILE_NAME = os.path.join(
            output_path, self.NAME, "val_annotation.nds")
        self.NDS_TEST_FILE_NAME = os.path.join(
            output_path, self.NAME, "test_annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["Imgs", "RDs"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "png"


class MegaDepth():
    def __init__(self, output_path) -> None:
        self.NAME = "MegaDepth_v1"
        self.INPUT_DIR = "/home/lyma/SHARE_DATA/datadepth/MegaDepth/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["imgs", "depths"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "h5"


class TartanAir():
    def __init__(self, output_path) -> None:
        self.NAME = "tartanair"
        self.INPUT_DIR = "/home/lyma/SHARE_DATA/datadepth/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["image_left", "depth_left"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "npy"


class HRWSI():
    def __init__(self, output_path) -> None:
        self.NAME = "HR-WSI"
        self.INPUT_DIR = "/data/lyma/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["imgs", "gts"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "png"
        
   
class BlendedMVS():
    def __init__(self, output_path) -> None:
        self.NAME = "BlendedMVS"
        self.INPUT_DIR = "/data/lyma/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["blended_images", "rendered_depth_maps"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "pfm"
        

class IRS():
    def __init__(self, output_path) -> None:
        self.NAME = "IRSDataset"
        self.INPUT_DIR = "/data/lyma/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["", ""]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "exr"
        

class ApolloScape():
    def __init__(self, output_path) -> None:
        self.NAME = "apolloscape"
        self.INPUT_DIR = "/data/lyma/"
        self.NDS_FILE_NAME = os.path.join(
            output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = "data"
        self.SUB_INPUT_DIR = ["camera_5", "disparity"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "png"
        
                   
if __name__ == "__main__":
    inria_obj = INRIA(".")
