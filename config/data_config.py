"""
Midas data config


!!!!!!IMPORTANT!!!!!!
DATA_TYPE_LIST 最多支持三个: "images", "depths", "segmentations"
DATA_TYPE_LIST 与SUB_INPUT_DIR 一一对应


INPUT目录结构
    INPUT_DIR
    -- NAME
    ---- SUB_INPUT_DIR[0]
    ---- SUB_INPUT_DIR[1]
    ---- SUB_INPUT_DIR[2]
    ...
    
OUTPUT目录结构 
    IMAGE_OUTPUT_DIR
    -- NAME
    ---- annotation.nds
    ---- data
    ------ DATA_TYPE_LIST[0] (images)
    ------ DATA_TYPE_LIST[1] (depths)
    ------ DATA_TYPE_LIST[2] (segmentations)
    ...
    
TODO 输入图片默认是jpg, 后面看看需不需要改
"""
import os


class INRIA():
    def __init__(self, output_path) -> None:
     
        self.NAME = "inria"
        self.INPUT_DIR = "/home/lyma/FocusOnDepth/datasets"
        self.NDS_FILE_NAME = os.path.join(output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = os.path.join(self.NAME, "data") 
        self.SUB_INPUT_DIR = ["images", "depths", "segmentations"]
        self.DATA_TYPE_LIST = ["images", "depths", "segmentations"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "jpg"
    
    
class NYUV2():  
    def __init__(self, output_path) -> None:
        self.NAME = "nyuv2"
        self.INPUT_DIR = "/home/lyma/FocusOnDepth/datasets"
        self.NDS_FILE_NAME = os.path.join(output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = os.path.join(self.NAME, "data") 
        self.SUB_INPUT_DIR = ["images", "depths", "segmentations"]
        self.DATA_TYPE_LIST = ["images", "depths", "segmentations"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "jpg"
    
    
class POSETRCK(): 
    def __init__(self, output_path) -> None:
        self.NAME = "posetrack" 
        self.INPUT_DIR = "/home/lyma/FocusOnDepth/datasets"
        self.NDS_FILE_NAME = os.path.join(output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = os.path.join(self.NAME, "data") 
        self.SUB_INPUT_DIR = ["images", "depths", "segmentations"]
        self.DATA_TYPE_LIST = ["images", "depths", "segmentations"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "jpg"



class ReDWeb():
    def __init__(self, output_path) -> None:
        self.NAME = "ReDWeb_V1" 
        self.INPUT_DIR = "/home/lyma/SHARE_DATA/datadepth/reDWeb/"
        self.NDS_FILE_NAME = os.path.join(output_path, self.NAME, "annotation.nds")
        self.OUTPUT_DIR = os.path.join(self.NAME, "data") 
        self.SUB_INPUT_DIR = ["Imgs", "RDs"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.DPETH_SUFFIX = "png"
        
        
if __name__ == "__main__":
    inria_obj = INRIA(".")