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
    ---- DATA_TYPE_LIST[0]
    ---- DATA_TYPE_LIST[1]
    ---- DATA_TYPE_LIST[2]
    ...
"""
import os


class INRIA():
    def __init__(self, output_path) -> None:
     
        self.NAME = "inria"
        self.INPUT_DIR = "/home/lyma/FOD/datasets/"
        self.NDS_FILE_NAME = os.path.join(output_path, self.NAME+"_images_pool.nds")
        
        self.SUB_INPUT_DIR = ["images", "depths", "segmentations"]
        self.DATA_TYPE_LIST = ["images", "depths", "segmentations"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.SUFFIX = "jpg"
    
    
class NYUV2():  
    def __init__(self, output_path) -> None:
        self.NAME = "nyuv2"
        self.INPUT_DIR = "/home/lyma/FOD/datasets/"
        self.NDS_FILE_NAME = os.path.join(output_path, self.NAME+"_images_pool.nds")
        self.SUB_INPUT_DIR = ["images", "depths", "segmentations"]
        self.DATA_TYPE_LIST = ["images", "depths", "segmentations"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.SUFFIX = "jpg"
    
    
class POSETRCK(): 
    def __init__(self, output_path) -> None:
        self.NAME = "posetrack" 
        self.INPUT_DIR = "/home/lyma/FOD/datasets/"
        self.NDS_FILE_NAME = os.path.join(output_path, self.NAME+"_images_pool.nds")
        self.SUB_INPUT_DIR = ["images", "depths", "segmentations"]
        self.DATA_TYPE_LIST = ["images", "depths", "segmentations"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.SUFFIX = "jpg"



class ReDWeb():
    def __init__(self, output_path) -> None:
        self.NAME = "ReDWeb_V1" 
        self.INPUT_DIR = "/home/lyma/SHARE_DATA/datadepth/reDWeb/"
        self.NDS_FILE_NAME = os.path.join(output_path, self.NAME+"_images_pool.nds")
        
        self.SUB_INPUT_DIR = ["Imgs", "RDs"]
        self.DATA_TYPE_LIST = ["images", "depths"]
        assert len(self.SUB_INPUT_DIR) == len(self.DATA_TYPE_LIST)
        self.SUFFIX = "png"
        
        
if __name__ == "__main__":
    inria_obj = INRIA(".")