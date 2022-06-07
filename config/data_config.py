"""
Midas data config

"""
import os


IMAGE_OUTPUT_DIR = "/mnt/cephfs/share/lyma/datasets/fod_mix_3_20220606/"


class INRIA():
    NAME = "inria"
    INPUT_DIR = "/home/lyma/FOD/datasets/"
    NDS_FILE_NAME = os.path.join(IMAGE_OUTPUT_DIR, NAME+"_images_pool.nds")
    
    
class NYUV2():  
    NAME = "nyuv2"
    INPUT_DIR = "/home/lyma/FOD/datasets/"
    NDS_FILE_NAME = os.path.join(IMAGE_OUTPUT_DIR, NAME+"_images_pool.nds")
    
    
class POSETRCK(): 
    NAME = "posetrack" 
    INPUT_DIR = "/home/lyma/FOD/datasets/"
    NDS_FILE_NAME = os.path.join(IMAGE_OUTPUT_DIR, NAME+"_images_pool.nds")



