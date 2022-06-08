import cv2
import os
import numpy as np
import shutil


from config.data_config import IMAGE_OUTPUT_DIR
    
    
def process_ori_image(ori_image_path, output_image_path):
    shutil.copy(ori_image_path, os.path.join(IMAGE_OUTPUT_DIR, output_image_path))