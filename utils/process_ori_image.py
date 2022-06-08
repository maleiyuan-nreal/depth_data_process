import cv2
import os
import numpy as np
import shutil
    
    
def process_ori_image(args, ori_image_path, output_image_path):
    shutil.copy(ori_image_path, os.path.join(args.output_path, output_image_path))