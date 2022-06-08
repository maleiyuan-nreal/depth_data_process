import cv2
import os
import numpy as np
import shutil


from config.data_config import IMAGE_OUTPUT_DIR


def process_segmentation(ori_seg_path, output_seg_path):
    shutil.copy(ori_seg_path, os.path.join(IMAGE_OUTPUT_DIR, output_seg_path))