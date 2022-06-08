import cv2
import os
import numpy as np
import shutil


def process_segmentation(args, ori_seg_path, output_seg_path):
    shutil.copy(ori_seg_path, os.path.join(args.output_path, output_seg_path))