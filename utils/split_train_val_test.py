"""
Auther: lyma@nreal.ai
Date: 2022/06/13

Description: split NDS file 
Input: annotation.nds
Output: annotation_train.nds、annotation_test.nds、annotation_val.nds

"""
import os
import numpy as np
import logging


from config.data_config import ReDWeb
from bfuncs.files import load_json_items, save_json_items


seed = 0
train_split_ratio = 0.8

def split_file(obj):
    annot_items = load_json_items(obj.NDS_FILE_NAME)
    total = len(annot_items)
    logging.info("total samples {}".format(total))

    np.random.seed(seed)
    np.random.shuffle(annot_items)
    
    
    train_num = int(total*train_split_ratio)
    train_annot_items = annot_items[:train_num]
    val_annot_items = annot_items[train_num:]
    save_json_items(obj.NDS_TRAIN_FILE_NAME, train_annot_items)
    save_json_items(obj.NDS_VAL_FILE_NAME, val_annot_items)
    
    assert len(train_annot_items) + len(val_annot_items) == total
    logging.info("splitted to train&val, train samples:{}".format(train_num))
    
    
    
if __name__ == "__main__":
    logging.basicConfig(filename="log/split.log", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)
    
    redweb_obj = ReDWeb("/data/lyma/midas_data")
    logging.info("splitting {}".format(redweb_obj.NAME))
    split_file(redweb_obj)