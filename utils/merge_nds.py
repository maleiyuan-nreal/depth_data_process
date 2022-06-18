import os


from bfuncs import (
    check_and_make_dir, check_and_make_dir_for_file,
    load_json_items, get_file_name, save_json_items
)


def mergeFiles(obj, fileList: list):
    
    res_items = []
    
    for filePath in fileList:
        res_items.extend(load_json_items(filePath))
        
    save_json_items(obj.NDS_FILE_NAME, res_items)