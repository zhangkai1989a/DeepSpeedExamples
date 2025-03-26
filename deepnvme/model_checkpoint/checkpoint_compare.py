#This script is for testing whether two checkpoints match; it prints all the differences

import torch
import os
import sys
import pickle
from collections import OrderedDict

exclude_key_str = {'ds_config/checkpoint/writer'}

def main():
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    print ("Begin comparison")
    print ("The first directory {}" .format(dir1))
    print ("The second directory {}" .format(dir2))
    print (' ')

    file_list1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    file_list2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]
    common_files = []
    
    for f in file_list1:
        if not (f in file_list2):
            log_error_file_mismatch_first(f)
        else:
            common_files.append(f)
    for f in file_list2:
        if not (f in file_list1):
            log_error_file_mismatch_second(f)
    
    for f in common_files:
        full_dir1 = os.path.join(dir1, f)
        full_dir2 = os.path.join(dir2, f)
        print ("Begin comparison")
        print("The first checkpoint {}" .format(full_dir1))
        print("The second checkpoint {}" .format(full_dir2))
        print(' ')
        model_first = torch.load(full_dir1)
        model_second = torch.load(full_dir2)
        object_compare(model_first, model_second, [])


def object_compare(model_first, model_second, key_chain):
    if not (type(model_first) == type(model_second)):
        log_error_value_mismatch(model_first, model_second, key_chain)
        return

    if type(model_first) is list:
        if len(model_first) != len(model_second):
            log_error_value_mismatch(model_first, model_second, key_chain)
            return
        for i in range(len(model_first)):
            object_compare(model_first[i], model_second[i], key_chain)
        return

    if type(model_first) is dict or type(model_first) is OrderedDict:
        common_keys = []
        for key in model_first:
            if key not in model_second:
                key_chain.append(key)
                log_error_key_mismatch_first(model_first[key], key_chain)
                key_chain.pop()
            else:
                common_keys.append(key)
                
        for key in model_second:
            if key not in model_first:
                key_chain.append(key)
                log_error_key_mismatch_second(model_second[key], key_chain) 
                key_chain.pop()
                
        for key in common_keys:
            key_chain.append(key)
            object_compare(model_first[key], model_second[key], key_chain)
            key_chain.pop()
        return
	
    if hasattr(model_first, '__dict__'):
        equality = (model_first.__dict__ == model_second.__dict__)
    else:
        equality = (model_first == model_second)
    if type(equality) is not bool:
        equality = (equality.all())
    if not equality:
        log_error_value_mismatch(model_first, model_second, key_chain)
    return    


def log_error_file_mismatch_first(filename):
    print("The following file appeared in the first but not the second directory: {}" .format(filename))
    print(' ')
    

def log_error_file_mismatch_second(filename):
    print("The following key appeared in the second but not the first directory: {}" .format(filename))
    print(" ")


def log_error_key_mismatch_first(model, key_chain):
    key_str = "/".join(key_chain)
    if not (key_str in exclude_key_str):
        print("The following key appeared in the first but not the second model: {}" .format(key_str))
        print("The value of the first model is: {}" .format(model))
        print(" ") 


def log_error_key_mismatch_second(model, key_chain):
    key_str = "/".join(key_chain)
    if not (key_str in exclude_key_str):
        print("The following key appeared in the second but not the first model: {}" .format(key_str))
        print("The value of the second model is: {}" .format(model))
        print(" ") 


def log_error_value_mismatch(model_first, model_second, key_chain):
    print ("The values of the following key do not match: {}" .format("/".join(key_chain)))
    print ("The value of the first model is: {}" .format(model_first))
    print ("The value of the second model is: {}" .format(model_second))
    print(" ")

if __name__ == "__main__":
    main()
