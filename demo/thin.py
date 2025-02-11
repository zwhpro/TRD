#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/9 12:39
# @Author  : Zheng Wenhao
# @File    : thin.py
# @Software: PyCharm
import os
import torch




def thin_pth():
    pth_path = r'F:\git\mmyolo\work_dirs\yolov5_s-v61_fast_1xb12-40e_cat\epoch_10.pth'
    state_dict = torch.load(pth_path)
    work_dir = os.path.dirname(pth_path)
    pth_name = str(os.path.basename(pth_path)).split('.')[0]
    out_path = os.path.join( work_dir, f'{pth_name}_thin.pth')
    torch.save(state_dict['state_dict'], out_path)
    print("Thin pth successful")
    print('| save path:', out_path)


if __name__ == '__main__':
    thin_pth()

