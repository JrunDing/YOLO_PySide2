# -*- coding: utf-8 -*-
# 存储工程的全局变量



def _init():  # 初始化
    global _global_dict
    _global_dict = {}
    #_global_dict['model_path'] = 'model_data/yolov5_s.pth'
    #_global_dict['file_type'] = 0
    #_global_dict['confidence'] = 0.5
    #_global_dict['IoU'] = 0.3


def set_value(key, value):
    # 定义一个全局变量
    _global_dict[key] = value

def get_value(key):
    # 获得一个全局变量，不存在则提示读取对应变量失败
    try:
        return _global_dict[key]
    except:
        print('读取'+key+'失败\r\n')
