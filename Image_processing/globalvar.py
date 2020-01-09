#!/usr/bin/python
# -*- coding: utf-8 -*-
import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
'''
全局变量管理:

img_path_global 当前图片路径，绝对路径

img_win_global  显示/当前图片窗口
show_parameter_win_global  表示参数窗口
show_history_win_global 表示历史记录显示的标签位置
item_win_global  滑轮标签显示的区域

img_list_global 表示操作的图像本身列表，img_list_global[0]代表原图，最后一个代表当前图像即传入的图像

last_img_global 表示滑轮调节的最后效果图

pulley_flag_global 滑轮操作标记类型
    light
    threshold
    resize
    rotate
    
single_case_flag_global 单例组件标记，None表示未初始化（下方的012代表下标）
    0-亮度
    1-阈值
    2-缩放
    3-旋转
'''
def _init():
    global _global_dict
    _global_dict = {}

    # 注意单例标签初始化
    single_case_flag = [None, None, None, None]
    set_value('single_case_flag_global', single_case_flag)


def set_value(name, value):
    _global_dict[name] = value

def get_value(name, defValue=None):
    try:
        return _global_dict[name]
    except KeyError:
        return defValue
