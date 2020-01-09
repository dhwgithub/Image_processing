'''
处理图像设置-左侧
'''
import cv2
import tkinter as tk
from tkinter.filedialog import *
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

import project_start
import globalvar
import img_history_setting
import catalogs

'''
为了更新历史记录
每个图像处理必须调用这个函数
'''
def update_history():
    fra = globalvar.get_value('show_history_win_global')
    img_history_setting.update_right_button_setting(fra)

# 单通道
def single_method():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1]

    # show_img = show_img.convert('RGB')
    # print(type(show_img), show_img.size)

    im2 = show_img.convert('L')
    img_list.append(im2)
    globalvar.set_value('img_list_global', img_list)
    project_start.show_img_method(filePath='none', flag=1)

    update_history()

'''
矩阵画图
'''
def draw_img_method():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list) - 1].convert('RGB')
    draw = ImageDraw.Draw(show_img)
    # 坐标
    x1, y1 = (10, 10)
    x2, y2 = (100, 100)
    # outline 外线，fill填充
    draw.rectangle((x1, y1, x2, y2), outline="red", width=5)
    # image.save('filepath', 'jpeg')
    show_img.show()

'''
得到鼠标点击区域的两点坐标
'''
def get_mouse_point():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list) - 1].convert('RGB')

    plt.imshow(show_img, cmap=plt.get_cmap("gray"))
    # 能使你从当前的坐标系中读取n个点，返回这n个点的x，y坐标，均为nX1的向量。可以按回车提前结束读数。
    pos = plt.ginput(2)
    plt.close()

    p1, p2 = pos

    # left = min(p1[0], p2[0])
    # top = min(p1[1], p2[1])
    # right = max(p1[0], p2[0])
    # bottom = max(p1[1], p2[1])

    return pos

'''
鼠标点击截取图片
'''
def cut_img_method():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1].convert('RGB')

    p1, p2 = get_mouse_point()
    left = min(p1[0], p2[0])
    top = min(p1[1], p2[1])
    right = max(p1[0], p2[0])
    bottom = max(p1[1], p2[1])
    show_img = show_img.crop([left, top, right, bottom])

    img_list.append(show_img)
    globalvar.set_value('img_list_global', img_list)
    project_start.show_img_method(filePath=None, flag=1)

    update_history()

'''
添加左侧按钮的功能按钮
'''
def add_left_button_setting(fra):
    b1 = tk.Button(fra, text='剪切', width=10, command=lambda: cut_img_method())
    b1.pack(padx=10, pady=10)
    b1 = tk.Button(fra, text='灰度化', width=10, command=lambda: single_method())
    b1.pack(padx=10, pady=10)
    b2 = tk.Button(fra, text='图像缩放', width=10, command=lambda: catalogs.resize_file())
    b2.pack(padx=10, pady=10)
    b3 = tk.Button(fra, text='亮度', width=10, command=lambda: catalogs.light_file())
    b3.pack(padx=10, pady=10)
    b4 = tk.Button(fra, text='美颜', width=10, command=lambda: catalogs.beauty_file())
    b4.pack(padx=10, pady=10)
    b4 = tk.Button(fra, text='阈值', width=10, command=lambda: catalogs.threshold_file())
    b4.pack(padx=10, pady=10)
    b4 = tk.Button(fra, text='图像旋转', width=10, command=lambda: catalogs.rotate_file())
    b4.pack(padx=10, pady=10)
