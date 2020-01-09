'''
处理图像--右下侧  历史记录
'''
import cv2
import tkinter as tk
from tkinter.filedialog import *
from PIL import Image, ImageTk

import project_start
import globalvar

'''显示原图'''
def show_src_img():
    project_start.show_img_method(filePath='none', flag=2)

'''显示指定历史记录的图像'''
def show_index_img(index):
    project_start.show_img_method(filePath='none', flag=3, index=index)

'''更新历史记录'''
def update_right_button_setting(fra):
    for b in fra.winfo_children():  # fr3.winfo_children() 容器中所有的元素
        b.destroy()  # 销毁
        # x.grid_forget()  # 将元素隐藏

    b_width = 25
    b1 = tk.Button(fra, text='原图', width=b_width, command=lambda: show_src_img())
    b1.pack(padx=10, pady=1)

    img_list = globalvar.get_value('img_list_global')

    history_num = len(img_list) - 16
    if history_num < 1:
        history_num = 1
    num = history_num
    for b in img_list[history_num:]:
        bt = tk.Button(fra, text=('历史记录', num), width=b_width,  command=lambda num=num: show_index_img(num))
        bt.pack(padx=10, pady=1)
        num = num + 1
