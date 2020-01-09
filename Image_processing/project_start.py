'''
主界面管理
'''
import cv2
import tkinter as tk
from tkinter.filedialog import *
from PIL import Image, ImageTk
import numpy as np

import catalogs
import processing_setting
import globalvar
import img_history_setting

'''
全局参数
'''
# 左窗口宽度
left_win_w = 100
# 右窗口宽度
right_w = 200
# 右上窗口高度
right_top_h = 200
# 总界面宽和高
root_w = 1500
root_h = 800
# 显示图像区域宽和高
w_nor = root_w - right_w - right_top_h  # 显示图像区域的总宽度
h_nor = root_h                          # 显示图像区域的总高度
# 图片放置位置
img_label = None

save_data = {}

# 自适应窗口显示图像
def adjust_img_size(img):
    global w_nor, h_nor
    w, h = img.size

    if w > w_nor or h > h_nor:
        if w/w_nor >= h/h_nor:
            img = img.resize((w_nor, int(w_nor/w*h)))
        else:
            img = img.resize((int(h_nor/h*w), h_nor))
        # 只进行缩放显示，并不进行真正的图片缩放，所以此处不再保存

    return img

# 接受图片路径/图片本身，且优先检查是否是图片本身
# flag=0(默认)表示是传入的路径，只能出现在打开文件的时候
# flag=1表示传入的是文件本身
# flag=2表示显示原图
# flag=3表示显示指定图片，需要下标
def show_img_method(filePath, flag=0, index=-1):
    img_win = globalvar.get_value('img_win_global')
    img_list = globalvar.get_value('img_list_global')
    if flag == 1:
        scal = globalvar.get_value('pulley_flag_global')
        if scal == 'resize':  #当是旋转操作时取消自适应窗口
            show_img = img_list[len(img_list)-1]
        else:
            show_img = adjust_img_size(img_list[len(img_list)-1])
        show_img = ImageTk.PhotoImage(show_img)
    elif flag == 2:
        show_img = adjust_img_size(img_list[0])
        show_img = ImageTk.PhotoImage(show_img)
    elif flag == 3:
        show_img = adjust_img_size(img_list[index])
        show_img = ImageTk.PhotoImage(show_img)
    elif os.path.exists(filePath) and os.path.isfile(filePath):
        show_img = Image.open(filePath)
        img_list.append(show_img)
        globalvar.set_value('img_list_global', img_list)
        show_img = adjust_img_size(show_img)
        show_img = ImageTk.PhotoImage(show_img)
    else:
        return None

    save_data[filePath] = show_img

    # 需要图片居中
    global w_nor, h_nor
    w, h = show_img.width(), show_img.height()
    w_p, h_p = (w_nor-w)//2 + 100, (h_nor-h)//2 + 40

    global img_label
    # 保证再次打开文件的时候可以正常显示而不覆盖
    for c in img_win.winfo_children():
        c.destroy()

    img_label = tk.Label(img_win, image=show_img)
    img_label.place(x=w_p, y=h_p)

# ----------------------------------------------------------------------------------------------------------------------------
# 主界面设置
# ----------------------------------------------------------------------------------------------------------------------------
def start_project():
    root = tk.Tk()
    root.title('图像处理软件')
    root_w, root_h = root.maxsize()
    root_w = root_w - 300
    root_h = root_h - 200
    root.geometry('{}x{}'.format(root_w, root_h))  # 大小，用字母x分割

    # 菜单项
    bar = catalogs.menu_sel_area(root)

    # 处理图像--左侧
    fr1 = tk.Frame(root, width=left_win_w, bg='lightblue', bd=2, relief='ridge')
    fr1.propagate(False)  # 取消自适应
    processing_setting.add_left_button_setting(fr1)
    fr1.pack(side='left', anchor='n', fill='y')

    # 处理图像--中间
    fr2 = tk.Frame(root, width=root_w - left_win_w - right_w, bd=1, relief='ridge')
    fr2.propagate(False)  # 取消自适应
    globalvar.set_value('img_win_global', fr2)
    img_path = r'C:\Users\DHW\Pictures\img\456.jpg'
    globalvar.set_value('img_path_global', img_path)
    img_list = []
    globalvar.set_value('img_list_global', img_list)
    show_img_method(img_path)
    fr2.pack(side='left', fill='y')

    # 处理图像--右上侧
    fr3 = tk.Frame(root, width=right_w, height=right_top_h, bd=0, relief='ridge')
    fr3.propagate(False)  # 取消自适应
    v = tk.StringVar()
    w = tk.Label(fr3, textvariable=v, fg='red')
    globalvar.set_value('item_win_global', w)
    v.set("欢迎使用！")
    w.pack()
    globalvar.set_value('show_parameter_win_global', fr3)
    fr3.pack(side='top', anchor='e')

    # 处理图像--右下侧
    fr4 = tk.Frame(root, width=right_w, height=root_h - right_top_h, bd=1, bg='lightblue', relief='ridge')
    fr4.propagate(False)  # 取消自适应
    globalvar.set_value('show_history_win_global', fr4)
    img_history_setting.update_right_button_setting(fr4)
    fr4.pack(side='bottom', anchor='e')

    # 显示菜单项
    root.config(menu=bar)
    # 显示窗口
    root.mainloop()


# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    globalvar._init()
    start_project()