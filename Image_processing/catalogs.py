#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
目录管理
'''
import tkinter as tk
from tkinter.filedialog import *
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw, ImageFont
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
from tkinter import messagebox  # 引入弹窗库

import project_start
import globalvar
import img_history_setting
import processing_setting

# ----------------------------------------------------------------------------------------------------------------------------
# 文件菜单栏设置
# ----------------------------------------------------------------------------------------------------------------------------
# 打开图片
def open_file():
    get_path = askopenfilename()
    globalvar.set_value('img_path_global', get_path)
    if not os.path.isfile(get_path):
        print('不是文件')
        return

    # 重置操作列表
    img_list = []
    globalvar.set_value('img_list_global', img_list)

    # 需要添加历史记录重置
    img_history_setting.update_right_button_setting(globalvar.get_value('show_history_win_global'))

    project_start.show_img_method(get_path)

# 另存为
import easygui as g
def save_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list) - 1]
    show_img.save('copy.jpg')

    file_ = open('copy.jpg', 'rb').read()
    another_path = g.filesavebox(default='new.jpg')
    if os.path.splitext(another_path)[1] != '.jpg':
        another_path += '.jpg'
    with open(another_path, 'wb') as new_file:
        new_file.write(file_)

'''
文件菜单
'''
def add_file_bar(file_):
    file_.add_command(label='打开图片', command=lambda: open_file())
    file_.add_command(label='另存为', command=lambda: save_file())


# ----------------------------------------------------------------------------------------------------------------------------
# 编辑菜单栏设置
# ----------------------------------------------------------------------------------------------------------------------------
# 撤销
def withdraw_file():
    img_list = globalvar.get_value('img_list_global')

    num = len(img_list)
    if num == 1:  # 只有一张原图，不需要撤销
        return

    img_list.pop()
    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=3, index=(len(img_list)-1))
    processing_setting.update_history()

# 水印
def water_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1].convert('RGBA')  # 转化为RGBA模式

    # 1.创建一张透明的图片，在透明图片上写上字，然后与原图合并
    w, h = show_img.size
    water = Image.new('RGBA', (w, h), (0, 0, 0, 0))

    fn_size = 100

    fn_str = g.enterbox(msg='请输入文字', title='水印文字设置', default='')
    if len(fn_str) == 0:
        fn_str = 'watermark'

    fn = ImageFont.truetype(r'C:\Windows\Fonts\Arial.ttf', fn_size)
    fn_w, fn_h = fn.getsize(fn_str)  # 获取字体宽高

    ct_w, ct_h = (w - fn_w) // 2, (h - fn_h) // 2  # 字体位置设置在中心
    draw = ImageDraw.Draw(water)
    draw.text((ct_w, ct_h), fn_str, font=fn, fill=(255, 255, 255, 100))
    water = water.rotate(45)

    # 2.图片合并，原图被水印覆盖
    show_img = Image.alpha_composite(show_img, water)

    img_list.append(show_img)
    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=1)
    processing_setting.update_history()

# 拷贝
def copy_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1].convert('RGB')
    show_img.save('copy.jpg')

# 等比例缩放
def resize_file():
    globalvar.set_value('pulley_flag_global', 'resize')
    single_case_flag = globalvar.get_value('single_case_flag_global')
    fra = globalvar.get_value('show_parameter_win_global')
    index = 2
    if single_case_flag[index] is None:
        item = []
        light_ = tk.Scale(fra, label='等比例缩放', orient=tk.HORIZONTAL,
                          from_=0.5, to=5, length=200,
                          tickinterval=1, resolution=0.1,
                          command=do_pulley)
        light_.set(1)
        item.append(light_)

        ok_b = tk.Button(fra, text='确定', width=10, command=lambda: ok_button_method())
        item.append(ok_b)

        single_case_flag[index] = item
        globalvar.set_value('single_case_flag_global', single_case_flag)

    clearfr()
    showfr(single_case_flag[index])


# 合并
def merge_file():
    get_path = askopenfilename()
    if not os.path.isfile(get_path):
        print('不是文件')
        return

    img_list = globalvar.get_value('img_list_global')
    im1 = img_list[len(img_list)-1].convert('RGB')
    im2 = Image.open(get_path).convert('RGB').resize(im1.size)
    show_img = Image.blend(im1, im2, 0.5)  # im1:"alpha  im3:1-alpha透明度
    img_list.append(show_img)
    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=1)
    processing_setting.update_history()

# 镜面翻转
def mirror_file(val):
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1].convert('RGB')
    if val == 'h':
        show_img = show_img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        show_img = show_img.transpose(Image.FLIP_TOP_BOTTOM)

    img_list.append(show_img)
    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=1)
    processing_setting.update_history()

# 证件照
def cvtBackground(path, color):
    """
    功能：给证件照更换背景色（常用背景色红、白、蓝）
    输入参数：path:照片路径
    color:背景色 <格式[B,G,R]>
    """
    im = cv2.imread(path)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # BGR和HSV的转换使用 cv2.COLOR_BGR2HSV
    mask = cv2.inRange(im_hsv, np.array([im_hsv[0, 0, 0] - 5, 100, 100]),
                       np.array([im_hsv[0, 0, 0] + 5, 255, 255]))  # 利用cv2.inRange函数设阈值，去除背景部分
    mask1 = mask  # 在lower_red～upper_red之间的值变成255
    img_median = cv2.medianBlur(mask, 5)  # 自己加，中值滤波，去除一些边缘噪点
    mask = img_median
    mask_inv = cv2.bitwise_not(mask)
    img1 = cv2.bitwise_and(im, im, mask=mask_inv)  # 将人物抠出
    bg = im.copy()
    rows, cols, channels = im.shape
    bg[:rows, :cols, :] = color
    img2 = cv2.bitwise_and(bg, bg, mask=mask)  # 将背景底板抠出
    img = cv2.add(img1, img2)

    return img

def cards_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1].convert('RGB')
    # show_img = np.array(show_img)

    show_img.save('temp.jpg')
    path = r'temp.jpg'
    colors = g.ccbox(msg='请选择你要传入照片的背景色', title='证件照', choices=('红色', '蓝色'))
    if colors == False:
        img = cvtBackground(path, [180, 0, 0])
    else:
        img = cvtBackground(path, [0, 0, 180])

    show_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_list.append(show_img)
    globalvar.set_value('img_list_global', img_list)
    project_start.show_img_method(filePath=None, flag=1)
    processing_setting.update_history()

'''
编辑菜单
'''
def add_edit_bar(edit_):
    edit_.add_command(label='撤销', command=lambda: withdraw_file())
    edit_.add_command(label='水印', command=lambda: water_file())
    edit_.add_command(label='拷贝', command=lambda: copy_file())
    edit_.add_command(label='证件照', command=lambda: cards_file())
    edit_.add_command(label='等比例缩放', command=lambda: resize_file())
    edit_.add_command(label='合并', command=lambda: merge_file())
    edit_.add_command(label='水平镜面翻转', command=lambda: mirror_file('h'))
    edit_.add_command(label='竖直镜面翻转', command=lambda: mirror_file('v'))


# ----------------------------------------------------------------------------------------------------------------------------
# 图像菜单栏设置
# ----------------------------------------------------------------------------------------------------------------------------
# 灰度化
def grey_file():
    processing_setting.single_method()

# 图像旋转（同时根据图片大小来改变尺寸）
def imgae_rotate(img, angle):
    height, width = img.shape[:2]
    max_w_h = max(height, width) * 2  # 旋转后图像尺寸

    # 构造旋转矩阵，第一个参数是旋转中心，第二个参数是旋转角度（0~360），第三个参数是缩放比例
    matRotate = cv2.getRotationMatrix2D((width//2, height//2), angle, 1)

    '''
    仿射变换中用到的旋转矩阵是一个2*3，其中2*2是旋转矩阵，2*1是平移矩阵
    因此把平移矩阵改一下即可旋转后不被截断
    '''
    matRotate[0, 2] += (max_w_h - width) / 2  # 重点在这步
    matRotate[1, 2] += (max_w_h - height) / 2  # 重点在这步

    # 仿射变换，第一个参数是原图像，第二个参数是旋转矩阵，第三个参数是输出矩阵的大小（若图片大小不变则为height, width）
    dst = cv2.warpAffine(img, matRotate, (max_w_h, max_w_h))
    rows, cols = dst.shape[:2]

    # np.array.any()是或操作，任意一个元素为True，输出为True。
    # np.array.all()是与操作，所有元素为True，输出为True。
    # 找到左边界
    for col in range(cols):
        if dst[:, col].any():
            left = col
            break
    # 找到右边界
    for col in range(cols - 1, 0, -1):
        if dst[:, col].any():
            right = col
            break
    # 找到上边界
    for row in range(rows):
        if dst[row, :].any():
            up = row
            break
    # 找到下边界
    for row in range(rows - 1, 0, -1):
        if dst[row, :].any():
            down = row
            break

    res_widths = abs(right - left)
    res_heights = abs(down - up)

    # 最后计算出最终图像的大小
    res = np.zeros([res_heights, res_widths, 3], np.uint8)

    # 剪切成最小图片
    for res_width in range(res_widths):
        for res_height in range(res_heights):
            res[res_height, res_width] = dst[up + res_height, left + res_width]

    return res

# 滑轮操作设置
def do_pulley(v):
    labels = globalvar.get_value('item_win_global')
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1].convert('RGB')

    flag = globalvar.get_value('pulley_flag_global')
    if flag == 'light':
        show_img = ImageEnhance.Brightness(show_img).enhance(float(v))
    elif flag == 'threshold':
        show_img = cv2.threshold(np.array(show_img), int(v), 255, cv2.THRESH_BINARY)[1]
        show_img = Image.fromarray(show_img)
    elif flag == 'resize':
        v = float(v)
        w, h = show_img.size
        w, h = int(w/v), int(h/v)
        if w <= 0:
            w = 1
        if h <= 0:
            h = 1
        show_img = show_img.resize((w, h))
    elif flag == 'rotate':
        show_img = imgae_rotate(np.array(show_img), float(v))
        show_img = Image.fromarray(show_img)

    # globalvar.set_value('last_img_global', show_img)

    img_list = globalvar.get_value('img_list_global')
    img_list.append(show_img)
    globalvar.set_value('img_list_global', img_list)

    show_img = ImageTk.PhotoImage(show_img)
    labels.config(image=show_img)

    project_start.show_img_method(filePath=None, flag=1)

    img_list = globalvar.get_value('img_list_global')
    last_img = img_list.pop()

    globalvar.set_value('last_img_global', last_img)

def clearfr():
    fra = globalvar.get_value('show_parameter_win_global')
    for x in fra.winfo_children():
        #         x.destroy() #销毁
        x.grid_forget()  # 隐藏元素 x.pack_gorget()


def showfr(zj):
    # 安照行列排列zj中的元素
    r = 0
    for x in zj:
        x.grid(row=r, column=0)
        r = r + 1

'''
滑轮参数确定按钮
'''
def ok_button_method():
    img_list = globalvar.get_value('img_list_global')
    last_img = globalvar.get_value('last_img_global')
    img_list.append(last_img)
    globalvar.set_value('img_list_global', img_list)
    processing_setting.update_history()

# 亮度
def light_file():
    globalvar.set_value('pulley_flag_global', 'light')
    single_case_flag = globalvar.get_value('single_case_flag_global')
    fra = globalvar.get_value('show_parameter_win_global')
    index = 0
    if single_case_flag[index] is None:
        item = []
        light_ = tk.Scale(fra, label='亮度', orient=tk.HORIZONTAL,
                              from_=0, to=5, length=200,
                              tickinterval=1, resolution=0.5,
                              command=do_pulley)
        light_.set(1)
        item.append(light_)

        ok_b = tk.Button(fra, text='确定', width=10, command=lambda: ok_button_method())
        item.append(ok_b)

        single_case_flag[index] = item
        globalvar.set_value('single_case_flag_global', single_case_flag)

    clearfr()
    showfr(single_case_flag[index])

# 饱和度
def saturation_file():
    pass
# 反相
def reverse_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1].convert('RGB')
    show_img = show_img.point(lambda x: 255-x)

    img_list.append(show_img)
    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=1)

    processing_setting.update_history()

# 阈值
def threshold_file():
    globalvar.set_value('pulley_flag_global', 'threshold')
    single_case_flag = globalvar.get_value('single_case_flag_global')
    fra = globalvar.get_value('show_parameter_win_global')
    index = 1
    if single_case_flag[index] is None:
        item = []
        light_ = tk.Scale(fra, label='阈值', orient=tk.HORIZONTAL,
                          from_=0, to=255, length=200,
                          tickinterval=51, resolution=5,
                          command=do_pulley)
        light_.set(1)
        item.append(light_)

        ok_b = tk.Button(fra, text='确定', width=10, command=lambda: ok_button_method())
        item.append(ok_b)

        single_case_flag[index] = item
        globalvar.set_value('single_case_flag_global', single_case_flag)

    clearfr()
    showfr(single_case_flag[index])

# 剪切
def cut_file():
    processing_setting.update_history()
    pass
# 图像旋转
def rotate_file():
    globalvar.set_value('pulley_flag_global', 'rotate')
    single_case_flag = globalvar.get_value('single_case_flag_global')
    fra = globalvar.get_value('show_parameter_win_global')
    index = 3
    if single_case_flag[index] is None:
        item = []
        light_ = tk.Scale(fra, label='旋转', orient=tk.HORIZONTAL,
                          from_=0, to=360, length=200,
                          tickinterval=60, resolution=1,
                          command=do_pulley)
        light_.set(0)
        item.append(light_)

        ok_b = tk.Button(fra, text='确定', width=10, command=lambda: ok_button_method())
        item.append(ok_b)

        single_case_flag[index] = item
        globalvar.set_value('single_case_flag_global', single_case_flag)

    clearfr()
    showfr(single_case_flag[index])

'''
图像菜单
'''
def add_image_bar(image_):
    image_.add_command(label='灰度化', command=lambda: grey_file())
    image_.add_command(label='亮度', command=lambda: light_file())
    image_.add_command(label='饱和度', command=lambda: saturation_file())
    image_.add_command(label='反相', command=lambda: reverse_file())
    image_.add_command(label='阈值', command=lambda: threshold_file())
    image_.add_command(label='剪切', command=lambda: cut_file())
    image_.add_command(label='图像旋转', command=lambda: rotate_file())

# ----------------------------------------------------------------------------------------------------------------------------
# 滤镜菜单栏设置
# ----------------------------------------------------------------------------------------------------------------------------
# 浮雕效果
def float_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1].convert('RGB')
    show_img = show_img.filter(ImageFilter.EMBOSS)
    img_list.append(show_img)

    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=1)
    processing_setting.update_history()

# 高斯模糊
def vague_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list) - 1].convert('RGB')
    show_img = show_img.filter(ImageFilter.GaussianBlur)
    img_list.append(show_img)

    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=1)
    processing_setting.update_history()

# 锐化
def sharpening_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list) - 1].convert('RGB')
    show_img = show_img.filter(ImageFilter.SHARPEN)
    img_list.append(show_img)

    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=1)
    processing_setting.update_history()

'''
马赛克，传入图像，左上角坐标，宽和高
'''
def do_mosaic(frame, x, y, w, h, neightbor=9):
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return

    for i in range(0, h-neightbor, neightbor):
        for j in range(0, w-neightbor, neightbor):
            rect = [j+x, i+y, neightbor, neightbor]
            color = frame[i+y][j+x].tolist()
            left_up = (rect[0], rect[1])
            right_down = (rect[0]+neightbor-1, rect[1]+neightbor-1)
            cv2.rectangle(frame, left_up, right_down, color, -1)

# 马赛克
def Mosaic_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list) - 1].convert('RGB')
    im = np.array(show_img)
    show_img = im.copy()

    p1, p2 = processing_setting.get_mouse_point()

    left = min(p1[0], p2[0])
    top = min(p1[1], p2[1])
    right = max(p1[0], p2[0])
    bottom = max(p1[1], p2[1])

    do_mosaic(show_img, int(p1[0]), int(p1[1]), int(abs(right-left)), int(abs(bottom-top)))

    show_img = Image.fromarray(show_img)
    img_list.append(show_img)
    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=1)
    processing_setting.update_history()

# 美颜
def beauty_file():
    img_list = globalvar.get_value('img_list_global')
    src = img_list[len(img_list)-1].convert('RGB')
    src = np.array(src)

    dst = np.zeros_like(src)
    # int value1 = 3, value2 = 1; 磨皮程度与细节程度的确定
    v1 = 3
    v2 = 1
    dx = v1 * 5  # 双边滤波参数之一
    fc = v1 * 12.5  # 双边滤波参数之一
    p = 0.1

    temp4 = np.zeros_like(src)

    temp1 = cv2.bilateralFilter(src, dx, fc, fc)
    temp2 = cv2.subtract(temp1, src)
    temp2 = cv2.add(temp2, (10, 10, 10, 128))
    temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
    temp4 = cv2.subtract(cv2.add(cv2.add(temp3, temp3), src), (10, 10, 10, 255))

    dst = cv2.addWeighted(src, p, temp4, 1 - p, 0.0)
    dst = cv2.add(dst, (10, 10, 10, 255))

    show_img = Image.fromarray(dst)

    img_list.append(show_img)
    globalvar.set_value('img_list_global', img_list)

    project_start.show_img_method(filePath=None, flag=1)
    processing_setting.update_history()

def add_filter_bar(filter_):
    filter_.add_command(label='浮雕效果', command=lambda: float_file())
    filter_.add_command(label='高斯模糊', command=lambda: vague_file())
    filter_.add_command(label='锐化', command=lambda: sharpening_file())
    filter_.add_command(label='马赛克', command=lambda: Mosaic_file())
    filter_.add_command(label='美颜', command=lambda: beauty_file())

# ----------------------------------------------------------------------------------------------------------------------------
# AI菜单栏设置
# ----------------------------------------------------------------------------------------------------------------------------
from aip import AipImageClassify
from aip import AipOcr

""" 你的 APPID AK SK """
APP_ID = '18179714'
API_KEY = 'tByFXaaMUHa00UaF9LEjIGHG'
SECRET_KEY = 'q17wmKFbX4xYmwGqbGfGGsNYcB492x32'
from io import BytesIO
import io

def content_find_file():
    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list)-1]

    global APP_ID, API_KEY, SECRET_KEY
    client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

    # print(type(show_img)) #PIL Image
    # 1 保存  2 读文件流
    show_img.save('baidu_ai_temp.jpg')
    baidu_ai_temp = open('baidu_ai_temp.jpg', 'rb').read()
    r = client.advancedGeneral(baidu_ai_temp)  # jpg png图片的二进制

    # print(r)
    t = r['result'][0]['keyword']
    # print(t)
    messagebox.showinfo(title='信息', message=('这可能是' + t + '！'))  # return ok

def word_fing_file():
    global APP_ID, API_KEY, SECRET_KEY
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    img_list = globalvar.get_value('img_list_global')
    show_img = img_list[len(img_list) - 1]

    show_img.save('baidu_ai_temp.jpg')
    baidu_ai_temp = open('baidu_ai_temp.jpg', 'rb').read()
    r = client.basicGeneral(baidu_ai_temp)
    txt = ''
    for x in r['words_result']:
        txt = txt + x['words'] + '\n'
    messagebox.showinfo(title='信息', message=txt)

def add_ai_bar(ai_):
    ai_.add_command(label='图像内容识别', command=lambda: content_find_file())
    ai_.add_command(label='图像文字识别', command=lambda: word_fing_file())

# ----------------------------------------------------------------------------------------------------------------------------
# 帮助菜单栏设置
# ----------------------------------------------------------------------------------------------------------------------------
from selenium import webdriver
import os
import requests
from bs4 import BeautifulSoup

# 搜索图片
def img_find_file():
    hd = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36'}
    bw = webdriver.Chrome()

    word = g.enterbox(msg='请输入你要查询的图片特征（请保证网络畅通，否则将无法正常下载。默认下载3张，如有需要请自行下载更多！）', title='下载图片', default='动物')
    bw.get('https://www.quanjing.com/search.aspx?q={}#{}|1|100|1|2||||||'.format(word, word))
    bw.implicitly_wait(5)

    ids = bw.find_elements_by_css_selector('.gallery_list img')
    song_names = [x.get_attribute('src') for x in ids]
    imgs = song_names[:3]

    path = os.path.join(os.path.join(os.path.expanduser("~"), 'Desktop'), 'my_save_imgs')
    if os.path.exists(path) == False:
        os.mkdir(path)

    index = 1
    for x in imgs:
        r = requests.get(x, headers=hd)
        f1 = open(path + '\\' + str(index) + '.jpg', 'wb')
        f1.write(r.content)
        f1.close()
        index = index + 1

    messagebox.showinfo(title='信息', message=('下载完成！'))  # return ok

def add_help_bar(help_):
    help_.add_command(label='搜索图片', command=lambda: img_find_file())


# ----------------------------------------------------------------------------------------------------------------------------
# 菜单项
# ----------------------------------------------------------------------------------------------------------------------------
def menu_sel_area(root):
    bar = tk.Menu(root)

    # 创建元素控件
    file_ = tk.Menu(bar, tearoff=0)
    edit_ = tk.Menu(bar, tearoff=0)
    image_ = tk.Menu(bar, tearoff=0)
    filter_ = tk.Menu(bar, tearoff=0)
    ai_ = tk.Menu(bar, tearoff=0)
    help_ = tk.Menu(bar, tearoff=0)

    # 添加顶层总菜单
    bar.add_cascade(label='文件', menu=file_)
    bar.add_cascade(label='编辑', menu=edit_)
    bar.add_cascade(label='图像', menu=image_)
    bar.add_cascade(label='滤镜', menu=filter_)
    bar.add_cascade(label='AI', menu=ai_)
    bar.add_cascade(label='帮助', menu=help_)

    # 添加 文件 菜单项
    add_file_bar(file_)

    # 添加 编辑 菜单项
    add_edit_bar(edit_)

    # 添加图像菜单栏
    add_image_bar(image_)

    # 添加滤镜菜单栏
    add_filter_bar(filter_)

    # 添加AI菜单栏
    add_ai_bar(ai_)

    # 添加帮助菜单栏
    add_help_bar(help_)

    return bar