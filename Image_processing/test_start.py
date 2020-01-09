import PIL
import cv2
import tkinter as tk
from tkinter.filedialog import *
from PIL import Image, ImageTk

import globalvar
import test_2

save_imgs = []

def set_value():
    # v = 6
    # globalvar.set_value('get_value', 6)

    img_test = r'C:\Users\DHW\Pictures\img\yz.jpg'
    show_img = Image.open(img_test)
    save_imgs.append(show_img)


    # print('test_start: ', show_img, type(show_img))
    globalvar.set_value('save_imgs', save_imgs)

if __name__ == '__main__':
    globalvar._init()
    set_value()
    test_2.get_val_test()

    # li = ['b', 'dd', 'sd', 'p']
    # li.pop(0)
    # print(li)
    # print(li.index('sd'))
    # print(li)