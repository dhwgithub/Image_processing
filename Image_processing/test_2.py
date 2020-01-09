import PIL
import cv2
import tkinter as tk
from tkinter.filedialog import *
from PIL import Image, ImageTk

import globalvar


def get_val_test():
    # v = globalvar.get_value('get_value')
    # print(v)

    '''
    从另一个文件获取图片
    :return:
    '''
    save_imgs = globalvar.get_value('save_imgs')
    print('test_2: ', save_imgs[0], type(save_imgs[0]))
    # show_img = Image.open(img_test)

    '''
    图片展示
    '''
    root = tk.Tk()
    root.title('图像处理软件test')
    root.geometry('500x500')

    fr2 = tk.Frame(root, width=500, bg='pink', bd=2, relief='ridge')
    fr2.propagate(False)  # 取消自适应

    show_img = ImageTk.PhotoImage(save_imgs[0])

    label = Label(fr2, image=show_img)
    label.place(x=0, y=0)

    fr2.pack(side='left', fill='y')

    root.mainloop()