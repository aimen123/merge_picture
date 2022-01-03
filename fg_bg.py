# liu- 图片合成

import os
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torch
import numpy as np
import cv2
import random

import torchvision.transforms as T


# 图片合并
def merge(fgimg, bgimg, bglabel, names, labimg, Tag):
    

    # 读入前景图
    foreground = cv2.imread(fgimg)
    # 读入背景图
    background = cv2.imread(bgimg)

    w,h = foreground.shape[0], foreground.shape[1]   # 前景w,h
    w = int(w)
    h = int(h)
    
    # w_move、h_move是前景图相对于背景图（0,0）点上的移动
    w_move = 256            # w_move   y上移动
    if Tag < 1212:                                                   
        h_move = 4          # h_move   x上移动
    # 1212是我背景图个数，当一个背景合成2个前景的时候，将h_move = 4 改为204，防止重叠
    if Tag >= 1212:         
        h_move = 204
        if (h_move + h) >= 415:       # 防止x方向上越界，我的数据背景wh均为416
            h_move = 154
        if (h_move + h) >= 415:
            h_move = 104
        if (h_move + h) >= 415:
            h_move = 54
            w_move = 50
    if (w_move + w) >= 415:          # 防止y方向上越界，我的数据背景wh均为416
        w_move = 200
    if (w_move + w) >= 415:          
        w_move = 156
    if (w_move + w) >= 415:
        w_move = 106
    if (w_move + w) >= 415:
        w_move = 56
    #labimg = cv2.erode(labimg,None,iterations=1)      # 腐蚀，训练集不用加,使合成看起来更好看

    for i in range(w):
        for j in range(h):
             if labimg[i, j, 1] == 0:       # 语义分割前景=0代表背景                                               
             #if labimg[i, j, 1] != 1: 
                 background[i+w_move,j+h_move,:] = background[i+w_move,j+h_move,:]
             elif labimg[i, j, 1] != 15 and sum(foreground[i,j,:]) > 15 and names != 3 and names != 4 and names != 5:
                 background[i+w_move,j+h_move,:] = foreground[i,j, :]
             elif labimg[i, j, 1] != 15 and sum(foreground[i,j,:]) <= 15 and names != 3 and names != 4 and names != 5:
                 background[i+w_move,j+h_move,:] = background[i+w_move,j+h_move,:]
             else:                          # 前5行删掉就可以了，是我针对我的数据合成效果做的更改
                 background[i+w_move,j+h_move,:] = foreground[i,j, :]

    #cv2.imshow('1',background)
    #cv2.waitKey(0)
    # 合成前景的标注框加在背景txt中
    f = open(bglabel,'a+')            # 代码运行前要 复制一份labeltxt以防出错，因为a+是在原来的txt中更改
    lab = names + 23                  # 我在原23个类别中添加新类别
    x = (h/2 + h_move)/416            # w,h是语义分割出来的，即为前景标注框
    y = (w/2 + w_move)/416
    le = h/416
    wi = w/416
    data = str(lab) + " " + str(x) + " " + str(y) + " " + str(le) + " " + str(wi) + "\n"    # 加“\n”
    f.write(data)
    f.close()
    return background

# 前景图片
def data(i):
    # 前景的标签类别
    name = ["person","cat","dog","fen","huichong","jichen"]
    # 前景的图片路径和语义分割标签路径
    fg_path = "/home/numen/Desktop/seg/fenge/" + name[i] + "/val/"
    lab_path = "/home/numen/Desktop/seg/fenge/" + name[i] + "/val_label/"
    # 前景图片列表
    fg_list = os.listdir(fg_path)
    return (fg_list,fg_path,lab_path)


def main():                                  
    # 背景图片路径和列表
    bgimg_path = "/home/numen/Desktop/train-val/images/val/"
    bgimg_list = os.listdir(bgimg_path)
    # 保存的图片路径
    save_path = '/home/numen/Desktop/seg/cc/'
    # 背景图片的txt标签
    bglab_path = "/home/numen/Desktop/train-val/labels/val/"
    bglab_list = os.listdir(bglab_path)

    
     
    i = 0   # 合成图片的个数
    tag = 0 # 标志
    num = 1212 # 这个1212是我背景图片的个数，num是一个阈值，如果大于背景图片个数，循环，一张背景合成两张前景
    for bg in (bgimg_list + bgimg_list):
        if i >= num:      # 如果只想一张背景和一张前景，这两句可以删除，                                              
            bgimg_path = '/home/numen/Desktop/seg/cc/'                 
        img = bgimg_path + bg
        label = bglab_path + bg[:-4] + '.txt'
        # 随机从列表中取一张前景图像，在我的图片数据中，i是背景的序号                                    
        if i < 300:   
            names = 0   # 前景id
            person_list, path, lab_path = data(names)   # 前景图片列表、前景图片路径、前景图片语义分割标签
            random_fgimg = random.choice(person_list)   # 随机前景图片
            fgimg = path + random_fgimg                 # 随机图片的路径
            labimg = lab_path + random_fgimg            # 随机图片的语义分割路径
        elif i >= 300 and i < 750:
            names = 1
            cat_list, path, lab_path = data(names)
            random_fgimg = random.choice(cat_list)
            fgimg = path + random_fgimg
            labimg = lab_path + random_fgimg
        elif i >= 750 and i < 1200:
            names = 2
            dog_list, path, lab_path = data(names)
            random_fgimg = random.choice(dog_list)
            fgimg = path + random_fgimg
            labimg = lab_path + random_fgimg
        elif i >= 1200 and i < 1250:
            names = 3
            fen_list, path, lab_path = data(names)
            random_fgimg = random.choice(fen_list)
            fgimg = path + random_fgimg
            labimg = lab_path + random_fgimg
        elif i >= 1250 and i < 1400:
            names = 4
            huichong_list, path, lab_path = data(names)
            random_fgimg = random.choice(huichong_list)
            fgimg = path + random_fgimg
            labimg = lab_path + random_fgimg
        elif i >= 1400 and i < 1550:
            names = 5
            jichen_list, path, lab_path = data(names)
            random_fgimg = random.choice(jichen_list)
            fgimg = path + random_fgimg
            labimg = lab_path + random_fgimg
        else:                                           # 前景数量少，停止合成
            tag = 1
        if tag != 1:
            print('正在合成第%s张：%s合到%s......' % (i, random_fgimg, bg))    # 显示
            labimg = cv2.imread(labimg)                                       # 读取前景图片
            mergeImg = merge(fgimg, img, label, names, labimg, i)             # 合成，输入（前景图片路径、背景img、背景txt、前景id、前景语义分割label、背景序号）
        if tag == 1:
            print('save',i)
            img = cv2.imread(img)
            mergeImg = img
        #cv2.imshow('1',mergeImg)
        #cv2.waitKey(0)
        newPath = save_path + bg[:-4] + '.jpg'
        cv2.imwrite(newPath, mergeImg)
        i += 1
    
# 入口
if __name__ == "__main__":

    main()

