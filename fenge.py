# liu- 取出前景图

import cv2
import os


def main():
    path = "/home/numen/Desktop/dog-val/images/"               # 图片路径
    path_label_ = "/home/numen/Desktop/dog-val/label/"     # 语义分割标签路径
    list = os.listdir(path)
    tag = 0
    for pg in list:
        print(tag, pg)
        path_img = path + pg                              # 前景图片路径
        path_label = path_label_ + pg[:-3] + 'png'        # 语义分割标签路径
        img = cv2.imread(path_label)                      # 语义分割图
        img_pic = cv2.imread(path_img)                    # 前景图
        x,y,w,h = [], [], img.shape[1], img.shape[0]      
        for i in range(w):
            for j in range(h):
                if img[j, i, 1] == 1:                     # 这个自己可以更改根据自己语义分割标签，一般0为背景
                    #img[j, i, 1] = 15               
                    x.append(i)
                    y.append(j)
                #else:
                    #img[j,i,1] = 0
        # 把所有需要分割图像的像素点进行排序，然后取x最小、最大，y最小、最大，再把图像取出来
        x.sort()   
        y.sort()
        xmin, ymin = x[0], y[0]
        xmax, ymax = x[-1], y[-1]
        cut_label = img[ymin:ymax, xmin:xmax]      # 从图片中抠出来的语义分割图像
        cut = img_pic[ymin:ymax, xmin:xmax]        # 如上，扣除的图像
        cut = cv2.resize(cut,(0,0),fx= 0.5,fy= 0.5)    # 这个步骤看自己，如果想缩小，最好是等比缩小，固定大小容易出问题
        cut_label = cv2.resize(cut_label,(0,0),fx= 0.5,fy= 0.5)
        #new_x = xmax-xmin
        #new_y = ymax-ymin
        #h_move = 100; w_move = 100
        #roi = background[h_move:h_move + new_x, w_move:w_move +new_y]
        #print(new_x,new_y)
        #cv2.imshow("1",cut_label)
        #cv2.waitKey(0)
        newPath = '/home/numen/Desktop/dog-val/new_images/' + pg         # change 
        cv2.imwrite(newPath, cut)
        newPath_label = '/home/numen/Desktop/dog-val/new_label/' + pg    # change 
        cv2.imwrite(newPath_label, cut_label)
        tag = tag + 1

if __name__ == "__main__":
    main()










