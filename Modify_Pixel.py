import os
import cv2
import random
import numpy as np
from PIL import Image

def FileVal(path):
    listdir = os.listdir(path)
    listone = os.path.join(path, listdir[3])
    listtwo = os.listdir(listone)
    listthree = os.path.join(listone, listtwo[0])
    listfour = os.listdir(listthree)##索引到训练集train里面文件

    list = os.path.join(listthree, listfour[2])
    raw_file = open(list,'r')
    raw_data = raw_file.readlines()
    train_label = []
    for i in raw_data:
        line = i.strip().split('\n')
        train_label.append(line[0] + '.jpg')
    a = len(train_label)
    num = random.sample(train_label, 230)#随机的对指定的图像进行像素操作
    return num


def ChangePixel(path):
    listdir01 = os.listdir(path)
    listdir02 = os.path.join(path, listdir01[0])
    image = os.listdir(listdir02)
    num = FileVal(path)
    count = 0
    for p in num:
        index = os.path.join(listdir02, p)
        img = cv2.imread(index)
        data = np.array(img)
        #改变B通道的像素
        for i in range(data[:,:,0].shape[0]):
            for j in range(data[:,:,0].shape[1]):
                if data[:,:,0][i][j] > 155:
                    data[:,:,0][i][j] = 255
        else:
            data[:,:,0][i][j] = 0
        #改变G通道的像素
        for i in range(data[:,:,1].shape[0]):
            for j in range(data[:,:,1].shape[1]):
                if data[:,:,1][i][j] > 155:
                    data[:,:,1][i][j] = 255
        else:
            data[:,:,1][i][j] = 0
        #改变R通道的像素
        for i in range(data[:,:,2].shape[0]):
            for j in range(data[:,:,2].shape[1]):
                if data[:,:,2][i][j] > 155:
                    data[:,:,2][i][j] = 255
        else:
            data[:,:,2][i][j] = 0

        figure = Image.fromarray(data)
        figure.save(index)
        # figure.save(p)
        count = count + 1
    return count

if __name__ == '__main__':
    #path = '/home/a401/Documents/SSD-COCO/VOCdevkit/VOC2007'
    path = '/home/a401/Documents/SSD-COCO/VOCdevkit/VOC2012'
    # a = FileVal(path)
    # print(a)
    count = ChangePixel(path)
    print(count)