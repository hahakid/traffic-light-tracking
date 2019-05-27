#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import os
import glob
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import time
ia.seed(int(float(time.time())/np.pi))


count=0 #总图片计数器

def getroi(im,coord):
    global count
    #print(count)
    if coord[2]>coord[0] and coord[3]>coord[1]:
        count += 1
        roi = im[coord[1]:coord[3],coord[0]:coord[2]]
        #print(count)
        return roi



def rerange(w,h,coord):
    if coord[0]<0:
        coord[0]=0
    if coord[1] < 0:
        coord[1] = 0
    if coord[2]>w:
        coord[2]=w
    if coord[3] > h:
        coord[3] = h
    return coord


#ground truth label format (xmid,ymid,w,h)
def getfromGT(source,destination,offset):

    dir=source #'../hui/'
    savepath=destination #'./hui/'
    imglist=glob.glob(dir+"*.jpg")
    for img in imglist:
        im=cv2.imread(img)
        label=img.replace('jpg','txt')
        f=open(label)
        labels=f.readlines()
        f.close()
        h,w,c=im.shape
        #print w,h,c
        #print labels
        if labels:
            for l in labels:
                l=l.strip().split(' ')
                #print img,l
                l[1] = float(l[1])
                l[2] = float(l[2])
                l[3] = float(l[3])
                l[4] = float(l[4])
                #xmin ymin xmax ymax
                coord=[int((l[1]-l[3]/2)*w),int((l[2]-l[4]/2)*h),int((l[1]+l[3]/2)*w),int((l[2]+l[4]/2)*h)]
                coord=rerange(w,h,coord)
                imroi=getroi(im,coord)
                cv2.imwrite(os.path.join(savepath,str(count).zfill(6)+'.jpg'),imroi)
                if offset!=0:
                    coord = [int((l[1] - l[3]*(1+offset) / 2) * w), int((l[2] - l[4]*(1+offset) / 2) * h), int((l[1] + l[3]*(1+offset) / 2) * w),
                             int((l[2] + l[4]*(1+offset) / 2) * h)]
                    coord = rerange(w, h, coord)
                    imroi = getroi(im, coord)
                    cv2.imwrite(os.path.join(savepath, str(count).zfill(6) + '.jpg'), imroi)

def augtest(path,outpath):
    #global count
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [iaa.Crop(percent=(0, 0.1)),
         iaa.Fliplr(0.5),
         sometimes(iaa.GaussianBlur(sigma=(0, 0.5))),
         sometimes(iaa.Affine(
             scale={"x": (1.01, 1.1), "y": (1.01, 1.1)},
             translate_percent={"x": (-0.05, 0.1), "y": (-0.05, 0.1)},
             rotate=(-15, 15),
             shear=(-5, 5),
             order=[0, 1],
             #cval=(0, 255),
             mode=ia.ALL
         )),
         #iaa.SomeOf((0, 5),[sometimes(iaa.Superpixels(p_replace=(0, 1.0),n_segments=(20, 200))),iaa.OneOf([iaa.GaussianBlur((0, 0.5)),iaa.AverageBlur(k=(2, 7)),iaa.MedianBlur(k=(3, 11)),]),],)
         ])

    imglist=glob.glob(path+"/*.jpg")
    #print(imglist)
    images=[]
    for i in imglist:
        #print(i)
        images.append(cv2.imread(i))
    images_aug = seq.augment_images(images)
    savepath=outpath #os.path.join(path,"aug")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    #print(savepath)
    name=int(len(imglist))+1
    for j in images_aug:
        p=os.path.join(savepath, str(name).zfill(6) + '.jpg')
        cv2.imwrite(p,j)
        name += 1

#逻辑上，应该只有测试集有;但是也可以考虑将训练集去生成对应的。
#对检测结果是否要offset？
def processmodelresult(resultfile,impath, modelout,thresh,offset):
    global count
    f=open(resultfile)
    rlist=f.readlines()
    f.close()
    #icount=0
    for i in rlist:
        i=i.split(' ')
        if float(i[1])>=thresh:
            #print('---',icount)
            #icount+=1
            name=i[0]+".jpg"
            imp=os.path.join(impath,name)
            im=cv2.imread(imp)
            h,w,c=im.shape
            coord=[int(float(i[2])),int(float(i[3])),int(float(i[4])),int(float(i[5]))]# xmin ymin xmax ymax
            coord=rerange(w,h,coord)
            imroi = getroi(im, coord)
            #print(os.path.join(modelout, str(count).zfill(6) + '.jpg'))
            cv2.imwrite(os.path.join(modelout, str(count).zfill(6) + '.jpg'), imroi)
            #如果加offset,最好不加，很多背景数据
            if offset:
                alpha=int((float(i[4])-float(i[2]))*0.5*offset)
                beta=int((float(i[3])-float(i[1]))*0.5*offset)
                coord[0] -= alpha  #xmin-a
                coord[1] -= beta #ymin-b
                coord[0] += alpha #xmax+a
                coord[0] += beta #ymax+b
                coord = rerange(w, h, coord)
                imroi = getroi(im, coord)
                #cv2.imwrite(os.path.join(modelout, str(count).zfill(6) + '.jpg'), imroi)
                #print(os.path.join(modelout, str(count).zfill(6) + '.jpg'))

def generatedata():
    offset = 0.2# GT 截取偏移，在x,y 要除0.5
    
    trainpath='./data/train/'
    testpath='./data/test/'
    
    #处理label生成的结果
    hui='../hui/'
    huiout='./hui'
    #getfromGT(hui,huiout,offset)
    #augtest(huiout)
    
    getfromGT(hui,trainpath,offset)
    augtest(huiout,trainpath)


    qu = '../qu/'
    quout = './qu'

    #getfromGT(qu, quout,offset)
    #augtest(quout)
    
    getfromGT(qu, trainpath,offset)
    augtest(quout,trainpath)
    

    #test = './test/'
    #testout = './testout'
    #offset=0.2
    #getfromGT(test, testout,offset)

    #augtest(testout)

    #处理模型生成的结果
    #mr_list=['mobilenetv1.txt','mobilenetv2.txt','resnet34.txt','vgg-16.txt','yolov2-tiny.txt','yolov3-tiny-3l.txt']
    mr=glob.glob('modelresult/*.txt')
    modelout='./modelout/'
    thresh=0.1
    for f in mr:
        #processmodelresult(f,qu,modelout,thresh,offset=0)
        processmodelresult(f,qu,testpath,thresh,offset=0)
    
    augtest(testpath,trainpath)
    print('Done')

generatedata()
