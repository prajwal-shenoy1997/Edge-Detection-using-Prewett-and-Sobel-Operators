import cv2 as cv
import numpy as np
import math

path="/home/prajwal/prajwal/opencv/lena.jpg"
img=cv.imread(path,0)

cv.namedWindow("Image after Prewett Edge Detection")
cv.namedWindow("Image after Sobel Edge Detection")
cv.namedWindow("Input")


pre_xkernel=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
m=pre_xkernel.shape[0]
a=int(m/2)
pre_ykernel  =np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
n=pre_xkernel.shape[1]
b=int(n/2)


sob_xkernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
sob_ykernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

img1=np.zeros((img.shape[0]+2,img.shape[1]+2))
img2=np.zeros((img.shape[0]+2,img.shape[1]+2))
img3=np.zeros((img.shape[0]+2,img.shape[1]+2))

img1[1:img.shape[0]+1,1:img.shape[1]+1]=img.copy()   #Image with its border padded
img2[1:img.shape[0]+1,1:img.shape[1]+1]=img.copy()   #Image to be processed as Edge detected Sobel Image
img3[1:img.shape[0]+1,1:img.shape[1]+1]=img.copy()   #Image to be processed as Edge detected Prewett Image

for i in range(1,img.shape[0]+1):
    for j in range(1,img.shape[1]+1):
            pre_xsum=0
            pre_ysum=0
            sob_xsum=0
            sob_ysum=0
            for l in range(-a,a+1):
                for k in range(-b,b+1):
                    pre_xsum=pre_xsum+pre_xkernel[l+1][k+1]*img1[i+l][j+k]
                    pre_ysum=pre_ysum+pre_ykernel[l+1][k+1]*img1[i+l][j+k]

                    sob_xsum=sob_xsum+sob_xkernel[l+1][k+1]*img1[i+l][j+k]
                    sob_ysum=sob_ysum+sob_ykernel[l+1][k+1]*img1[i+l][j+k]


            img2[i][j]=math.sqrt(pre_xsum**2+pre_ysum**2)  #Edge magnitude
            img3[i][j]=math.sqrt(sob_xsum**2+sob_ysum**2)
            #img1[i][j]=abs(xsum)+abs(ysum)
            if img2[i][j]>255:
                img2[i][j]=255
            if img3[i][j]>255:
                img3[i][j]=255

cv.imwrite('img4.png',img2)
cv.imwrite('img5.png',img3)
img3=cv.imread("img4.png",0)
img4=cv.imread("img5.png",0)
cv.imshow('Input',img)
cv.imshow('Image after Sobel Edge Detection',img4)
cv.imshow('Image after Prewett Edge Detection',img3)
cv.waitKey(0)
cv.destroyWindow('Lena')
