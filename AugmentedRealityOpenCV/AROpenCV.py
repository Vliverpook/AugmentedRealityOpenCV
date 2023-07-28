import cv2
import numpy as np

cap=cv2.VideoCapture(0)
imgTarget=cv2.imread('TargetImage.jpg')
myVid=cv2.VideoCapture('video.mp4')

detection=False
frameCounter=0

#读取视频的第一帧，修改尺寸为目标图片大小
success, imgVideo=myVid.read()
hT,wT,cT=imgTarget.shape#获取目标图片大小


#orb特征匹配
orb=cv2.ORB_create(nfeatures=1000)
#获取目标图片的特征值
kp1, des1=orb.detectAndCompute(imgTarget,None)
# imgTarget=cv2.drawKeypoints(imgTarget,kp1,None)



while True:
    success, imgWebcam=cap.read()
    # imgWebcam=cv2.imread('1.jpg')
    # hW,wW,cW=imgWebcam.shape
    # imgWebcam=cv2.resize(imgWebcam,(int(wW/6),int(hW/6)))
    imgAug=imgWebcam.copy()
    #获取摄像头图片的特征值
    kp2, des2=orb.detectAndCompute(imgWebcam, None)
    # imgWebcam=cv2.drawKeypoints(imgWebcam,kp2,None)

    #处理视频播放逻辑，未检测到图片时重置播放进度，播放完之后重新播放
    if detection==False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter=0
    else:
        if frameCounter==myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo=myVid.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))

    #描述子的比较
    bf=cv2.BFMatcher()
    matches=bf.knnMatch(des1,des2,k=2)
    good=[]
    for m,n in matches:
        if m.distance<0.75 *n.distance:
            good.append(m)
    print(len(good))

    imgFeatures=cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None,flags=2)

    #通过特征点的对应关系寻找变换矩阵
    if len(good)>20:
        detection=True
        srcPts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        matrix, mask=cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
        print(matrix)

        #给出原图角点位置通过变换矩阵求新图中在角点的位置
        pts=np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        dst=cv2.perspectiveTransform(pts,matrix)
        img2=cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)

        #对视频进行透视变换
        imgWarp=cv2.warpPerspective(imgVideo,matrix,(imgWebcam.shape[1],imgWebcam.shape[0]))

        #创建遮罩，填充白色并反转颜色
        maskNew=np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
        maskInv=cv2.bitwise_not(maskNew)

        #添加遮罩到图片中
        imgAug=cv2.bitwise_and(imgAug,imgAug,mask=maskInv)

        #添加视频到元图
        imgAug=cv2.bitwise_or(imgWarp,imgAug)

    cv2.imshow('maskNew',imgAug)
    # cv2.imshow('ImgWarp',imgWarp)
    # cv2.imshow('ImgFeatures',imgFeatures)
    # cv2.imshow('ImgVideo',imgVideo)
    # cv2.imshow('ImgWebcam',imgWebcam)
    cv2.waitKey(1)
    frameCounter+=1