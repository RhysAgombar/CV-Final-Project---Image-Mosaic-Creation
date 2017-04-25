import cv2
import numpy as np
from scipy import signal
import scipy as sp
import scipy.ndimage as nd
import scipy.ndimage.filters as ft
import os

class segment:
    x1, x2 = 0, 0
    y1, y2 = 0, 0
    
    def divide(self):
        seg1, seg2, seg3, seg4 = segment(), segment(), segment(), segment()
        seg1.x1 = self.x1
        seg1.x2 = self.x1 + (self.x2 - self.x1)/2
        seg1.y1 = self.y1
        seg1.y2 = self.y1 + (self.y2 - self.y1)/2
     
        seg2.x1 = self.x1 + (self.x2 - self.x1)/2
        seg2.x2 = self.x1 + (self.x2 - self.x1)
        seg2.y1 = self.y1 + (self.y2 - self.y1)/2
        seg2.y2 = self.y1 + (self.y2 - self.y1)
        
        seg3.x1 = self.x1 + (self.x2 - self.x1)/2
        seg3.x2 = self.x1 + (self.x2 - self.x1)
        seg3.y1 = self.y1
        seg3.y2 = self.y1 + (self.y2 - self.y1)/2
        
        seg4.x1 = self.x1
        seg4.x2 = self.x1 + (self.x2 - self.x1)/2
        seg4.y1 = self.y1 + (self.y2 - self.y1)/2
        seg4.y2 = self.y1 + (self.y2 - self.y1)
        
        return [seg1, seg2, seg3, seg4]
        
def subdivideImage(img, segList):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    
    kernel = np.zeros((img.shape[0],img.shape[1]))

    kernel[crow][ccol] = 4
    kernel[crow - 1][ccol] = -1
    kernel[crow + 1][ccol] = -1
    kernel[crow][ccol - 1] = -1
    kernel[crow][ccol + 1] = -1
    
    f2 = np.fft.fft2(kernel)
    shiftedKernel = np.fft.fftshift(f2)
    
    fshift = fshift * shiftedKernel
    
    img_back = np.fft.ifft2(fshift)
    img_back = np.fft.ifftshift(img_back)
    img_back = np.abs(img_back)
    
    img_back = sp.stats.threshold(img_back, threshmin=70, newval=0.0)
    img_back = sp.stats.threshold(img_back, threshmax=1, newval=255.0)
    
    img_back[0,:] = 0
    img_back[:,0] = 0
    img_back[img_back.shape[0]-1,:] = 0
    img_back[:,img_back.shape[1]-1] = 0
    
    #cv2.imshow('Thresholded',cv2.convertScaleAbs(img_back))
    
    nl = []
    
    for i in range(0, len(segList)):
        seg = segList[i]
        avg = np.sum(img_back[seg.x1:seg.x2,seg.y1:seg.y2]) / ((seg.x2 - seg.x1) * (seg.y2 - seg.y1))
        if (avg > 0):
            sl = seg.divide()
            for j in range(0, len(sl)):
                nl.append(sl[j])
        else:
            nl.append(seg)
            
    return nl
        

def gauss1d(sigma,mean,x,ord):    
    x=x-mean
    num=np.multiply(x,x)
    variance=np.power(sigma,2)
    denom=2*variance  
    g = np.zeros(num.shape)
    
    for i in range(0, len(num)):
        g[i] = np.exp(-num[i]/denom)/np.power((np.pi*denom),0.5)

    if (ord == 1):
        g = np.multiply(-g,(x/variance))
    elif (ord == 2):
        g = np.multiply(g, ((num-variance)/(np.power(variance,2))))
    
    return g

def makefilter(scale,phasex,phasey,pts,sup):
    gx=gauss1d(3*scale,0,pts[0,:],phasex)
    gy=gauss1d(scale,0,pts[1,:],phasey)
    
    gc = np.multiply(gx,gy)
    gc = np.reshape(gc,(sup,sup))
    
    gcMin, gcMax = gc.min(), gc.max()
    f = (gc - gcMin)/(gcMax - gcMin)    
    return f
    
def makeLMfilters():
    SUP = 49
    SCALEX = np.array([np.power(np.sqrt(2),1), np.power(np.sqrt(2),2), np.power(np.sqrt(2),3)])
    NORIENT = 6           
    NROTINV = 12
    NBAR = len(SCALEX)*NORIENT
    NEDGE = len(SCALEX)*NORIENT
    NF = NBAR+NEDGE+NROTINV
    F = np.zeros([SUP,SUP,NF])
    hsup = (SUP-1)/2
    
    ind = np.linspace(-hsup,hsup, SUP)
    X,Y = np.meshgrid(ind, ind)    
    orgpts = np.array([X.ravel(), Y.ravel()])

    count=0
    for scale in range(0, len(SCALEX)):
        for orient in range(0, NORIENT):
            angle = np.pi*orient/NORIENT
            c = np.cos(angle)
            s = np.sin(angle)
            
            test = np.array([[c, -s], [s, c]])
            
            rotpts =  np.dot(test, orgpts)              
            
            test = makefilter(SCALEX[scale],0,1,rotpts,SUP)
            
            F[:,:,count] = makefilter(SCALEX[scale],0,1,rotpts,SUP)
            F[:,:,count+NEDGE] = makefilter(SCALEX[scale],0,2,rotpts,SUP)
            count=count+1
    
    count=NBAR+NEDGE
    j = 0
    SCALES=np.array([np.power(np.sqrt(2),1), np.power(np.sqrt(2),2), np.power(np.sqrt(2),3), np.power(np.sqrt(2),4)])
    for i in range(0,len(SCALES)):
        
        gc = np.reshape(signal.gaussian(SUP, SCALES[i]),(49,1)) * np.transpose(np.reshape(signal.gaussian(SUP, SCALES[i]),(49,1)))
        gcMin, gcMax = gc.min(), gc.max()
        f = (gc - gcMin)/(gcMax - gcMin)    
        F[:,:,count] = f
        
        gc = np.reshape(signal.gaussian(SUP, SCALES[i]),(49,1)) * np.transpose(np.reshape(signal.gaussian(SUP, SCALES[i]),(49,1)))
        gcMin, gcMax = gc.min(), gc.max()
        f = (gc - gcMin)/(gcMax - gcMin)   

        gc = nd.gaussian_laplace(f, SCALES[j])
        gcMin, gcMax = gc.min(), gc.max()
        f = (gc - gcMin)/(gcMax - gcMin)   
        test = f

        F[:,:,count+1] = test
        
        gc = np.reshape(signal.gaussian(SUP, SCALES[i]),(49,1)) * np.transpose(np.reshape(signal.gaussian(SUP, SCALES[i]),(49,1)))
        gcMin, gcMax = gc.min(), gc.max()
        f = (gc - gcMin)/(gcMax - gcMin)  

        gc = nd.gaussian_laplace(f, 3*SCALES[j])
        gcMin, gcMax = gc.min(), gc.max()
        f = (gc - gcMin)/(gcMax - gcMin)   
        test = f
        
        F[:,:,count+2] = test
    
        count=count+3
        j = j + 1
    
    return F
    
#***********************************
def colourVoting(img):
    colourVote = np.zeros((17,17,17))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            vx = np.round(img[i][j][0]/17)
            vy = np.round(img[i][j][1]/17)
            vz = np.round(img[i][j][2]/17)
            
            colourVote[vx][vy][vz] += 1
    
    c1d = colourVote.flatten()
    indx = c1d.argsort()[-5:]
    
    cx,cy,cz = np.unravel_index(indx, colourVote.shape)
    
    hold = np.zeros(3)
    for x,y,z in zip(cx,cy,cz):
        hold[0] = x
        hold[1] = y
        hold[2] = z
        
    return hold
#***********************************
def filterImage(Filters, image):
    filterDisp = np.zeros((49*12,49*4))
    res = np.zeros((49*12,49*4))
    
    j,k = 0,0
    for i in range(0, F.shape[2]):
        filterDisp[j * 49:(j+1)*49, k*49:(k+1)*49] = np.round(F[:,:,i],3)
        j = j + 1
        if (j > 11):
            j = 0
            k = k + 1
    
    resM = np.zeros((49,49))
    resOut = np.zeros(F.shape[2])
    for i in range(0, F.shape[2]):
        resM = signal.convolve2d(image, F[:,:,i].astype('uint8'), mode='same', boundary='fill')
        resOut[i] = np.mean(resM)    
    
    return resOut

#*****************************************************************************
img = cv2.imread('t2.jpg')
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fdir = 'TestImages'
scale = 50

print "Start"

F = makeLMfilters()

print "Filter Made"

#***********
x = img.shape[0] - (img.shape[0] % scale)
xn = (img.shape[0] % scale) / scale

if (xn < 0.5):
    xn = 0
else:
    xn = 1
x += (scale * xn)

y = img.shape[1] - (img.shape[1] % scale)
yn = (img.shape[1] % scale) / scale

if (yn < 0.5):
    yn = 0
else:
    yn = 1
y += (scale * yn)
#***********

img = cv2.resize(img,(y,x)).astype('uint8')
imgGrey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img2 = img.copy()
img3 = img.copy()

numDiv = 2

imageList = []
colourList = []
textureList = []
combList = []

for i in range (0, numDiv):
    imageList.append([])
    colourList.append([])
    textureList.append([])
    combList.append([])

'''
for filename in os.listdir('TestImages'):
     for i in range (0, numDiv):
        print filename, i
        sub = 2*i
        if (sub < 1):
            sub = 1
        dsImg = cv2.resize(cv2.imread('TestImages/' + filename),(scale/(sub),scale/(sub)))
        imageList[i].append(dsImg)
        r = np.mean(dsImg[:,:,0]).astype('uint8')
        g = np.mean(dsImg[:,:,1]).astype('uint8')
        b = np.mean(dsImg[:,:,2]).astype('uint8')
        colourList[i].append(np.array([r,g,b]))#colourVoting(dsImg))
        fltr = filterImage(F, cv2.cvtColor(dsImg, cv2.COLOR_RGB2GRAY))
        textureList[i].append(fltr)
'''     
        
if os.path.isfile('textureValuesD.npy') != True:
    for filename in os.listdir(fdir):
        for i in range (0, numDiv):
            print filename
            sub = 2*i
            if (sub < 1):
                sub = 1
            dsImg = cv2.resize(cv2.imread('TestImages/' + filename),(scale/(sub),scale/(sub)))
            imageList[i].append(dsImg)
            r = np.mean(dsImg[:,:,0]).astype('uint8')
            g = np.mean(dsImg[:,:,1]).astype('uint8')
            b = np.mean(dsImg[:,:,2]).astype('uint8')
            colourList[i].append(np.array([r,g,b]))
            fltr = filterImage(F, cv2.cvtColor(dsImg, cv2.COLOR_RGB2GRAY))
            textureList[i].append(fltr)
                    
    np.save('textureValuesD.npy', textureList)
    np.save('colourValuesD.npy', colourList)
    np.save('imgListD.npy', imageList)
    
else:
     colourList = np.load('colourValuesD.npy')  
     textureList = np.load('textureValuesD.npy')  
     imageList = np.load('imgListD.npy')  
        
        

segmentList = []
for i in range(0, (x/scale)):
    for j in range(0, (y/scale)):
        seg = segment()
        seg.x1 = i*scale
        seg.x2 = (i+1)*scale
        seg.y1 = j*scale
        seg.y2 = (j+1)*scale
        segmentList.append(seg)

for i in range(0, numDiv):
    segmentList = subdivideImage(imgGrey, segmentList)

for i in range(0, len(segmentList)):
    seg = segmentList[i]
    imgSeg = img[seg.x1:seg.x2,seg.y1:seg.y2,:]
    imgSegGrey = imgGrey[seg.x1:seg.x2,seg.y1:seg.y2]
    colour = [np.mean(imgSeg[:,:,0]).astype('uint8'),np.mean(imgSeg[:,:,1]).astype('uint8'),np.mean(imgSeg[:,:,2]).astype('uint8')]
    fltr = filterImage(F, imgSegGrey)
    cDist, fDist = 1e9, 1e9
    cHolder, fHolder = 0, 0
    
    lv = np.round((scale /(float) (seg.x2 - seg.x1))).astype('uint8')
    if lv > 1:
        count = 0
        while (lv > 2):
            lv = np.floor((lv /(float) (2))).astype('uint8')
            count += 1
        
        print count
        lv = count
    
    for k in range(0, len(colourList[lv])):                          
        clHold = colourList[lv][k][0].astype('int64')
        cDistH = np.power((colour[0] - clHold),2).astype('float64')
        clHold = colourList[lv][k][1].astype('int64')
        cDistH += np.power((colour[1] - clHold),2).astype('float64')
        clHold = colourList[lv][k][2].astype('int64')
        cDistH += np.power((colour[2] - clHold),2).astype('float64')
        cDistH = np.sqrt(cDistH)
        
        fltrH = textureList[lv][k] #filterImage(F, cv2.cvtColor(imageList[k], cv2.COLOR_RGB2GRAY))
        fDistH = 0
        
        for l in range(0, len(fltrH)):
            fDistH += np.power((fltr[l] - fltrH[l]),2).astype('float64')
        fDistH = np.sqrt(fDistH)
        
        print i, j, k
        
        if ((cDist * 0.7 + fDist * 0.3) > (cDistH * 0.7 + fDistH * 0.3)):
            cDist = cDistH
            fDist = fDistH
            holder = k
        
    
    ILx = seg.x2 - seg.x1
    ILy = seg.y2 - seg.y1
    
    img2[seg.x1:seg.x2,seg.y1:seg.y2,:] = cv2.resize(imageList[lv][holder],(ILy,ILx))
    #img3[seg.x1:seg.x2,seg.y1:seg.y2,:] = colour

cv2.imshow("Original", img)
cv2.imshow("Mosaic", img2)
