import cv2
import numpy as np
from scipy import signal
import scipy.ndimage as nd
import scipy.ndimage.filters as ft
import os
from scipy import spatial
import argparse

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
    #orgpts = np.transpose(orgpts)

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
    
    j,k = 0,0
    for i in range(0, Filters.shape[2]):
        filterDisp[j * 49:(j+1)*49, k*49:(k+1)*49] = np.round(Filters[:,:,i],3)
        j = j + 1
        if (j > 11):
            j = 0
            k = k + 1
    
    resM = np.zeros((49,49))
    resOut = np.zeros(Filters.shape[2])
    for i in range(0, Filters.shape[2]):
        resM = signal.convolve2d(image, Filters[:,:,i].astype('uint8'), mode='same', boundary='fill')
        resOut[i] = np.mean(resM)    
    
    return resOut
#*********************************************************************************************************
#*********************************************************************************************************
#*********************************************************************************************************

def createMosaic(image, folder):
    print "Start"

    F = makeLMfilters()
    
    print "Filter Made"
    
    img = cv2.imread(image)
    fdir = folder
    
    scale = 10
    
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
    
    
    imageList = []
    colourList = []
    textureList = []
    
    if os.path.isfile('textureValues.npy') != True:
        for filename in os.listdir(fdir):
            dsImg = cv2.resize(cv2.imread(fdir + '/' + filename),(scale,scale))
            imageList.append(dsImg)
            r = np.mean(dsImg[:,:,0]).astype('uint8')
            g = np.mean(dsImg[:,:,1]).astype('uint8')
            b = np.mean(dsImg[:,:,2]).astype('uint8')
            colourList.append(np.array([r,g,b]))#colourVoting(dsImg))
            fltr = filterImage(F, cv2.cvtColor(dsImg, cv2.COLOR_RGB2GRAY))
            textureList.append(fltr)
            print filename
            
        np.save('textureValues.npy', textureList)
        np.save('colourValues.npy', colourList)
        np.save('imgList.npy', imageList)
        
    else:
        colourList = np.load('colourValues.npy')  
        textureList = np.load('textureValues.npy')  
        imageList = np.load('imgList.npy')  
    
    combList = []
    
    for i in range(0, len(colourList)):
        combList.append(np.zeros(49))
        combList[i] = np.hstack((textureList[i], colourList[i]))
    CLtree = spatial.KDTree(combList)
    
    print "Loops Start"
    for i in range(0, (x/scale)):
        print i, "of", (x/scale)
        for j in range(0, (y/scale)):
            imgSeg = img[i*scale:(i+1)*scale,j*scale:(j+1)*scale,:]
            imgSegGrey = imgGrey[i*scale:(i+1)*scale,j*scale:(j+1)*scale]
            colour = [np.mean(imgSeg[:,:,0]).astype('uint8'),np.mean(imgSeg[:,:,1]).astype('uint8'),np.mean(imgSeg[:,:,2]).astype('uint8')]#colourVoting(test) * 15
            fltr = filterImage(F, imgSegGrey)
            cDist, fDist = 1e9, 1e9
            cHolder, fHolder = 0, 0
            
            combRes = np.zeros(49)
            combRes = np.hstack((fltr, colour))
            
            distances, indices = CLtree.query(combRes, k=1)        
            
            img2[i*scale:(i+1)*scale,j*scale:(j+1)*scale,:] = imageList[indices]
            img3[i*scale:(i+1)*scale,j*scale:(j+1)*scale,:] = colour
    
            
    
    cv2.imshow("Mosaic", img2)
    cv2.imshow("Colours", img3)
    cv2.imshow("Original", img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Final Project')
    parser.add_argument('imgfile1', help='Image file')
    parser.add_argument('folder', help='Folder for Image Mosaic Pieces')
    args = parser.parse_args()
    
    createMosaic(args.imgfile1, args.folder)