# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:32:33 2015

@author: Andre
"""
#import seaborn as sns
#sns.set()
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sp
from scipy.spatial import distance
from skimage.feature import peak_local_max
from PIL import Image


plt.close('all')
#####################################################################################
def twoD_Gaussian(x_y, offset, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = x_y
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

################################################################################### 
imagesFiles = [ '680.tif',  'red.tif', 'green.tif']

image = Image.open('680.tif')
image = image.transpose(Image.FLIP_TOP_BOTTOM)  # rotate and flip to change
image = image.transpose(Image.ROTATE_270)  
image = np.array(image)
#Mean factor for tresholding (everything less than treshold is zero)
mean_factor = 1.8
img_mean_base = np.mean(image)

msk = image<(img_mean_base*mean_factor)
image[msk] = 0.0

#Check treshold, otherwise it may find more than one peak
thr_base = np.max(image)-32*np.std(image)
peaks_base = peak_local_max(image, min_distance=5, threshold_abs=thr_base)

plt.figure()
plt.imshow(image)
plt.grid('off')
plt.scatter(peaks_base[:,1], peaks_base[:,0], facecolor='none', edgecolor="white")

im_red = plt.imread('red.tif')
im_green = plt.imread('green.tif')

mean_factor = 1.8
img_mean_base = np.mean(im_red)

msk = im_red<(img_mean_base*mean_factor)
im_red[msk] = 0.0

mean_factor = 1.8
img_mean_base = np.mean(im_green)

msk = im_green<(img_mean_base*mean_factor)

im_green[msk] = 0.0



plt.figure()

plt.contourf(im_green,60, cmap='Greens')
plt.contourf(im_red,60, cmap='Reds', alpha=0.5)
plt.contourf(image,60, cmap='Greys', alpha=0.5)
plt.grid('off')


#############################################

centerPos = []

for i,items in enumerate(imagesFiles):
    
    image = plt.imread(items) #read image file
    
    squareSize = 16
    
    X, Y = np.mgrid[initialPoint[i][0]-squareSize/2:initialPoint[i][0]+squareSize/2 +1,initialPoint[i][1]-squareSize/2:initialPoint[i][1]+squareSize/2 +1]
    
    imageMask =  image[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1] #to mask it correctly we have to invert X and Y here
    
    ig = (np.min(image), np.max(image) , initialPoint[i][0], initialPoint[i][1], 2.0, 2.0, 0.0)
    inpars, pcov = sp.curve_fit(twoD_Gaussian,(X,Y), imageMask.ravel(), p0=ig, maxfev=20000)
    fit3 = twoD_Gaussian((X, Y), *inpars).reshape(len(X),len(Y))
    
    
    #Plot image
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, imageMask, rstride=1, cstride=1, cmap="hsv", linewidth=0, antialiased=False, alpha=0.5)
    ax1.plot_wireframe(X, Y, fit3, color="black", linewidth=3, alpha=0.3)
    ax1.set_xlabel("Pixels")
    ax1.set_ylabel("Pixels")
    ax1.set_zlabel("Intensity")
    plt.title(items)
    plt.show()
    
       
    # Getting center and standard deviation in x and y
    xCenter = inpars[2] #*PixSize      # Center of x in nanometer
    yCenter = inpars[3] #*PixSize       # Center of y in nanometer
    print('==========================')
    print(items)
    print('X pos: ' + str(xCenter))
    print('Y pos: ' + str(yCenter))
    print('===========================')
    
    splitName = items.split('.tif')
    centerPos.append([int(splitName[0]), xCenter, yCenter])

centerPos = np.array(centerPos)

distancesAll = distance.cdist(centerPos[:,1:3], centerPos[:,1:3])

pixelRef_ind = np.where(centerPos[:,0] == zeroPoint)[0][0]

distanceCenter = distancesAll[pixelRef_ind, :]

distanceCenter[0:pixelRef_ind] = distanceCenter[0:pixelRef_ind]*-1

plt.figure() 
plt.plot(distanceCenter, centerPos[:,0], 'o')
plt.xlabel('Pixel Displacement')
plt.ylabel('Wavelength (nm)')
plt.title('Pixel Displacement by Wavelength')


z = np.polyfit(distanceCenter, centerPos[:,0], 3)
p = np.poly1d(z)

xx = np.arange(np.min(distanceCenter)-2, np.max(distanceCenter)+2, 0.5)
yy = p(xx)
plt.plot(xx,yy, '--r')
plt.show()






