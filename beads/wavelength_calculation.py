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
from scipy.interpolate import interp1d, spline


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

image = Image.open(imagesFiles[0])
im_red = Image.open(imagesFiles[1])
im_green = Image.open(imagesFiles[2])

#Mean factor for tresholding (everything less than treshold is zero)
mean_factor = 1.7
image = np.array(image)
#img_mean_base = np.mean(image)
#
#msk = image<(img_mean_base*mean_factor)
#image[msk] = 0.0

#Check treshold, otherwise it may find more than one peak
thr_base = np.max(image)-30*np.std(image)
peaks_base = peak_local_max(image, min_distance=10, threshold_abs=thr_base)
peaks_base[:,0], peaks_base[:,1] = peaks_base[:,1], peaks_base[:,0].copy()


delete_base = []
for i,points in enumerate(peaks_base):
    if points[1]-30<0:
        delete_base.append(i)
    if 512-points[0]<15:
        delete_base.append(i)
peaks_base = np.delete(peaks_base, delete_base, 0)
print(str(peaks_base.shape[0]) + ' 680 peaks')
#plt.figure()
#plt.imshow(image)
#plt.grid('off')
#plt.scatter(peaks_base[:,0], peaks_base[:,1], facecolor='none', edgecolor="white")

##############################################################################
im_red = np.array(im_red)
#mean_factor = 1.7
#img_mean_base = np.mean(im_red)

#msk = im_red<(img_mean_base*mean_factor)
#im_red[msk] = 0.0

thr_red = np.max(im_red)-18*np.std(im_red)
peaks_red = peak_local_max(im_red, min_distance=10, threshold_abs=thr_red)
peaks_red[:,0], peaks_red[:,1] = peaks_red[:,1], peaks_red[:,0].copy()


delete_base = []
for i,points in enumerate(peaks_red):
    if points[1]-30<0:
        delete_base.append(i)
    if 512-points[0]<15:
        delete_base.append(i)
peaks_red = np.delete(peaks_red, delete_base, 0)

print(str(peaks_red.shape[0]) + ' red peaks')
#
##
##
#plt.figure(3)
#plt.imshow(im_red)
#plt.grid('off')
#plt.scatter(peaks_red[:,0], peaks_red[:,1], facecolor='none', edgecolor="white")

#############################################################################
im_green = np.array(im_green)
#mean_factor = 1.8
#img_mean_base = np.mean(im_green)

#msk = im_green<(img_mean_base*mean_factor)

#im_green[msk] = 0.0
thr_green = np.max(im_green)-16.5*np.std(im_green)
peaks_green = peak_local_max(im_green, min_distance=10, threshold_abs=thr_green)
peaks_green[:,0], peaks_green[:,1] = peaks_green[:,1], peaks_green[:,0].copy()

delete_base = []
for i,points in enumerate(peaks_green):
    if points[1]-30<0:
        delete_base.append(i)
    if 512-points[0]<15:
        delete_base.append(i)
peaks_green = np.delete(peaks_green, delete_base, 0)

print(str(peaks_green.shape[0]) + ' green peaks')
#
#plt.figure(4)
#plt.imshow(im_green)
#plt.grid('off')
#plt.scatter(peaks_green[:,0], peaks_green[:,1], facecolor='none', edgecolor="white")

###############################################################################



#plt.figure()
#
#plt.contourf(im_green,60, cmap='Greens')
#plt.contourf(im_red,60, cmap='Reds', alpha=0.6)
#plt.contourf(image,60, cmap='Greys', alpha=0.4)
#plt.grid('off')
#plt.gca().invert_yaxis() #invert y axis to match imageJ

#############################################
base_centers = np.empty((0,2))
base_centers_trans = np.empty((0,2))
sumBase = np.empty((21,peaks_base.shape[0]))

for i, peaks in enumerate(peaks_base):
    squareSize = 20
    #peaks = peaks_base
    
    X, Y = np.mgrid[peaks[0]-squareSize/2:peaks[0]+squareSize/2 +1,peaks[1]-squareSize/2:peaks[1]+squareSize/2 +1]
    
    imageMask =  image[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1] #to mask it correctly we have to invert X and Y here
    imageMaskcopy =  np.copy(image)[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1]
    sumBaseTemp = np.sum(imageMaskcopy, axis=1)
    sumBase[:,i] = sumBaseTemp
  
    
    ig = (np.min(image), np.max(image) , peaks[1], peaks[0], 2.0, 2.0, 0.0)
    inpars, pcov = sp.curve_fit(twoD_Gaussian,(Y,X), imageMask.ravel(), p0=ig, maxfev=20000)
    fit3 = twoD_Gaussian((X, Y), *inpars).reshape(len(X),len(Y))
    
    base_centers = np.append (base_centers, [[inpars[2], inpars[3]]], axis=0)
    base_centers_trans = np.append (base_centers_trans, [[inpars[2]-peaks[1]+10, inpars[3]-peaks[0]+10]], axis=0)
    
    #Plot image
    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(111, projection='3d')
    #ax1.plot_surface(X, Y, imageMask, rstride=1, cstride=1, cmap="hsv", linewidth=0, antialiased=False, alpha=0.5)
    #ax1.plot_wireframe(X, Y, fit3, color="black", linewidth=3, alpha=0.3)
    #ax1.set_xlabel("Pixels")
    #ax1.set_ylabel("Pixels")
    #ax1.set_zlabel("Intensity")
    # 
    #plt.show()
    
    
#squareSize = 16
#peaks = peaks_base
#
#X, Y = np.mgrid[peaks[6][0]-squareSize/2:peaks[6][0]+squareSize/2 +1,peaks[6][1]-squareSize/2:peaks[6][1]+squareSize/2 +1]
#
#imageMask =  image[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1] #to mask it correctly we have to invert X and Y here
#
#ig = (np.min(image), np.max(image) , peaks[6][0], peaks[6][1], 1.0, 1.0, 0.0)
#inpars, pcov = sp.curve_fit(twoD_Gaussian,(X,Y), imageMask.ravel(), p0=ig, maxfev=20000)
#fit3 = twoD_Gaussian((X, Y), *inpars).reshape(len(X),len(Y))
#
#plt.figure()
#plt.imshow(image)
#plt.grid('off')
#plt.scatter(base_centers[:,1], base_centers[:,0], facecolor='none', edgecolor="white")
#plt.scatter(peaks_base[:,0], peaks_base[:,1], facecolor='none', edgecolor="r")
    
    
#######################################################################################################
red_centers = np.empty((0,2))
red_centers_trans = np.empty((0,2))
sumRed = np.empty((21,peaks_base.shape[0]))

for i,peaks in enumerate(peaks_base):
    squareSizeX = 20
    squareSizeY = 20
    #peaks = peaks_base
    
    X, Y = np.mgrid[peaks[0]-squareSizeX/2:peaks[0]+squareSizeX/2 +1,peaks[1]-squareSizeY/2:peaks[1]+squareSizeY/2 +1]
    
    redMask =  im_red[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1] #to mask it correctly we have to invert X and Y here
    redMaskcopy =  np.copy(im_red)[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1]
    sumRedTemp = np.sum(redMaskcopy, axis=1)
    sumRed[:,i] = sumRedTemp
    
#    ig = (np.min(im_red), np.max(im_red) , peaks[1], peaks[0], 1.0, 1.0, 0.0)
#    inpars2, pcov = sp.curve_fit(twoD_Gaussian,(Y,X), redMask.ravel(), p0=ig, maxfev=20000)
#    fit3 = twoD_Gaussian((X, Y), *inpars2).reshape(len(X),len(Y))
#    
#    red_centers = np.append (red_centers, [[inpars2[2], inpars2[3]]], axis=0)
#    red_centers_trans = np.append (red_centers_trans, [[inpars2[2]-peaks[1]+10, inpars2[3]-peaks[0]+10]], axis=0)

    #Plot image
#    fig1 = plt.figure()
#    ax1 = fig1.add_subplot(111, projection='3d')
#    ax1.plot_surface(X, Y, redMask, rstride=1, cstride=1, cmap="hsv", linewidth=0, antialiased=False, alpha=0.5)
#    ax1.plot_wireframe(X, Y, fit3, color="black", linewidth=3, alpha=0.3)
#    ax1.set_xlabel("Pixels")
#    ax1.set_ylabel("Pixels")
#    ax1.set_zlabel("Intensity")
    # 


#squareSizeX = 26
#squareSizeY = 26
#peaks = peaks_base
#    
#X, Y = np.mgrid[peaks[5][0]-squareSizeX/2:peaks[5][0]+squareSizeX/2 +1,peaks[5][1]-squareSizeY/2:peaks[5][1]+squareSizeY/2 +1]
#
#redMask =  im_red[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1] #to mask it correctly we have to invert X and Y here
#
#ig = (np.min(image), np.max(image) , peaks[5][0], peaks[5][1], 1.0, 1.0, 0.0)
#inpars, pcov = sp.curve_fit(twoD_Gaussian,(X,Y), redMask.ravel(), p0=ig, maxfev=20000)
#fit3 = twoD_Gaussian((X, Y), *inpars).reshape(len(X),len(Y))

#plt.figure()
#plt.imshow(im_red)
#plt.grid('off')
#plt.scatter(red_centers[-1,1], red_centers[-1,0], facecolor='none', edgecolor="black")

#######################################################################################################
green_centers = np.empty((0,2))
sumGreen = np.empty((41,peaks_base.shape[0]))

for i,peaks in enumerate(peaks_base):
    squareSizeX = 20
    squareSizeY = 40
#    dx = quareSize/2
#    dy = quareSize/2

    
    #peaks = peaks_base
    
    #peakDispY = peaks[1]-30
    
#    if peaks[0]>dx:
#        min_x = peaks[0] - dx
#    else:
#        min_x=0                                    #If point is close to left border
#    if (peaks[0]+dx)>=np.shape(image)[0]: #if point is close to right border
#        max_x = np.shape(image)[0]-1
#    else:
#        max_x = peaks[0] + dx
#    if peakDispY>dy:
#        min_y = peakDispY - dy
#    else:
#        min_y = 0
#    if (peakDispY+dy)>=np.shape(image)[1]:
#        max_y = np.shape(image)[1]-1
#    else:
#        max_y = peakDispY + dy
    
    X, Y = np.mgrid[peaks[0]-squareSizeX/2:peaks[0]+squareSizeX/2 +1,peaks[1]-squareSizeY/2:peaks[1]+squareSizeY/2 +1]
    
    greenMask = im_green[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1] #to mask it correctly we have to invert X and Y here
    
    greenMaskcopy =  np.copy(im_green)[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1]
    sumGreenTemp = np.sum(greenMaskcopy, axis=1)
    sumGreen[:,i] = sumGreenTemp
    
    
    
#    ig = (np.min(im_green), np.max(im_green) , 132, peakDispY, 2.0, 2.0, 0.0)
#    inpars, pcov = sp.curve_fit(twoD_Gaussian,(X,Y), greenMask.ravel(), p0=ig, maxfev=20000)
#    #fit3 = twoD_Gaussian((X, Y), *inpars).reshape(len(X),len(Y))
#    
#    green_centers = np.append (green_centers, [[inpars[2], inpars[3]]], axis=0)
#
#plt.figure()
#plt.imshow(im_green)
#plt.grid('off')
#plt.scatter(base_centers[3,0], base_centers[3,1], facecolor='none', edgecolor="white")

#########################################################################################################
coef = np.array([8e-5,0.0197,3.4768, 680])
p = np.poly1d(coef)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

spectralMeanTableRed = np.empty((np.shape(base_centers_trans)[0],1))
#spectralMeanTableGreen = np.empty((np.shape(base_centers_trans)[0]+20,2))

for i, centers in enumerate(base_centers_trans):
    pixels = np.arange(0,21,1)
    x_center = centers[0] 
    y_center = centers[1]
    pixels_x = pixels - x_center
    pixels_y = pixels - y_center
    pixel_disp = p(pixels_y)
    pixel_disp_new = np.linspace(pixel_disp.min(), pixel_disp.max(), 300)
   
    sumNormRed = (sumRed[:,i]) /np.max(sumRed[:,i]) -0.7
    sumNormGreen = (sumGreen[:,i]) /np.max(sumGreen[:,i]) -0.7
    sumNormBase = (sumBase[:,i])/np.max(sumBase[:,i]) 
    spectralMeanRed = np.sum(np.multiply(sumRed[:,i],pixel_disp))/np.sum(sumRed[:,i])
    #spectralMeanGreen = np.sum(np.multiply(sumGreen[:,i],pixel_disp-30))/np.sum(sumGreen[:,i])
    spectralMeanTableRed[i,0] = spectralMeanRed
    #spectralMeanTableGreen[i,1] = spectralMeanGreen
    plt.figure(1)
    plt.scatter([680.0],spectralMeanRed )
   # plt.scatter([580.0],spectralMeanGreen )
    plt.xticks(np.arange(560,720,10))
    plt.ylim(570,720)
#    
#    if i%1 ==  0:
    plt.figure(2)
#    #plt.plot(pixel_disp, sumNormRed, 'r')
#    plt.plot(pixel_disp, smooth(sumNormRed, 3))
    #plt.plot(pixel_disp, smooth(sumNormBase, 3), 'b')
    #plt.plot(pixel_disp, smooth(sumNormGreen, 3), 'g')
#    plt.xlim(655,720)
#    

plt.figure(1)
#plt.scatter([680.0], np.mean(spectralMeanTable), c='black', s=30)
plt.errorbar([680.0], np.mean(spectralMeanTableRed), yerr=np.std(spectralMeanTableRed), marker='o', markersize=3)
    



