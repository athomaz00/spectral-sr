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


plt.close('all')
#####################################################################################
#2D Gaussian asymetric and rotated
#offset = background
#amplitude = Peak Intensity
#x0 = x point
#y0 = y point
#sigma_x = width in x direction
#sigma_y = width in y direction
#theta = rotation angle

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
# =============================================================================
# Images files to load and initial points to do the calibration    
# =============================================================================

imagesFiles = [ '488.tif',  '568.tif', '680.tif', '735.tif']


initialPoint = np.array([[153, 285], [154, 321], [156,351], [157,362]])

zeroPoint = 680 #point for 0 pixel displacement


#############################################

centerPos = []

for i,items in enumerate(imagesFiles):
    
    image = plt.imread(items) #read image file
    
    squareSize = 20
    
    xmin = initialPoint[i][0]-squareSize/2
    xmax = initialPoint[i][0]+squareSize/2
    ymin = initialPoint[i][1]-squareSize/2
    ymax = initialPoint[i][1]+squareSize/2
    
    x = np.arange(xmin, xmax,1)
    y = np.arange(ymin, ymax,1)
    
    X, Y = np.meshgrid(x,y) #note: meshgrid is needed to use an asymetric grid with plot_surface, I tried mgrid and it didn't work.
    
    
    imageMask =  image[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1] #to mask it correctly we have to invert X and Y here
    
    ig = (np.min(image), np.max(image) , initialPoint[i][1], initialPoint[i][0], 2.0, 2.0, 0.0)
    inpars, pcov = sp.curve_fit(twoD_Gaussian,(Y,X), imageMask.ravel(), p0=ig, maxfev=20000)
    fit3 = twoD_Gaussian((Y, X), *inpars).reshape(X.shape[0], Y.shape[1])
    
    
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
    xCenter = inpars[3]       # Center of x in nanometer
    yCenter = inpars[2]       # Center of y in nanometer
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

plt.figure(5) 
plt.plot(distanceCenter, centerPos[:,0], 'o')
plt.xlabel('Pixel Displacement')
plt.ylabel('Wavelength (nm)')
plt.title('Pixel Displacement by Wavelength')


z = np.polyfit(distanceCenter, centerPos[:,0], 3)
p = np.poly1d(z)
print(p)

xx = np.arange(np.min(distanceCenter)-2, np.max(distanceCenter)+2, 0.5)
yy = p(xx)
plt.plot(xx,yy, '--r')
#plt.show()






