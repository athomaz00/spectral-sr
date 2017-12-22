# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:32:33 2015

@author: Andre
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sp


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
    

image = plt.imread('580.tif') #read image file




initialPoint = [232, 252]


#############################################
squareSize = 16

#for some reason X,Y in the image is inverted check this out later
X, Y = np.mgrid[initialPoint[0]-squareSize/2:initialPoint[0]+squareSize/2 +1,initialPoint[1]-squareSize/2:initialPoint[1]+squareSize/2 +1]

imageMask =  image[int(np.min(Y)):int(np.max(Y))+1, int(np.min(X)):int(np.max(X))+1] #to mask it correctly we have to invert X and Y here

ig = (np.min(image), np.max(image) , initialPoint[0], initialPoint[1], 2.0, 2.0, 0.0)
inpars, pcov = sp.curve_fit(twoD_Gaussian,(X,Y), imageMask.ravel(), p0=ig, maxfev=20000)
fit3 = twoD_Gaussian((X, Y), *inpars).reshape(len(X),len(Y))


#Plot image
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X,Y,imageMask,rstride=1, cstride=1, cmap="hsv", linewidth=0, antialiased=False, alpha=0.5)
ax1.plot_wireframe(X,Y,fit3, color="black", linewidth=3, alpha=0.3)
ax1.set_xlabel("Pixels")
ax1.set_ylabel("Pixels")
ax1.set_zlabel("Intensity")


# Getting center and standard deviation in x and y
xCenter = inpars[2] #*PixSize      # Center of x in nanometer
yCenter = inpars[3] #*PixSize       # Center of y in nanometer
print('X pos: ' + str(xCenter))
print('Y pos: ' + str(yCenter))

# Obtaining b
#residual = image-fit3;                  # Extract residual
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111, projection='3d')
#ax2.plot_surface(X,Y,residual,rstride=1, cstride=1, cmap="hsv", linewidth=1, antialiased=False)
##residual = residual<0
#b = np.sqrt(sum(np.ravel(residual)*np.ravel(residual))/(len(np.ravel(residual))-1))










