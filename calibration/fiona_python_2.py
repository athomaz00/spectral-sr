# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:32:33 2015

@author: Andre
"""
import pylab as pl
import numpy as np
#import gaussfitter
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sp
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter


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

image = pl.imread('680-1.png') #read image file


size_X,size_Y = np.shape(image)

X,Y = np.mgrid[0:size_X,0:size_Y]

ig = (10.0, 10.0 , 10.0, 12.0, 2.0, 2.0, 0.0)
inpars, pcov = sp.curve_fit(twoD_Gaussian,(X,Y), image.ravel(), p0=ig)
fit3 = twoD_Gaussian((X, Y), *inpars).reshape(len(X),len(Y))


#Plot image
fig1 = pl.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X,Y,image,rstride=1, cstride=1, cmap="hsv", linewidth=0, antialiased=False, alpha=0.5)
ax1.plot_wireframe(X,Y,fit3, color="black", linewidth=3)
ax1.set_xlabel("Pixels")
ax1.set_ylabel("Pixels")
ax1.set_zlabel("Intensity")


# Getting center and standard deviation in x and y
xCenter = inpars[2] #*PixSize      # Center of x in nanometer
yCenter = inpars[3] #*PixSize       # Center of y in nanometer


# Obtaining b
residual = image-fit3;                  # Extract residual
fig2 = pl.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X,Y,residual,rstride=1, cstride=1, cmap="hsv", linewidth=1, antialiased=False)
#residual = residual<0
b = np.sqrt(sum(np.ravel(residual)*np.ravel(residual))/(len(np.ravel(residual))-1))










