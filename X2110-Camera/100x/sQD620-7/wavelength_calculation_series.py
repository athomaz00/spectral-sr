# =============================================================================
# This code is a variation of the wavelength_calculation.py to calculate the 
# spectra in wavelength units for a series of images. 
# 
# Input: sequence image files with the different colors on imageFiles variable
#
# Output: xlsx files at the df (spectra) variable and int_df (intensity) variable
#
# coef variable is the 3rd polynomial coefcients from the prism calibration
#
# Code written by Andre Thomaz 05/02/2018
# =============================================================================


#import seaborn as sns
#sns.set()
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sp
from skimage.feature import peak_local_max
from scipy import sparse
import pandas as pd
from skimage import io



plt.close('all')
#########################################################################################

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



listToClean = [5]


    

imagesFiles = [ '605-1.tif',  'sQD620-1.tif']


image = io.imread(imagesFiles[0])
im_red = io.imread(imagesFiles[1])








#Check treshold, otherwise it may find more than one peak
thr_base = np.max(image[0,:, :])-50*np.std(image[0])
peaks_base = peak_local_max(image[0,:, :], min_distance=5, threshold_abs=thr_base)
peaks_base[:,0], peaks_base[:,1] = peaks_base[:,1], peaks_base[:,0].copy()
    
#Define the size of the box/rectangle to crop the image to find peaks
x_sides = 18
y_top = 14 #if one of the colors is too far away from 680nm rectangle
y_bot = 14
    
    
#if points are to close to the borders dont consider it 
delete_base = []
for j,points in enumerate(peaks_base):
    if points[1]-y_top<0:                             #if point is too close to the top
        delete_base.append(j)
    elif points[1] + y_bot >image[0].shape[0]:           #if point is too close to the bottom
        delete_base.append(j)
    if points[0] + 10 > image[0].shape[1]:               #if point is too close to the right
        delete_base.append(j)
    if points[0] - 10 < 0:                            #if point is too close to the left
        delete_base.append(j)
peaks_base = np.delete(peaks_base, delete_base, 0)
peaks_pf = peaks_base.shape[0]

print(str(peaks_base.shape[0]) + ' 680 peaks')

#plot image and detected peaks
plt.figure(1)
plt.imshow(image[0,:, :])
plt.grid('off')
plt.scatter(peaks_base[:,0], peaks_base[:,1], facecolor='none', edgecolor="white")

##############################################################################


#############################################################################


#Calculation of the centers detected by the peaks of local maxima

base_centers = np.empty((0,2))
base_centers_trans = np.empty((0,2)) #trans is in the coordinate system of the masked image below

#if a detected color is to far from the reference peak we need to increase the y box height
box = x_sides
box_y_top = y_top
box_y_bot = y_bot

rows_y = box_y_top + box_y_bot

sumBase = [] #empty matrix to hold the collapsed sum of images/spectra


#fitting of a Gaussian to each detected peak before:
for k in range(len(image)): 
    for i, peaks in enumerate(peaks_base):
             
        xmin = peaks[0] - box/2
        xmax = peaks[0] + box/2
        ymin = peaks[1] - box_y_top
        ymax = peaks[1] + box_y_bot
        
        x = np.arange(xmin, xmax,1)
        y = np.arange(ymin, ymax,1)
        
        Y, X = np.meshgrid(x,y) #note: meshgrid is needed to use an asymetric grid with plot_surface, I tried mgrid and it didn't work.
        
        imageMask =  image[k][int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1] #to mask it correctly we have to invert X and Y here
        imageMaskcopy =  np.copy(image[k])[int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1] #copy to do calculation in masked image coordinates
        
        
        
        sumBaseTemp = np.sum(imageMaskcopy, axis=1) #collapse the x direction to get spectra along y direction
        sumBase.append(sumBaseTemp)
        

      
        ig = (np.min(imageMask), np.max(imageMask) , peaks[1]+0.5, peaks[0]+0.5, 2.0, 2.0, 0.0) #0.5 used to force fit because peaks were to close to the fit
        inpars, pcov = sp.curve_fit(twoD_Gaussian,(X,Y), imageMask.ravel(), p0=ig, maxfev=20000)
        fit3 = twoD_Gaussian((Y, X), *inpars).reshape(X.shape[0], Y.shape[1])
        
        base_centers = np.append (base_centers, [[inpars[3], inpars[2]]], axis=0) #invert order of parameters 
        base_centers_trans = np.append (base_centers_trans, [[inpars[3]-xmin, inpars[2]-ymin]], axis=0) #parameters minx ymin and xmin to translate 
                                                                                                        #to the coordinate system of the masked image
         #Plot masked image and fitting
        #fig = plt.figure(i+10)
        #ax1 = fig.gca(projection='3d')
        #ax1.plot_surface(Y, X, imageMask, cmap="hsv", linewidth=0, antialiased=False, alpha=0.8)
    #    ax1.plot_wireframe(X, Y, fit3, color="black", linewidth=3, alpha=0.3)
    #    ax1.set_xlabel("Pixels")
    #    ax1.set_ylabel("Pixels")
    #    ax1.set_zlabel("Intensity")
    #     
    #    plt.show()
sumBase = np.array(sumBase)
sumBase = sumBase.T

 
#Test Peaks fitting    
plt.figure(2)
plt.imshow(image[0,:, :])
plt.grid('off')
plt.scatter(peaks_base[:,0], peaks_base[:,1], facecolor='none', edgecolor="g")
plt.scatter(base_centers[:,0], base_centers[:,1], facecolor='none', edgecolor="r")    
    
#######################################################################################################

qd_centers = np.empty((0,2))
qd_centers_trans = np.empty((0,2)) #trans is in the coordinate system of the masked image below
#if a detected color is to far from the reference peak we need to increase the y box height
box = x_sides
box_y_top = y_top
box_y_bot = y_bot


rows_y = box_y_top + box_y_bot



sumRed = []
intensity = []
amp = []
sig = []
for k in range(len(im_red)): 
    for i,peaks in enumerate(peaks_base):
    
        xmin = peaks[0] - box/2
        xmax = peaks[0] + box/2
        ymin = peaks[1] - box_y_top
        ymax = peaks[1] + box_y_bot
        
        x = np.arange(xmin, xmax,1)
        y = np.arange(ymin, ymax,1)
        
        Y, X = np.meshgrid(x,y) #note: meshgrid is needed to use an asymetric grid with plot_surface, I tried mgrid and it didn't work
        
        redMask =  im_red[k][int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1] #to mask it correctly we have to invert X and Y here
        redMaskcopy =  np.copy(im_red[k])[int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1]
    
        sumRedTemp = np.sum(redMaskcopy, axis=1)
        sumRed.append(sumRedTemp)
        
        if (np.average(sumRedTemp)-np.min(sumRedTemp)) > 3000.0:
            
        
            ig = (np.min(redMask), np.max(redMask) , peaks[1]+0.5, peaks[0]+0.5, 2.0, 2.0, 0.0) #0.5 used to force fit because peaks were to close to the fit
            try:
                inpars, pcov = sp.curve_fit(twoD_Gaussian,(X,Y), redMask.ravel(), p0=ig, maxfev=2000)
                fit3 = twoD_Gaussian((Y, X), *inpars).reshape(X.shape[0], Y.shape[1])
                
            except RuntimeError:
                inpars = [0, 0, 0, 0 ,0, 0]
            if (inpars[3] > im_red[0].shape[0])  or (inpars[2]> im_red[0].shape[0]):
                inpars = [0, 0, 0, 0 ,0, 0]
            if (inpars[3]<0)  or (inpars[2]<0):
                inpars = [0, 0, 0, 0 ,0, 0]
                
        else:
            inpars = [0, 0, 0, 0 ,0, 0]
            
            
        intensity.append(np.abs(2*np.pi*inpars[1]*inpars[4]*inpars[5]))
        amp.append(inpars[1])
        sig.append([inpars[4], inpars[5]])
            
        qd_centers = np.append (qd_centers, [[inpars[3], inpars[2]]], axis=0)
        qd_centers_trans = np.append (qd_centers_trans, [[inpars[3]-xmin, inpars[2]-ymin]], axis=0) #parameters minx ymin and xmin to translate 
                                                                                                        #to the coordinate system of the masked image
         #Plot masked image and fitting


   
    #fig = plt.figure(i+10)
    #ax1 = fig.gca(projection='3d')
    #ax1.plot_surface(X, Y, redMask, cmap="jet", linewidth=0, antialiased=False, alpha=0.5)
    #ax1.plot_wireframe(X, Y, fit3, color="black", linewidth=3, alpha=0.3)
    #ax1.set_xlabel("Pixels")
    #ax1.set_ylabel("Pixels")
    #ax1.set_zlabel("Intensity")
sumRed = np.array(sumRed)
sumRed = sumRed.T



#Test Peaks fitting    
plt.figure(3)
plt.imshow(im_red[1])
plt.grid('off')
plt.scatter(peaks_base[:,0], peaks_base[:,1], facecolor='none', edgecolor="g")
plt.scatter(qd_centers[:,0], qd_centers[:,1], facecolor='none', edgecolor="r")   
    
#######################################################################################################



#########################################################################################################



#Wavelength Calculation
#coef comes from the calibration => coefficient of a 3rd order polynomial    
coef = np.array([
7.467e-05,0.02642, 3.511, 605])

p = np.poly1d(coef)




wavelengthTable = []

#for each center in the translated coordinate system of the masked image calculates the new wavelenght calibration
#by making the fitted center zero pixel displacement and applying the 3rd polynomial
for k in range(len(im_red)): 
    #for in for i, centers in enumerate(base_centers_trans):
    for i,centers in enumerate(base_centers_trans[k*peaks_pf:(k*peaks_pf + peaks_pf)]):
        print(i+(k*peaks_pf))
        pixels = np.arange(0,rows_y,1)
        x_center = centers[0]
        y_center = centers[1]
        pixels_x = pixels - x_center
        pixels_y = pixels - y_center
        pixel_disp = p(pixels_y)
        
 
            
        wavelengthTable.append(list(pixel_disp.T)) 
        wavelengthTable.append(list(sumRed[:, i*k].T)) 




df = pd.DataFrame(wavelengthTable)
df = df.T
col_names = ['Wavelength', 'Red']
#
df.columns = col_names*(int(len(df.columns)/2))
file_name = imagesFiles[1].split('.tif')




writer = pd.ExcelWriter('series-' + file_name[0] + '.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

int_df = pd.DataFrame(intensity) 
int_df.columns = ['Intensity']
writer = pd.ExcelWriter('int-series-' + file_name[0] + '.xlsx')
int_df.to_excel(writer,'Sheet1')
writer.save()