# =============================================================================
# This code calculates the wavelength based on the calibration of the prism. 
# Input: image files with the different colors on imageFiles variable
# Output: graphs of the spectra by wavelength
# coef variable is the 3rd polynomial coefcients from the prism calibration
# Code written by Andre Thomaz 1/22/2018
# =============================================================================


#import seaborn as sns
#sns.set()
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sp
from skimage.feature import peak_local_max
from PIL import Image
from scipy import sparse
#from scipy.spatial import distance
import pandas as pd



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
    

imagesFiles = [ '680-5.tif',  'red-5.tif', 'green-5.tif']



image = Image.open(imagesFiles[0])
im_red = Image.open(imagesFiles[1])
im_green = Image.open(imagesFiles[2])



image = np.array(image)

#Check treshold, otherwise it may find more than one peak
thr_base = np.max(image)-35*np.std(image)
peaks_base = peak_local_max(image, min_distance=5, threshold_abs=thr_base)
peaks_base[:,0], peaks_base[:,1] = peaks_base[:,1], peaks_base[:,0].copy()

#Define the size of the box/rectangle to crop the image to find peaks
x_sides = 18
y_top = 50 #if one of the colors is too far away from 680nm rectangle
y_bot = 20


#if points are to close to the borders dont consider it 
delete_base = []
for i,points in enumerate(peaks_base):
    if points[1]-y_top<0:                             #if point is too close to the top
        delete_base.append(i)
    elif points[1] + y_bot >image.shape[0]:           #if point is too close to the bottom
        delete_base.append(i)
    if points[0] + 10 > image.shape[1]:               #if point is too close to the right
        delete_base.append(i)
    if points[0] - 10 < 0:                            #if point is too close to the left
        delete_base.append(i)
peaks_base = np.delete(peaks_base, delete_base, 0)

#peaks_base = np.copy(peaks_base[0:6,:]) #Fix this!!!!!!!!!!!!!!!!!!!!!!!
print(str(peaks_base.shape[0]) + ' 680 peaks')

#plot image and detected peaks
plt.figure(1)
plt.imshow(image)
plt.grid('off')
plt.scatter(peaks_base[:,0], peaks_base[:,1], facecolor='none', edgecolor="white")

##############################################################################
im_red = np.array(im_red)


#thr_red = np.max(im_red)-18*np.std(im_red)
##peaks_red = peak_local_max(im_red, min_distance=10, threshold_abs=thr_red)
##peaks_red[:,0], peaks_red[:,1] = peaks_red[:,1], peaks_red[:,0].copy()
#
#
#delete_base = []
#for i,points in enumerate(peaks_red):
#    if points[1]-30<0:
#        delete_base.append(i)
#    if 512-points[0]<15:
#        delete_base.append(i)
#peaks_red = np.delete(peaks_red, delete_base, 0)
#
#print(str(peaks_red.shape[0]) + ' red peaks')
#
##
##
#plt.figure(3)
#plt.imshow(im_red)
#plt.grid('off')
#plt.scatter(peaks_red[:,0], peaks_red[:,1], facecolor='none', edgecolor="white")

#############################################################################
im_green = np.array(im_green)

#thr_green = np.max(im_green)-16.5*np.std(im_green)
#peaks_green = peak_local_max(im_green, min_distance=10, threshold_abs=thr_green)
#peaks_green[:,0], peaks_green[:,1] = peaks_green[:,1], peaks_green[:,0].copy()
#
#delete_base = []
#for i,points in enumerate(peaks_green):
#    if points[1]-30<0:
#        delete_base.append(i)
#    if 512-points[0]<15:
#        delete_base.append(i)
#peaks_green = np.delete(peaks_green, delete_base, 0)
#
#print(str(peaks_green.shape[0]) + ' green peaks')
#
#plt.figure(4)
#plt.imshow(im_green)
#plt.grid('off')
#plt.scatter(peaks_base[:,0], peaks_base[:,1], facecolor='none', edgecolor="white")

#############################################

#Calculation of the centers detected by the peaks of local maxima

base_centers = np.empty((0,2))
base_centers_trans = np.empty((0,2)) #trans is in the coordinate system of the masked image below

#if a detected color is to far from the reference peak we need to increase the y box height
box = x_sides
box_y_top = y_top
box_y_bot = y_bot

rows_y = box_y_top + box_y_bot

sumBase = np.empty((rows_y, peaks_base.shape[0])) #empty matrix to hold the collapsed sum of images/spectra


#fitting of a Gaussian to each detected peak before:

for i, peaks in enumerate(peaks_base):
         
    xmin = peaks[0] - box/2
    xmax = peaks[0] + box/2
    ymin = peaks[1] - box_y_top
    ymax = peaks[1] + box_y_bot
    
    x = np.arange(xmin, xmax,1)
    y = np.arange(ymin, ymax,1)
    
    Y, X = np.meshgrid(x,y) #note: meshgrid is needed to use an asymetric grid with plot_surface, I tried mgrid and it didn't work.
    
    imageMask =  image[int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1] #to mask it correctly we have to invert X and Y here
    imageMaskcopy =  np.copy(image)[int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1] #copy to do calculation in masked image coordinates
    
    
    
    sumBaseTemp = np.sum(imageMaskcopy, axis=1) #collapse the x direction to get spectra along y direction
    sumBase[:,i] = sumBaseTemp
  
    ig = (np.min(imageMask), np.max(imageMask) , peaks[1]+0.5, peaks[0]+0.5, 2.0, 2.0, 0.0) #0.5 used to force fit because peaks were to close to the fit
    inpars, pcov = sp.curve_fit(twoD_Gaussian,(X,Y), imageMask.ravel(), p0=ig, maxfev=20000)
    fit3 = twoD_Gaussian((Y, X), *inpars).reshape(X.shape[0], Y.shape[1])
    
    base_centers = np.append (base_centers, [[inpars[2], inpars[3]]], axis=0)
    base_centers_trans = np.append (base_centers_trans, [[inpars[2]-ymin, inpars[3]-xmin]], axis=0) #parameters minx ymin and xmin to translate 
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



#base_centers_temp = []
#base_centers_trans_temp = []
#
#for i,row in enumerate(base_centers):
#    if i == base_centers.shape[0]:
#        break
#    else: 
#        dist_cell = np.linalg.norm(base_centers[i]-base_centers[i+1])
#        if dist_cell>0.01:
#            base_centers_temp.append([row[0], row[1]])
#            base_centers_trans_temp.append([base_centers_trans[i][0], base_centers_trans[i][1]])


            
#base_centers = np.array([base_centers_temp][0])
base_centers[:, 0], base_centers[:, 1] = base_centers[:, 1], base_centers[:, 0].copy() #invert back to X,Y order
#base_centers_trans = np.array([base_centers_trans_temp][0])
base_centers_trans[:, 0], base_centers_trans[:, 1] = base_centers_trans[:, 1], base_centers_trans[:, 0].copy() #invert back to X,Y order
#print(str(base_centers.shape[0]) + ' 680 peaks cleaned')
#        
#Test Peaks fitting    
plt.figure(2)
plt.imshow(image)
plt.grid('off')
plt.scatter(peaks_base[:,0], peaks_base[:,1], facecolor='none', edgecolor="g")
plt.scatter(base_centers[:,0], base_centers[:,1], facecolor='none', edgecolor="r")    
    
#######################################################################################################


#if a detected color is to far from the reference peak we need to increase the y box height
box = x_sides
box_y_top = y_top
box_y_bot = y_bot


rows_y = box_y_top + box_y_bot



sumRed = np.empty((rows_y, base_centers.shape[0]))

for i,peaks in enumerate(peaks_base):
    
    xmin = peaks[0] - box/2
    xmax = peaks[0] + box/2
    ymin = peaks[1] - box_y_top
    ymax = peaks[1] + box_y_bot
    
    x = np.arange(xmin, xmax,1)
    y = np.arange(ymin, ymax,1)
    
    Y, X = np.meshgrid(x,y) #note: meshgrid is needed to use an asymetric grid with plot_surface, I tried mgrid and it didn't work
    
    redMask =  im_red[int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1] #to mask it correctly we have to invert X and Y here
    redMaskcopy =  np.copy(im_red)[int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1]

    sumRedTemp = np.sum(redMaskcopy, axis=1)
    sumRed[:,i] = sumRedTemp
    
    #fig = plt.figure(i+10)
    #ax1 = fig.gca(projection='3d')
    #ax1.plot_surface(X, Y, redMask, cmap="jet", linewidth=0, antialiased=False, alpha=0.5)
    #ax1.plot_wireframe(X, Y, fit3, color="black", linewidth=3, alpha=0.3)
    #ax1.set_xlabel("Pixels")
    #ax1.set_ylabel("Pixels")
    #ax1.set_zlabel("Intensity")
    
#######################################################################################################
box = x_sides
box_y_top = y_top
box_y_bot = y_bot


rows_y = box_y_top + box_y_bot
#rows_y += 1


sumGreen = np.empty((rows_y, base_centers.shape[0]))

for i,peaks in enumerate(peaks_base):
    
    xmin = peaks[0] - box/2
    xmax = peaks[0] + box/2
    ymin = peaks[1] - box_y_top
    ymax = peaks[1] + box_y_bot
    
    x = np.arange(xmin, xmax,1)
    y = np.arange(ymin, ymax,1)
    
    Y, X = np.meshgrid(x,y) #note: meshgrid is needed to use an asymetric grid with plot_surface, I tried mgrid and it didn't work
    

    
    greenMask = im_green[int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1] #to mask it correctly we have to invert X and Y here
  
    greenMaskcopy =  np.copy(im_green)[int(np.min(X)):int(np.max(X))+1, int(np.min(Y)):int(np.max(Y))+1]
    sumGreenTemp = np.sum(greenMaskcopy, axis=1)

    sumGreen[:,i] = sumGreenTemp
    
#    fig = plt.figure(i+10)
#    ax1 = fig.gca(projection='3d')
#    ax1.plot_surface(X, Y, greenMask, cmap="jet", linewidth=0, antialiased=False, alpha=0.5)


#########################################################################################################
#Function to remove baseline by "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens 2005
def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = sparse.linalg.spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z


#Wavelength Calculation
#coef comes from the calibration => coefficient of a 3rd order polynomial    
coef = np.array([
0.0001042,  0.0304,  4.435,  680])

p = np.poly1d(coef)

#In this case there is some leaking between the channels and also because the mask box is big a lot of the 
#masked image is background, setting this intervals to 0 takes care of this
sumGreen[36:,:] = 0
#sumGreen[35:,:] = 0
sumRed[0:34,:] = 0
sumBase[0:35,:] = 0


#Tables to hold spectral mean for each channel
spectralMeanTableRed = np.empty((np.shape(base_centers_trans)[0],1))
spectralMeanTableGreen = np.empty((np.shape(base_centers_trans)[0],1))
spectralMeanTableBase = np.empty((np.shape(base_centers_trans)[0],1))
wavelengthTable = []



#for each center in the translated coordinate system of the masked image calculates the new wavelenght calibration
#by making the fitted center zero pixel displacement and applying the 3rd polynomial
for i, centers in enumerate(base_centers_trans):
    pixels = np.arange(0,rows_y,1)
    x_center = centers[0]
    y_center = centers[1]
    pixels_x = pixels - x_center
    pixels_y = pixels - y_center
    pixel_disp = p(pixels_y)

   
    sumNormRed = (sumRed[:,i]) /np.max(sumRed[:,i])
    sumNormGreen = (sumGreen[:,i]) /np.max(sumGreen[:,i]) 
    sumNormBase = (sumBase[:,i])/np.max(sumBase[:,i]) 
    
    #Save spectra by wavelength
    wavelengthTable.append(list(pixel_disp.T)) 
    wavelengthTable.append(list(sumNormRed.T)) 
    wavelengthTable.append(list(sumNormGreen.T)) 
    
    #Spectral mean Calculation (wavelength average weighted by intensity)
    spectralMeanRed = np.sum(np.multiply(sumRed[:,i],pixel_disp))/np.sum(sumRed[:,i])
    
    spectralMeanGreen = np.sum(np.multiply(sumGreen[:,i],pixel_disp))/np.sum(sumGreen[:,i])
    
    spectralMeanBase = np.sum(np.multiply(sumBase[:,i],pixel_disp))/np.sum(sumBase[:,i])
    
    
    spectralMeanTableRed[i,0] = spectralMeanRed
    spectralMeanTableGreen[i,0] = spectralMeanGreen
    spectralMeanTableBase[i,0] = spectralMeanBase
    
    
    plt.figure(4)
    plt.scatter([680.0],spectralMeanRed )
    plt.scatter([580.0],spectralMeanGreen )
    #plt.scatter([677.0],spectralMeanBase )
##    plt.xticks(np.arange(560,720,20))
#    plt.yticks(np.arange(560,720,20))
#    plt.xlim(560,720)
#    plt.ylim(560,720)
##    
    #if i in [5]:
    if i not in [100]:
        plt.figure(6)
        ##        zz = baseline_als(sumRed[30:,i], 1000000, 0.0001)
        ##        corre = sumRed[30:,i]-zz
        ##        corre = corre/np.max(corre)
        plt.plot(pixel_disp[34:],sumNormRed[34:], label=i)
        #plt.plot(pixel_disp,sumNormBase, label=i)
        #    #plt.legend()
        # 
        #
        plt.plot(pixel_disp[0:34],sumNormGreen[0:34], label=i)
        #plt.legend()
        plt.xlim(550,800)
#        plt.xlim(640,720)
#        plt.ylim(0,1.2)
#    
#Plot spectral mean for each color/dye
plt.figure(3)
plt.errorbar([580.0], np.mean(spectralMeanTableGreen),yerr=np.std(spectralMeanTableGreen), marker='o', markersize=5)
plt.errorbar([680.0], np.mean(spectralMeanTableRed), yerr=np.std(spectralMeanTableRed), marker='o', markersize=5)
#plt.errorbar([680.0], np.mean(spectralMeanTableBase), yerr=np.std(spectralMeanTableBase), marker='o', markersize=5)
plt.xticks(np.arange(560,750,20))
plt.yticks(np.arange(560,750,20))
plt.xlim(570,700)
plt.ylim(560,720)
plt.grid('on')
plt.xlabel('Nominal Wavelength (nm)')
plt.ylabel('Spectral Mean (nm)')

#
df = pd.DataFrame(wavelengthTable)
df = df.T
col_names = ['Wavelength', 'Red', 'Green']

df.columns = col_names*(int(len(df.columns)/3))
file_name = imagesFiles[0].split('-')
file_name = file_name[1].split('.')

writer = pd.ExcelWriter('output-' + file_name[0] + '.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()
#    
