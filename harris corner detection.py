#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import Utilities as utl
import cv2
import numpy as np
import matplotlib.pyplot as plt


def NonMaximalSuppression(img, radius):
    """
    consider only the max value
    within window of size(radius x radius)
    around each pixel and assume all other value with 0
    """
    
    rows,columns= img.shape

   
    for i in range(radius,rows - radius):
        for j in range(radius,columns - radius):
            
            window = img[i - radius:i + radius, j - radius:j + radius]
            
            if img[i, j] != np.max(window):#only taking maximum
                img[i, j] = 0

    return img

"""
1- gradients in both the X and Y directions.
2- smooth the derivative a little using gaussian 
> try on TransA, SimA
> save output as  lab4-1-a-1.png, lab4-1-a-1.png
3- Calculate R:
	3.1 Loop on each pixel:
	3.2 Calculate M for each pixel:
		3.2.1 calculate a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2 
	3.3 Calculate Det_M = np.linalg.det(a) or Det_M = a11*a22 - a12*a21; and trace=a11+a22
	3.4 Calculate Response at this pixel = det-k*trace^2
	3.5 Display the result, but make sure to re-scale the data in the range 0 to 255 
4- Threshold and Non-Maximal Suppression 

"""
# 1- gradients in both the X and Y directions.
def harris(img, thresh=200, radius=2, verbose=True):
    Gx, Gy = utl.get_gradients_xy(img, 5)
    if verbose:
        cv2.imshow("Gradients", np.hstack([Gx, Gy]))

    # 2- smooth the derivative a little using gaussian
    #Student Code ~ 2 Lines
    Gx = cv2.GaussianBlur(Gx, (5, 5), sigmaX=3,sigmaY=0)
    Gy = cv2.GaussianBlur(Gy, (5, 5), sigmaX=0,sigmaY=3)
    #End Student Code
    
    cv2.imshow("Blured", np.hstack([Gx, Gy]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 3- Calculate R:
    R = np.zeros(img.shape)
    k = 0.04

    # 	3.1 Loop on each pixel:
    for i in range(len(Gx)):
        for j in range(len(Gx[i])):
    # 	3.2 Calculate M for each pixel:
    # 		    M = [[a11, a12],
    #                [a21, a22]]
    #           with a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2
            #Student Code ~ 1 line of code
            M = np.array([[int(Gx[i,j])*int(Gx[i,j]), int(Gx[i,j])*int(Gy[i, j])],
                          [int(Gx[i,j])*int(Gy[i,j]), int(Gy[i,j])*int(Gy[i, j])]])
            #Student Code

    # 	3.3 Calculate Det_M = np.linalg.det(a) or Det_M = a11*a22 - a12*a21; and trace=a11+a22
            Det_M = np.linalg.det(M)

    # 	3.4 Calculate Response at this pixel = det-k*trace^2
    #   where trace of M is the sum of its diagonals
            #Student Code ~ 1 line of code
            R[i, j] = Det_M - k*(M[0,0]+M[1, 1])**2
            #End Student Code

    # 4 Display the result, but make sure to re-scale the data in the range 0 to 255

    R = utl.rescale(R, 0, 255)

    plt.imshow(R, cmap="gray")
    plt.show()
    # 5- Threshold and Non-Maximal Suppression
    # Student Code ~ 2 lines of code
    # Threshold for an optimal value, it may vary depending on the image.
    R = NonMaximalSuppression(R, radius)
    R[R > 200] = 255
    R[R < 200] = 0
    # End Student Code
    plt.imshow(R, cmap="gray")
    plt.show()

    return R

img_pairs = [['transA', 'simA']]
dir = 'input/'
i = 0

for [img1, img2] in img_pairs:
    i += 1
    img1 = cv2.imread(dir + img1 + '.jpg', 0)
    img2 = cv2.imread(dir + img2 + '.jpg', 0)


    r1 = harris(img1)
    r2 = harris(img2)  # Note that the threshold may need to be different for each image
    plt.figure(i)
    plt.subplot(221), plt.imshow(img1, cmap='gray')
    plt.title('Original Image 1')
    plt.subplot(222), plt.imshow(img2, cmap='gray')
    plt.title('Original Image 2')
    plt.subplot(223), plt.imshow(r1, cmap='gray')
    plt.title('Corners Detected - Image 1')
    plt.subplot(224), plt.imshow(r2, cmap='gray')
    plt.title('Corners Detected - Image 2')
    plt.show()


# In[ ]:




