# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Group Number 9
#Name : Anvita Upadhyay and Isha Gupta
#UBName : anvitaup, ishagupt
#Person Number : 50147506 50208184
#Title : CSE 573 - Project - Disparity for Stereo Vision – Dynamic Programming
#-------------------------------------------------------------------------------

import numpy as np
np.seterr(over='ignore')
import cv2
import math
from matplotlib import pyplot as plt

def main():
    
    # Loading original left and right images
    original_Left_Image = cv2.imread("E:/Sem 1/CVIP/Project/Data/Data/view1.png",0)
    original_Right_Image = cv2.imread("E:/Sem 1/CVIP/Project/Data/Data/view5.png",0)
    
    leftImage = np.asarray(original_Left_Image, dtype = np.float)
    rightImage = np.asarray(original_Right_Image, dtype = np.float)
    
    # Loading given ground truth images
    ground_Truth_Left_Image = cv2.imread("E:/Sem 1/CVIP/Project/Data/Data/disp1.png",0)
    ground_Truth_Right_Image = cv2.imread("E:/Sem 1/CVIP/Project/Data/Data/disp5.png",0)
    
    ground_Truth_Left = np.asarray(ground_Truth_Left_Image, dtype = np.float)
    ground_Truth_Right = np.asarray(ground_Truth_Right_Image, dtype = np.float)
        
    rowCount, columnCount = leftImage.shape
    
    costMatrix = np.empty([columnCount, columnCount], dtype=np.float)
    M = np.empty([columnCount, columnCount], dtype=np.float)
    disparity_Map_Left = np.empty([rowCount, columnCount], dtype=np.float)
    disparity_Map_Right = np.empty([rowCount, columnCount], dtype=np.float)
    
    # By trial and error, getting best results for Occulusion value 6
    occlusion = 6

    cost = 0.0
    costMatrix[0, 0] = 0
    for row in range(0, rowCount):
    
        if(row%50 == 0):
            print "Processing row ", row
        
        for i in range(1, columnCount):
            costMatrix[i, 0] = i * occlusion
            costMatrix[0, i] = i * occlusion
            
        for i in range(0, columnCount):
            for j in range(0, columnCount):

                # Cost function for matching features in the left and right images
                cost = abs(leftImage[row, i] - rightImage[row, j])
                
                min1 = costMatrix[i-1, j-1] + cost
                min2 = costMatrix[i-1, j] + occlusion
                min3 = costMatrix[i, j-1] + occlusion
        
                cmin = min(min1, min2, min3)

                costMatrix[i, j] = cmin
                
                # Forming path matrix
                if(cmin == min1):
                    M[i, j] = 1
                if(cmin == min2):
                    M[i, j] = 2
                if(cmin == min3):
                    M[i, j] = 3

        p = columnCount - 1
        q = columnCount - 1
        
        while(p != 0 and q !=0):
            
            # if feature in left and right image matches
            if(M[p, q] == 1):
                disparity_Map_Left[row, p] = abs(p-q)
                disparity_Map_Right[row, q] = abs(q-p)                
                p = p - 1
                q = q - 1
                
            # if feature in left image is occuluded
            elif(M[p, q] == 2):
                disparity_Map_Left[row, p] = 0
                p = p - 1
        
            # if feature in right image is occuluded
            elif(M[p, q] == 3):
                disparity_Map_Right[row, q] = 0
                q = q - 1
                
        costMatrix = np.empty([columnCount, columnCount], dtype=np.float)
        M = np.empty([columnCount, columnCount], dtype=np.float)
    
    mse_Left = CalculateMSE(disparity_Map_Left, ground_Truth_Left)
    mse_Right = CalculateMSE(disparity_Map_Right, ground_Truth_Right)
    
    print "Mean Square Error of Left Image : ", mse_Left
    print "Mean Square Error of Right Image : ", mse_Right    
    
    PlotImages(leftImage, rightImage, disparity_Map_Left, disparity_Map_Right)
#-------------------------------------------------------------------------------                

# Calculating the Mean Square Error of the original image and the reconstructed image
def CalculateMSE(disparityMap, groundTruth):

    MSE = 0.0

    for i in range(0, groundTruth.shape[0]):
        for j in range(0, groundTruth.shape[1]):
            if(disparityMap[i][j] != 0):
                MSE += math.pow((groundTruth[i][j] - disparityMap[i][j]), 2)
    MSE = MSE/(groundTruth.shape[0] * groundTruth.shape[1])
    
    return MSE
#-------------------------------------------------------------------------------
    
# Plot Images
def PlotImages(leftImage, rightImage, disparity_Map_Left, disparity_Map_Right):
    plt.subplot(221),plt.imshow(leftImage, cmap = 'gray')
    plt.title('Original Left Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(222),plt.imshow(rightImage, cmap = 'gray')
    plt.title('Original Right Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(223),plt.imshow(disparity_Map_Left, cmap = 'gray')
    plt.title('Left Disparity Map'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(224),plt.imshow(disparity_Map_Right, cmap = 'gray')
    plt.title('Right Disparity Map'), plt.xticks([]), plt.yticks([])
    
    plt.show()
#-------------------------------------------------------------------------------
    
main()
#===============================================================================