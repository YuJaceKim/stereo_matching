# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Group Number 9
#Name : Anvita Upadhyay and Isha Gupta
#UBName : anvitaup, ishagupt
#Person Number : 50147506 50208184
#Title : CSE 573 - Project - Disparity for Stereo Vision – Block Matching
#-------------------------------------------------------------------------------

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

# Loading the original images
view1 = cv2.imread('E:/Sem 1/CVIP/Project/Data/Data/view1.png',0)
view1 = np.asarray((view1),dtype = np.float)

view2 = cv2.imread('E:/Sem 1/CVIP/Project/Data/Data/view5.png',0)
view2 = np.asarray((view2),dtype = np.float)

# Loading the ground truth images
gndTruth1 = cv2.imread('E:/Sem 1/CVIP/Project/Data/Data/disp1.png',0)
gndTruth2 = cv2.imread('E:/Sem 1/CVIP/Project/Data/Data/disp5.png',0)
gndTruth1 = np.asarray((gndTruth1),dtype = np.float)
gndTruth2 = np.asarray((gndTruth2),dtype = np.float)
#-------------------------------------------------------------------------------

# Function to calculate disparity between two views of an image
def CalculateDisparity(blockSize,image1,image2, gndTruth, viewType):

    # Pad the image with zeros. Padding changes based on the block size
    if(blockSize ==3 ):
        padValue = 1
    elif(blockSize ==9 ):
        padValue = 4
        
    view1Padded = cv2.copyMakeBorder(image1,padValue,padValue,padValue,padValue,cv2.BORDER_CONSTANT,value=0)
    view2Padded = cv2.copyMakeBorder(image2,padValue,padValue,padValue,padValue,cv2.BORDER_CONSTANT,value=0)

    rows,columns = image1.shape

    # Store minSSD of all pixels 
    minSSD = np.full((image1.shape),99999.0,image1.dtype)
    DisparityMap = np.zeros(image1.shape,image1.dtype)
    
    if(blockSize ==3 ):
        initialValue = 1
    elif(blockSize ==9 ):
        initialValue = 4
        
    print "Processing..."
        
    for i in range(initialValue,rows):

        if(i%50 == 0):
            print "Processing row:", i

        for j in range(initialValue,columns):
            if(blockSize ==3 ):
                BlockView1 = np.array([[view1Padded[i-1,j-1],view1Padded[i-1,j],view1Padded[i-1,j+1]],
                                    [view1Padded[i,j-1]  ,view1Padded[i,j]  ,view1Padded[i,j+1  ]],
                                    [view1Padded[i+1,j-1],view1Padded[i+1,j],view1Padded[i+1,j+1]]])
            else:
                BlockView1 = np.array([[view1Padded[i-4,j-4],view1Padded[i-4,j-3],view1Padded[i-4,j-2],view1Padded[i-4,j-1],view1Padded[i-4,j],view1Padded[i-4,j+1],view1Padded[i-4,j+2],view1Padded[i-4,j+3],view1Padded[i-4,j+4]],
                                       [view1Padded[i-3,j-4],view1Padded[i-3,j-3],view1Padded[i-3,j-2],view1Padded[i-3,j-1],view1Padded[i-3,j],view1Padded[i-3,j+1],view1Padded[i-3,j+2],view1Padded[i-4,j+3],view1Padded[i-3,j+4]],
                                       [view1Padded[i-2,j-4],view1Padded[i-2,j-3],view1Padded[i-2,j-2],view1Padded[i-2,j-1],view1Padded[i-2,j],view1Padded[i-2,j+1],view1Padded[i-2,j+2],view1Padded[i-2,j+3],view1Padded[i-2,j+4]],
                                       [view1Padded[i-1,j-4],view1Padded[i-1,j-3],view1Padded[i-1,j-2],view1Padded[i-1,j-1],view1Padded[i-1,j],view1Padded[i-1,j+1],view1Padded[i-1,j+2],view1Padded[i-1,j+3],view1Padded[i-1,j+4]],
                                       [view1Padded[i,j-4],  view1Padded[i,j-3],  view1Padded[i,j-2],  view1Padded[i,j-1],  view1Padded[i,j],  view1Padded[i,j+1],  view1Padded[i,j+2],  view1Padded[i,j+3],  view1Padded[i,j+4]],
                                       [view1Padded[i+1,j-4],view1Padded[i+1,j-3],view1Padded[i+1,j-2],view1Padded[i+1,j-1],view1Padded[i+1,j],view1Padded[i+1,j+1],view1Padded[i+1,j+2],view1Padded[i+1,j+3],view1Padded[i+1,j+4]],
                                       [view1Padded[i+2,j-4],view1Padded[i+2,j-3],view1Padded[i+2,j-2],view1Padded[i+2,j-1],view1Padded[i+2,j],view1Padded[i+2,j+1],view1Padded[i+2,j+2],view1Padded[i+2,j+3],view1Padded[i+2,j+4]],
                                       [view1Padded[i+3,j-4],view1Padded[i+3,j-3],view1Padded[i+3,j-2],view1Padded[i+3,j-1],view1Padded[i+3,j],view1Padded[i+3,j+1],view1Padded[i+3,j+2],view1Padded[i+3,j+3],view1Padded[i+3,j+4]],
                                       [view1Padded[i+4,j-4],view1Padded[i+4,j-3],view1Padded[i+4,j-2],view1Padded[i+4,j-1],view1Padded[i+4,j],view1Padded[i+4,j+1],view1Padded[i+4,j+2],view1Padded[i+4,j+3],view1Padded[i+4,j+4]]])
            
              
            if(viewType == 1):
                startRange = j
                
                endRange = j - 100
                if(j-100 < 0):
                    endRange = 0
                     
                iterator = -1      
            elif(viewType == 2):
                startRange = j

                endRange = j + 100
                if(j+100 > columns):
                    endRange = columns

                iterator = 1
                
            for k in range(startRange, endRange, iterator):

                if(blockSize ==3 ):
                    BlockView2 = np.array([[view2Padded[i-1,k-1],view2Padded[i-1,k],view2Padded[i-1,k+1]],
                                            [view2Padded[i,k-1]  ,view2Padded[i,k]  ,view2Padded[i,k+1]],
                                            [view2Padded[i+1,k-1],view2Padded[i+1,k],view2Padded[i+1,k+1]]])
                else:
                    BlockView2 = np.array([[view2Padded[i-4,k-4],view2Padded[i-4,k-3],view2Padded[i-4,k-2],view2Padded[i-4,k-1],view2Padded[i-4,k],view2Padded[i-4,k+1],view2Padded[i-4,k+2],view2Padded[i-4,k+3],view2Padded[i-4,k+4]],
                                           [view2Padded[i-3,k-4],view2Padded[i-3,k-3],view2Padded[i-3,k-2],view2Padded[i-3,k-1],view2Padded[i-3,k],view2Padded[i-3,k+1],view2Padded[i-3,k+2],view2Padded[i-3,k+3],view2Padded[i-3,k+4]],
                                           [view2Padded[i-2,k-4],view2Padded[i-2,k-3],view2Padded[i-2,k-2],view2Padded[i-2,k-1],view2Padded[i-2,k],view2Padded[i-2,k+1],view2Padded[i-2,k+2],view2Padded[i-2,k+3],view2Padded[i-2,k+4]],
                                           [view2Padded[i-1,k-4],view2Padded[i-1,k-3],view2Padded[i-1,k-2],view2Padded[i-1,k-1],view2Padded[i-1,k],view2Padded[i-1,k+1],view2Padded[i-1,k+2],view2Padded[i-1,k+3],view2Padded[i-1,k+4]],
                                           [view2Padded[i,k-4],  view2Padded[i,k-3],  view2Padded[i,k-2],  view2Padded[i,k-1],  view2Padded[i,k],  view2Padded[i,k+1],  view2Padded[i,k+2],  view2Padded[i,k+3],  view2Padded[i,k+4]],
                                           [view2Padded[i+1,k-4],view2Padded[i+1,k-3],view2Padded[i+1,k-2],view2Padded[i+1,k-1],view2Padded[i+1,k],view2Padded[i+1,k+1],view2Padded[i+1,k+2],view2Padded[i+1,k+3],view2Padded[i+1,k+4]],
                                           [view2Padded[i+2,k-4],view2Padded[i+2,k-3],view2Padded[i+2,k-2],view2Padded[i+2,k-1],view2Padded[i+2,k],view2Padded[i+2,k+1],view2Padded[i+2,k+2],view2Padded[i+2,k+3],view2Padded[i+2,k+4]],
                                           [view2Padded[i+3,k-4],view2Padded[i+3,k-3],view2Padded[i+3,k-2],view2Padded[i+3,k-1],view2Padded[i+3,k],view2Padded[i+3,k+1],view2Padded[i+3,k+2],view2Padded[i+3,k+3],view2Padded[i+3,k+4]],
                                           [view2Padded[i+4,k-4],view2Padded[i+4,k-3],view2Padded[i+4,k-2],view2Padded[i+4,k-1],view2Padded[i+4,k],view2Padded[i+4,k+1],view2Padded[i+4,k+2],view2Padded[i+4,k+3],view2Padded[i+4,k+4]]])
            
                # Calculate SSD value for current shift using 3x3 block and compare with MIN SSD
                
                SSD = 0
                for m in range(blockSize):
                        for n in range(blockSize):
                            SSD = SSD + math.pow((BlockView1[m,n] - BlockView2[m,n]),2)
                    
                if (SSD < minSSD[i,j]):
                    minSSD[i,j] = SSD
                    DisparityMap[i,j] = abs(j-k)
            
            
    #--------------------------------------------------------
    #--------------CALCULATE MSE----------------------------
    #--------------------------------------------------------
    
    MSE = 0.0
    for i in xrange(gndTruth.shape[0]):
        for j in xrange(gndTruth.shape[1]):
            MSE = MSE + math.pow((gndTruth[i][j] - DisparityMap[i][j]), 2)
    MSE = MSE/(gndTruth.shape[0] * gndTruth.shape[1])
    
    if(blockSize == 3):  
        # viewType 1 is for Left to Right and 2 for Right to Left
        if(viewType == 1):
            print "Mean Square Error for Left view with 3x3 block = %d" %MSE
        else:
            print "Mean Square Error for Right view with 3x3 block = %d" %MSE            
    else:
        if(viewType == 1):
            print "Mean Square Error for Left view with 9x9 block = %d" %MSE
        else:
            print "Mean Square Error for Right view with 9x9 block = %d" %MSE          

    return DisparityMap  
#-------------------------------------------------------------------------------            
    
def displayImages(img1,img2,img3,img4, blockSize):
    
    plt.subplot(221),plt.imshow(img1, cmap = 'gray')
    plt.title('Left Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(img2, cmap = 'gray')
    plt.title('Right Input Image'), plt.xticks([]), plt.yticks([])

    if(blockSize == 3):
        plt.subplot(223),plt.imshow(img3, cmap = 'gray')
        plt.title('3x3: Left wrt Right Disparity'), plt.xticks([]), plt.yticks([])
        plt.subplot(224),plt.imshow(img4, cmap = 'gray')
        plt.title('3x3: Right wrt Left Disparity'), plt.xticks([]), plt.yticks([])
    elif(blockSize == 9):
        plt.subplot(223),plt.imshow(img3, cmap = 'gray')
        plt.title('9x9: Left wrt Right Disparity'), plt.xticks([]), plt.yticks([])
        plt.subplot(224),plt.imshow(img4, cmap = 'gray')
        plt.title('9x9: Right wrt Left Disparity'), plt.xticks([]), plt.yticks([])
    
    plt.figure()
    plt.show()
#-------------------------------------------------------------------------------

def main():
    
     Disp3x3LtoR = CalculateDisparity(3,view1,view2,gndTruth1, 1)
     Disp3x3RtoL = CalculateDisparity(3,view2,view1,gndTruth2, 2)     
     Disp9x9LtoR = CalculateDisparity(9,view1,view2,gndTruth1, 1)
     Disp9x9RtoL = CalculateDisparity(9,view2,view1,gndTruth2, 2)
     
     displayImages(view1,view2,Disp3x3LtoR,Disp3x3RtoL, 3)
     displayImages(view1,view2,Disp9x9LtoR,Disp9x9RtoL, 9)
#-------------------------------------------------------------------------------

main()
#===============================================================================