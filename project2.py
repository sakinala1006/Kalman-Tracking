#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:20:17 2020

@author: saisrinijasakinala
"""
import imageio
import math
import numpy as np
from scipy import ndimage as nd

    
def featureTrack(files,f,f2,m2,new_pointsx,new_pointsy):
    #FUNCTION TO AUTO CORRELATE AMONG IMAGES.
    
    pointsx=[]
    pointsy=[]
    for featurex, featurey in zip(new_pointsx,new_pointsy):
    
        P=[]
        newx=featurex
        newy=featurey
        
        #setting the kernel of size 5x5
        sigma=3
        d=sigma
        denominator=1/2/math.pi/sigma**2
        kernel=np.zeros((d+2,d+2),dtype=float)
        for i in range(d+2):
            for j in range(d+2):
                kx,ky=i-d,j-d
                kernel[i,j]=denominator*math.exp(-1.0*(kx**2+ky**2)/(2*sigma**2))
        
        #iterating through every point within a window of size 9x9 around each feature point
        for u in range(featurex-4,featurex+5):
            for v in range(featurey-4,featurey+5):
                
                #checking boundaries
                if u<0: u=0
                if v<0: v=0
                if u>len(f2)-1: u=len(f2)-1
                if v>len(f2[0])-1: v=len(f2[0])-1
                sdiff=0
                
                i=featurex-2
                j=featurey-2
                kl=0
                km=0
                
                if i<0: i=0
                if j<0: j=0
                if i>len(f)-1: i=len(f)-1
                if j>len(f[0])-1: j=len(f[0])-1
                
                for inx in range(u-2,u+3):
                    for iny in range(v-2,v+3):
                        if inx<0: inx=0
                        if iny<0: iny=0
                        if inx>len(f2)-1: inx=len(f2)-1
                        if iny>len(f2[0])-1: iny=len(f2[0])-1
                        sdiff += (((f2[inx][iny]-f[i][j])**2)*kernel[kl][km]) #finding the sum of squared differences.
                        
                        j+=1
                        km+=1
                    i+=1
                    kl+=1
                    j=featurey-2
                    km=0
                    if i<0: i=0
                    if j<0: j=0
                    if i>len(f)-1: i=len(f)-1
                    if j>len(f[0])-1: j=len(f[0])-1
                    
                P.append([u,v,sdiff]) #appending the sums and locations of all the pixels within the window to a list

        Q=[]
        for i in range(0,len(P)):
            if(P[i][2]!=0):
                Q.append([P[i][0],P[i][1],P[i][2]]) #copying non-zero values into another list

        if(len(Q)==0): #if there are no non-zero elements in the list, set newx and newy to the first element.
            newx=P[0][0]
            newy=P[0][1]
            
        else: #finding the minimum of all the sums
            mini=Q[0][2]
            for i in range(0,len(Q)):
                if(Q[i][2]<mini):
                    mini=Q[i][2]
                    newx=Q[i][0]
                    newy=Q[i][1]
        
        #setting the intensities at new feature points to RED(255,0,0).
        m2[newx][newy][0]=255.0
        m2[newx][newy][1]=0.0
        m2[newx][newy][2]=0.0
        pointsx.append(newx)
        pointsy.append(newy)
    s='/Users/saisrinijasakinala/Desktop/Cv project 2/output/moon_output_frames/moon'+str(files)+'.png'
    imageio.imwrite(s,m2)
        
    return pointsx,pointsy

def KalmanTrack(files,f,f2,m3,xptsK,yptsK,new_pointsx,new_pointsy,listP0,v1,v2):
    #FUNCTION TO PERFORM FEATURE TRACKING USING KALMAN'S APPROACH.
    
    #setting all the pre defined matrices.
    A=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    Q=np.array([[0.25,0,0,0],[0,0.25,0,0],[0,0,0.25,0],[0,0,0,0.25]])
    H=np.array([[1,0,0,0],[0,1,0,0]])
    R=np.array([[1,0],[0,1]])
    I=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    
    i=0
    npx=[]
    npy=[]
    listP1=listP0.copy() #listP1 has the P0 matrices for each feature point
    
    #iterating over all the feature points in the image
    for featurex, featurey in zip(xptsK, yptsK):
        P0 = np.array(listP0[i]) 
        
        S0_ = np.array([[featurex],[featurey],[v1],[v2]]) #calculating S0-
        
        S1_=np.array(np.matmul(A,S0_),dtype=float) #calculating S1-
        
        P1_= np.array(((np.matmul(np.matmul(A,P0),A.T))+Q),dtype=float) #calculating P1-
        
        K=np.array(np.matmul(np.matmul(P1_,H.T),np.linalg.inv(np.matmul(np.matmul(H,P0),H.T)+R)),dtype=float) #calculation Kalman gain K
        
        P1 = np.array(np.matmul(np.subtract(I,np.matmul(K,H)),P1_),dtype=float) #calculating P1
        
        w = np.array([[new_pointsx[i]-xptsK[i]],[new_pointsy[i]-yptsK[i]]],dtype=float) #taking zero mean gaussian noise
        
        m1 = np.array((np.matmul(H,S1_)+w),dtype=float) #calculating measurement vector m1
        
        S1 = np.array(S1_ + np.matmul(K,np.subtract(m1,np.matmul(H,S1_)))) #calculating S1
        
        #appending new X and new Y values
        npx.append(S1[0])
        npx.append(S1[1])
        listP1[i]=P1
        
        #setting the intensities at new feature points to RED(255,0,0).
        m3[int(S1[0])][int(S1[1])][0]=255.0
        m3[int(S1[0])][int(S1[1])][1]=0.0
        m3[int(S1[0])][int(S1[1])][2]=0.0
    s='/Users/saisrinijasakinala/Desktop/Cv project 2/output/moon_kalman_frames/moon'+str(files)+'.png'
    imageio.imwrite(s,m2)
    return v1,v2,listP1,npx,npy

  
if __name__=='__main__':
    f=imageio.imread("/Users/saisrinijasakinala/Desktop/Cv project 2/input/moon_frames/0.png",as_gray=True)
    m=imageio.imread("/Users/saisrinijasakinala/Desktop/Cv project 2/input/moon_frames/0.png")
    
    #Copying the image array into two other arrays for storing Ix and Iy
    Ix=np.copy(f)
    Iy=np.copy(f)
    
    
    #finding first order gaussian derivative of the image along x and y axes individually
    # SIGMA = 3
    xFirstOrder=nd.filters.gaussian_filter(f,3,[1,0])
    yFirstOrder=nd.filters.gaussian_filter(f,3,[0,1])
    
    
    #convolving the image with zero order gaussian derivative
    # SIGMA = 3
    xZeroOrder=nd.filters.gaussian_filter1d(f,3,axis=0)
    yZeroOrder=nd.filters.gaussian_filter1d(f,3,axis=1)
    
    #let us set the WINDOW SIZE to be 7
    #ignoring the boundaries, start from (3,3) pixel till (len-4,len-4) pixel
    #convolve the images obtained in the previous two steps.
    #Ix=Gx.G(y)*I
    #Iy=G(x).Gy*I
    for i in range(3,len(f)-4):
        for j in range(3,len(f[0])-4):
            s=0
            t=0
            for k in range(i-3,i+4):
                for l in range(j-3,j+4):
                    s = s + (xFirstOrder[k][l]*yZeroOrder[k][l])
                    t = t + (yFirstOrder[k][l]*xZeroOrder[k][l])
            Ix[i][j]=s
            Iy[i][j]=t
    
    #Ix^2, Iy^2 and Ix.Iy are calculated
    ixsquare=np.square(Ix)
    iysquare=np.square(Iy)
    ixiy=np.copy(f)
    for i in range(0,len(f)):
        for j in range(0,len(f[0])):
            ixiy[i][j] = Ix[i][j] * Iy[i][j]
    
    #Matrix is formed in the form of [[Ix^2  Ix.Iy]
    #                                [Ix.Iy  Iy^2]]
    #eigen values are calculated for each pixel using this matrix A
    #minimum of the two eigen values for each pixel is put into another array
    # which is eigg here.
    eigg=np.copy(f)
    for i in range(0,len(f)):
        for j in range(0,len(f[0])):
            A=[[ixsquare[i][j], ixiy[i][j]],[ixiy[i][j],iysquare[i][j]]]
            eigg[i][j]=np.min(np.linalg.eigvals(A))
    
    #find the local maxima in each window and if the pixel is less than maxVal, make it zero.
    #lis is a list to keep the elements of the window to find the maxima for it.
    #eigmax is another array to keep track of the eigen values which are greater than maxVal.
    lis=[]
    eigmax=np.copy(eigg)
    for i in range(3,len(f)-4):
        for j in range(3,len(f[0])-4):
            for k in range(i-3,i+4):
                for l in range(j-3,j+4):
                    lis.append(eigg[k][l])
            maxi=max(lis)
            if(maxi>eigg[i][j]):
                eigmax[i][j]=0        
            lis=[]
    
    #Choose the number of top K eigens to track, in K.
    #In the image, the pixels at these points will be set to RED(255,0,0) to indicate the feature points.
    K=8
    new_pointsx, new_pointsy = np.unravel_index(eigmax.flatten().argsort()[-K:], eigmax.shape)
    for i,j in zip(new_pointsx,new_pointsy):
        m[i][j][0]=255.0
        m[i][j][1]=0.0
        m[i][j][2]=0.0
    
    #saving images
    imageio.imwrite('/Users/saisrinijasakinala/Desktop/Cv project 2/output/moon_output_frames/moon0.png',m)
    imageio.imwrite('/Users/saisrinijasakinala/Desktop/Cv project 2/output/moon_kalman_frames/moon0.png',m)
    
    #copying variables for using in KalmanTrack function
    xptsK=new_pointsx.copy()
    yptsK=new_pointsy.copy()
    
    #setting P0 for K points in the image
    P0_=np.array([[9,0,0,0],[0,9,0,0],[0,0,25,0],[0,0,0,25]],dtype=float)
    
    #making a list of P0 matrices for K points
    listP0=[]
    for i in range(K):
        listP0.append(P0_)
    v1=0
    v2=0
    
    #iterating over files 
    for files in range(1,174):
        s='/Users/saisrinijasakinala/Desktop/Cv project 2/input/moon_frames/'+str(files-1)+'.png'
        f=imageio.imread(s,as_gray=True)
        s='/Users/saisrinijasakinala/Desktop/Cv project 2/input/moon_frames/'+str(files)+'.png'
        f2=imageio.imread(s,as_gray=True)
        m2=imageio.imread(s)
        m3=m2.copy()
        new_pointsx,new_pointsy=featureTrack(files,f,f2,m2,new_pointsx,new_pointsy)
        v1,v2,listP0,xptsK,yptsK=KalmanTrack(files,f,f2,m3,xptsK,yptsK,new_pointsx,new_pointsy,listP0,v1,v2)
    
    
