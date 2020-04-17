import cv2
import numpy as np

from myWarp import*

def affineLKtracker(I,T,rect,p_prev):
    x,y,w,h=rect
    # I is a grayscale image of the current frame
    # T is the template image in grayscale
    # rect is the bounding box tht marks the template region in tmp 
        # tuple of corners c1,c2 with their associated x and y values
        # ((c1x,c1y),(c2x,c2y))
    # p_prev are the parameters of the previous warp
        # list of parameters [p1,p2,p3,p4,p5,p6]

    # thresh = 
    # while dp < thresh:
    for i in range(10):
        # Step 1: Warp I using W to get I_w
        I_w=myWarp(I,p_prev,rect)

        # Step 2: Compute the error image T - I_w (err)
        diff=computeError(T,I_w)

        # Step 3a: Compute the gradients of I (I_x, I_y)
        Ix, Iy = myGradients(I)

        # Step 3b: Warp I_x and I_y using W
        Ix_warped=myWarp(Ix,p_prev,rect)
        Iy_warped=myWarp(Iy,p_prev,rect)

        # Step 4-5: Compute the steepest descent images delI*dW/dp (sdi)
        H=np.zeros((6,6))
        sdi = []
     
        for i in range(0,w):
            sdi_col = []
            for j in range(0,h):
                ind_x = i/w
                ind_y = j/h
                J=np.array([[ind_x,0,ind_y,0,1,0],[0,ind_x,0,ind_y,0,1]])
                
                grad=np.array([[(Ix_warped[j,i]),(Iy_warped[j,i])]])

                sdi_col.append(np.dot(grad,J))

                # Step 6: Compute the Hessian matrix (H)
                H+=np.dot(np.transpose(sdi_col[-1]),sdi_col[-1])
            sdi.append(sdi_col)

        # Step 7: Compute Sum_x(SDI'*err)
        pixel_sum = np.zeros((1,6))
        for i in range(w):
            for j in range(h):
                pixel_sum += ((sdi[i][j])*diff[j,i])/(100)

        # Step 8: Compute delta p : dp = H_inv * Sum_x(SDI'*err)
        delta_p = np.linalg.inv(H).dot(pixel_sum.T)

        # Step 9: Update p_prev
        p_prev = p_prev + delta_p.T[0]
        p_prev = [float(i) for i in p_prev]


    p_new = p_prev
    print(p_new)

    return p_new

if __name__ == '__main__':

    dataset='Baby' #'Baby', "Bolt", or "Car"
    newROI=False # Toggle this to True if you want to reselect the ROI for this dataset

    ROIs={"Baby":(158,71,59,77),"Bolt":(270,77,39,66),"Car":(73,53,104,89)} # Dataset:(x,y,w,h)
    frame_total={"Baby":113,"Bolt":293,"Car":659}


    # Get ROI for Template - Draw the bounding box for the template image (first frame)
    # Get first frame
    filepath='../media/'+dataset+'/img/0001.jpg'
    # filepath='../media/'+dataset+'/img/Picture1.png'
    frame=cv2.imread(filepath)
    if newROI:
        cv2.imshow('Frame',frame)
        (x,y,w,h) = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=False)
    
    else:
        x,y,w,h=ROIs[dataset]

    color_template = frame[y:y+h,x:x+w]
    rect=(x,y,w,h)
    template = cv2.cvtColor(color_template, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Template",template)
    # cv2.waitKey(0)

    p=[1,0,-x,0,1,-y]

    for frame_num in range (2, frame_total[dataset]+1):
    # for frame_num in range(2,3):
        img_name=('0000'+str(frame_num))[-4:]+'.jpg'

        filepath='../media/'+dataset+'/img/'+img_name

        color_frame=cv2.imread(filepath)
        gray_frame=cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
     
        p = affineLKtracker(gray_frame,template,rect,p)
        corners = warpROI(p,rect)
        print(corners)
        
        for i in range(-1,3):
            cv2.line(color_frame,corners[i],corners[i+1],(0,255,0))

        cv2.imshow('Tracked Image',color_frame)
        cv2.waitKey(0)


