import cv2
import numpy

def affineLKtracker(I,T,rect,p_prev):
    # I is a grayscale image of the current frame
    # T is the template image in grayscale
    # rect is the bounding box tht marks the template region in tmp 
        # tuple of corners c1,c2 with their associated x and y values
        # ((c1x,c1y),(c2x,c2y))
    # p_prev are the parameters of the previous warp
        # list of parameters [p1,p2,p3,p4,p5,p6]

    #thresh = 
    #while dp < thresh:

        # Step 1a: Define an affine warp W using p_prev

        # Step 1b: Warp I using W to get I_w

        # Step 2: Compute the error image T - I_w (err)

        # Step 3a: Compute the gradients of I (I_x, I_y)

        # Step 3b: Warp I_x and I_y using W

        # Step 4: Evaluate the Jacobian dW/dp at (x;p) (J)

        # Step 5: Compute the steepest descent images delI*J (SDI)

        # Step 6: Compute the Hessian matrix (H)

        # Step 7: Compute Sum_x(SDI'*err)

        # Step 8: Compute delta p : dp = H_inv * Sum_x(SDI'*err)

        # Step 9: Update p_prev

    #p_new = p_prev 

    return p_new

if __name__ == '__main__':

    dataset='Baby' #'Baby', "Bolt", or "Car"
    newROI=False # Toggle this to True if you want to reselect the ROI for this dataset

    ROIs={"Baby":(158,71,59,77),"Bolt":(270,77,39,66),"Car":(73,53,104,89)} # Dataset:(x,y,w,h)
    frame_total={"Baby":113,"Bolt":293,"Car":659}


    # Get ROI for Template - Draw the bounding box for the template image (first frame)
    # Get first frame
    filepath='../media/'+dataset+'/img/0001.jpg'
    frame=cv2.imread(filepath)
    if newROI:
        cv2.imshow('Frame',frame)
        (x,y,w,h) = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=False)
    
    else:
        x=ROIs[dataset][0]
        y=ROIs[dataset][1]
        w=ROIs[dataset][2]
        h=ROIs[dataset][3]
    color_template = frame[y:y+h,x:x+w]
    rect=((x,y),(x+w,y+h))
    template = cv2.cvtColor(color_template, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Template",template)
    cv2.waitKey(0)



    # load the video 
    for frame_num in range (1, frame_total[dataset]+1):
        img_name=('0000'+str(frame_num))[-4:]+'.jpg'
        filepath='../media/'+dataset+'/img/'+img_name
        # print(filepath)
        color_frame=cv2.imread(filepath)
        gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Frame",gray_frame)
        cv2.waitKey(1)


        

    # for subsequent frame in video:
        # p = affineLKtracker(I,T,rect,p_prev)

