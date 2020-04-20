import cv2
import numpy as np
import os

def affineLKtracker(I,T,rect,p):
    x,y,w,h=rect
    # I is a grayscale image of the current frame
    # T is the template image in grayscale
    # rect defines the ROI for the template in the first frame
    # p is a list of the previous values that define the affine warp

    thresh = .008
    count = 0
    cnt = 0

<<<<<<< HEAD
    # while True:
    for i in range(5):
=======
    for i in range(1):
>>>>>>> origin/master
        W = np.float32([[1+p[0],p[2],p[4]],[p[1],1+p[3],p[5]]])

        # Step 1: Warp I using W to get I_w
        I_w = cv2.warpAffine(I, W, (w, h))
        cv2.imshow('I_w',makeImage(I_w))
        cv2.waitKey(1)

        # Step 2: Compute the error image T - I_w (err)
        diff= np.subtract(T,I_w)

        # Step 3a: Compute the gradients of I (I_x, I_y)
        ksize=3
        Ix = cv2.Sobel(I,cv2.CV_64F,1,0,ksize=ksize)
        # Ix = np.interp(Ix, (-ksize,ksize), (-1, 1))
        Iy = cv2.Sobel(I,cv2.CV_64F,0,1,ksize=ksize)
        # Iy = np.interp(Iy, (-ksize,ksize), (-1, 1))

        # Step 3b: Warp I_x and I_y using W
        Ix_warped = cv2.warpAffine(Ix, W, (w, h))
        Iy_warped = cv2.warpAffine(Iy, W, (w, h))

        # Step 4-5: Compute the steepest descent images delI*dW/dp (sdi)
        H=np.zeros((6,6))
        sdi = []

        sdi_images = []
        for k in range(6):
            sdi_images.append(np.zeros((h,w)))
     
        for i in range(0,w):
            sdi_col = []
            for j in range(0,h):
                J=np.array([[i,0,j,0,1,0],[0,i,0,j,0,1]])
                
                grad=np.array([[(Ix_warped[j,i]),(Iy_warped[j,i])]])

                sdi_vals = np.dot(grad,J)
                sdi_col.append(sdi_vals)

                for k in range(6):
                    sdi_images[k][j,i] = sdi_vals.T[k]

                # Step 6: Compute the Hessian matrix (H)
                H+=np.dot(sdi_col[-1].T,sdi_col[-1])
            
            sdi.append(sdi_col)

        # for k in range(5):
        #     if k == 0:
        #         comb = np.concatenate([sdi_images[k],sdi_images[k+1]],axis=1)
        #     else:
        #         comb = np.concatenate([comb,sdi_images[k+1]],axis=1)

        # cv2.imshow('SDI',makeImage2(comb,-1,1))
        # cv2.waitKey(0)        

        # Step 7: Compute Sum_x(SDI'*err)
        pixel_sum = np.zeros((1,6))
        for i in range(w):
            for j in range(h):
                pixel_sum += ((sdi[i][j])*diff[j,i])

        # Step 8: Compute delta p : dp = H_inv * Sum_x(SDI'*err)
        delta_p = np.linalg.inv(H).dot(pixel_sum.T)

        p_sum = 0
        for i,p_val in enumerate(delta_p):
            if i == 4:
                p_val/=I.shape[1]
            if i == 5:
                p_val/= I.shape[0]

            p_sum += abs(p_val)


        # Step 9: Update p_prev
        p = p + delta_p.T[0]
        p = [float(i) for i in p]

<<<<<<< HEAD
        if p_sum < thresh:
            cnt += 1

        # if p_sum > 0.5:
        #     break

        # if cnt >= 3:
        #     break
=======
    W = np.float32([[1+p[0],p[2],p[4]],[p[1],1+p[3],p[5]]])
    I_w = cv2.warpAffine(I, W, (w, h))
    cv2.imshow('I_w',makeImage(I_w))
    cv2.waitKey(1)
>>>>>>> origin/master

        if count == 20:
            break

        count += 1

    return p,count


def warpROI(p,rect):
    xs,ys,w,h=rect
    W = np.float32([[1+p[0],p[2],p[4]],[p[1],1+p[3],p[5]],[0,0,1]])

    W_inv = np.linalg.inv(W)
    
    # corners = [(xs,ys),(xs+w,ys),(xs+w,ys+h),(xs,ys+h)]
    corners = [(0,0),(w,0),(w,h),(0,h)]

    new_corners = []
    for x,y in corners:
        new_x = W_inv[0,0]*x+W_inv[0,1]*y+W_inv[0,2]
        new_y = W_inv[1,0]*x+W_inv[1,1]*y+W_inv[1,2]
        new_corners.append((int(new_x),int(new_y)))

    return new_corners


def makeImage(arr):
    img = np.interp(arr, (0,1), (0, 255))
    img = np.uint8(img)

    return img

def makeImage2(arr,a,b):
    img = np.interp(arr, (a,b), (0, 255))
    img = np.uint8(img)

<<<<<<< HEAD
    return img


def main():
    dataset='Car' #'Baby', "Bolt", or "Car"
    newROI=True # Toggle this to True if you want to reselect the ROI for this dataset
    writeToVideo = True
    show = True
=======
def drawROI(frame,roi_image):
    img2gray = cv2.cvtColor(roi_image,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    roi_black = cv2.bitwise_and(frame,frame, mask = mask)
    # img2_fg = cv2.bitwise_and(roi_image,roi_image, mask = mask_inv)
    # cv2.imshow("img2fg",img2_fg)

    tracked_image=cv2.add(roi_black,roi_image)

    return tracked_image



def main():
    dataset='Bolt' #'Baby', "Bolt", or "Car"
    newROI=False # Toggle this to True if you want to reselect the ROI for this dataset
>>>>>>> origin/master

    ROIs={"Baby":(158,71,59,77),"Bolt":(270,77,39,66),"Car":(73,53,104,89)} # Dataset:(x,y,w,h)
    frame_total={"Baby":113,"Bolt":293,"Car":659}
    frame_rates = {"Baby":10,"Bolt":20,"Car":10}
    # Get ROI for Template - Draw the bounding box for the template image (first frame)
    # Get first frame
    filepath='../media/'+dataset+'/img/0001.jpg'
    # filepath='../media/'+dataset+'/img/Picture1.png'
    frame=cv2.imread(filepath)
    if newROI:
        cv2.imshow('Frame',frame)
        (x,y,w,h) = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=False)

        cv2.destroyWindow('Frame')
    
    else:
        x,y,w,h=ROIs[dataset]

    color_template = frame[y:y+h,x:x+w]
    rect=(x,y,w,h)

    template = cv2.cvtColor(color_template, cv2.COLOR_BGR2GRAY)

    T = np.float32(template)/255

    p=[0,0,0,0,-x,-y]

<<<<<<< HEAD
    if writeToVideo:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_name = '../output/'+dataset+'_output.mp4'
        fps_out = frame_rates[dataset]

        if os.path.exists(video_name):
            os.remove(video_name)
=======
    blank = np.zeros((frame.shape[0],frame.shape[1],3),'uint8')
    roi_temp = cv2.rectangle(blank,(x,y),(x+w,y+h),(0,255,0),2)
>>>>>>> origin/master

        out = cv2.VideoWriter(video_name,fourcc,fps_out,(frame.shape[1],frame.shape[0]))


    # for frame_num in range (2, frame_total[dataset]+1):
    for frame_num in range (2, 50):
        img_name=('0000'+str(frame_num))[-4:]+'.jpg'
        filepath='../media/'+dataset+'/img/'+img_name

        color_frame=cv2.imread(filepath)
        gray_frame=cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        I = np.float32(gray_frame)/255
     
        p,count = affineLKtracker(I,T,rect,p)

        print('Frame ' + str(frame_num-1) + ' of ' + str(frame_total[dataset]-1) + ' converged in ' + str(count) + ' iterations')
        
        # Draw the new ROI
        corners = warpROI(p,rect)

<<<<<<< HEAD
        for i in range(-1,3):
            cv2.line(color_frame,corners[i],corners[i+1],(0,255,0),2)
=======
        tracked_image=drawROI(color_frame,roi)
>>>>>>> origin/master

        if show:
            cv2.imshow('Tracked Image',color_frame)
            cv2.waitKey(1)

        if writeToVideo:
            out.write(color_frame)

<<<<<<< HEAD
    if writeToVideo:
        out.release()
=======
        cv2.imshow('Tracked Image',tracked_image)
        cv2.waitKey(5)
>>>>>>> origin/master


if __name__ == '__main__':
    main()

    


