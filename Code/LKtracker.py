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

def main():
    # load the video 
    # Draw the bounding box for the template image (first frame)
    # for subsequent frame in video:
        # p = affineLKtracker(I,T,rect,p_prev)

