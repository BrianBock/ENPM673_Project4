import cv2
import numpy as np
from skimage.metrics import structural_similarity

def myWarp(image,p):
	print(p)
	p1,p2,p3,p4,p5,p6=p
	h,w=image.shape
	M=np.float32([[p1,p2,p3],[p4,p5,p6]])

	warped_img=cv2.warpAffine(image, M, (w, h))

	cv2.imshow("Warped",warped_img)
	cv2.waitKey(0)

	return warped_img


def computeError(prev_image,new_image):
	(score, diff) = structural_similarity(prev_image, new_image, full=True)
	diff = (diff * 255).astype("uint8")
	cv2.imshow("Difference",diff)
	cv2.waitKey(0)
	return diff


def myGradients(image):
	Ix = (cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)*255).astype("uint8")
	Iy = (cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)*255).astype("uint8")

	cv2.imshow("Ix",Ix)
	cv2.imshow("Iy",Iy)
	cv2.waitKey(0)

	return Ix, Iy




