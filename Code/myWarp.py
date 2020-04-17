import cv2
import numpy as np
# from skimage.metrics import structural_similarity

def myWarp(image,p,rect):
	# print(p)
	p1,p2,p3,p4,p5,p6=p
	# h,w=image.shape
	x,y,w,h=rect
	M=np.float32([[p1,p2,p3],[p4,p5,p6]])

	warped_img=cv2.warpAffine(image, M, (w, h))

	# cv2.imshow("Warped",warped_img)
	# cv2.waitKey(0)

	return warped_img

def warpROI(p,rect):
	p1,p2,p3,p4,p5,p6=p
	M=np.float32([[p1,p2,p3],[p4,p5,p6]])
	x,y,w,h=rect
	corners = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]

	new_corners = []
	for x,y in corners:
		new_x = M[0,0]*x+M[0,1]*y
		new_y = M[1,0]*x+M[1,1]*y
		new_corners.append((int(new_x),int(new_y)))

	return new_corners
	

def computeError(prev_image,new_image):
	diff= np.subtract(prev_image,new_image,dtype='float32')

	diff = np.interp(diff, (-255,255), (-1, 1))

	return diff


def myGradients(image):
	ksize=3
	Ix = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=ksize)
	scaled_Ix=np.interp(Ix, (-255*ksize,255*ksize), (-1, 1))

	Iy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=ksize)
	scaled_Iy=np.interp(Iy, (-255*ksize,255*ksize), (-1, 1))

	return scaled_Ix, scaled_Iy


def makeImage(arr):
	img = np.interp(arr, (-1,1), (0, 255))
	img = np.uint8(img)

	return img





