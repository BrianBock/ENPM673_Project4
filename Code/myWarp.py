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


def computeError(prev_image,new_image):
	# (score, diff) = structural_similarity(prev_image, new_image, full=True)
	# diff = (diff * 255).astype("uint8")
	diff=cv2.absdiff(prev_image,new_image)
	# cv2.imshow("Difference",diff)
	# cv2.waitKey(0)
	return diff


def myGradients(image):
	ksize=3
	Ix = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=ksize)

	newIx=np.interp(Ix, (-255*ksize,255*ksize), (0, 255))

	# abs_sobel64f = np.absolute(Ix)
	Xsobel_8u = np.uint8(newIx)

	Iy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=ksize)
	print(min(Iy.flatten()),max(Iy.flatten()))
	newIy=np.interp(Iy, (-255*ksize,255*ksize), (0, 255))

	# abs_sobel64f = np.absolute(Iy)
	Ysobel_8u = np.uint8(newIy)

	# cv2.imshow("Ix",Xsobel_8u)
	# cv2.imshow("Iy",Ysobel_8u)
	# cv2.waitKey(0)

	return Xsobel_8u, Ysobel_8u




