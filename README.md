# ENPM673 Project 4

Justin Albrecht and Brian Bock

This project implements the Lucas Kanade optical flow tracker to track an object as it moves through a video.

## How to Run

Clone the entire directory. Open a new terminal window and navigate to the `Code` directory. Type `python LKtracker.py`. If you have additional versions of python installed you may need to run `python3 LKtracker.py` instead.

By default, the program uses a predefined region of interest to define the template from the first frame. If you would like to select your own ROI, you may do so by toggling `newROI` to `True` at the top of the main function in `LKtracker.py`.

You can change the dataset being tested by changing the value of `dataset` at the top of the main function in `LKtracker.py`.

## Dependencies
	Python 3 (version 3.8.1)
	cv2 (version 4.0.0)
	numpy



## Videos

Bolt - https://youtu.be/0Zc1G8rSu7A

Baby - https://youtu.be/HgPUPXOXnho

Car - https://youtu.be/XbIi8FRZqsc

Car (histogram equalization) - https://youtu.be/PkEDOV5CIa4

Car (two rectangles) - https://youtu.be/EfyFZgLkukU


## Additional Reading

https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf

https://www.ri.cmu.edu/project/lucas-kanade-20-years-on/