# project: Gesture recognition for Perceptual User Interface
# team: MANISH SONI , Aastha tyagi

# organize imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
from pyautogui import press, typewrite, hotkey
import subprocess

# global variables
bg = None
observedIm=None
camera= None
clone= None

#-------------------------------------------------------------------------------
# Function - To find the running average over the background
#-------------------------------------------------------------------------------
def bg_avg(image, accumWeight):
    global bg

    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return
    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)

#-------------------------------------------------------------------------------
# Function - To find the running average over the foreground
#-------------------------------------------------------------------------------
def ob_avg(image, accumWeight):
    global observedIm

    if observedIm is None:
        observedIm = image.copy().astype("float")
        return
    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, observedIm, accumWeight)


#-------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
#-------------------------------------------------------------------------------
def segment(image, backgroundVar, threshold=15):
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(backgroundVar.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (img, contours, hierarchy) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(contours) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)

#-------------------------------------------------------------------------------
# Function - To count the number of fingers in the segmented hand region
#-------------------------------------------------------------------------------
def count(thresholded, segmented):

	(X_coord,Y_coord,radius)=centerAndRadiusOfSegmentedConvexHull(thresholded, segmented)

	# find the circumference of the circle
	circumference = (2 * np.pi * radius)

	# take out the circular region of interest which has 
	# the hand region including palm and the fingers
	circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
	# draw the circular ROI
	cv2.circle(circular_roi, (X_coord, Y_coord), radius, 255, 1)
	
	# take bit-wise AND between thresholded hand using the circular ROI as the mask
	# which gives the cuts obtained using mask on the thresholded hand image
	circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

	# compute the contours in the circular ROI
	(img, contours, hierarchy) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# initalize the finger count in hand segment
	count = 0

	# loop through the contours found
	for contour in contours:
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(contour)
		if ((Y_coord + (Y_coord * 0.25)) > (y + h)) and ((circumference * 0.25) > contour.shape[0]):
			count += 1

	return count
#---------------------------------------------------------------------------------#
        # get the center and radius of convex hull shape in ecludian space
#---------------------------------------------------------------------------------#
def centerAndRadiusOfSegmentedConvexHull(thresholded, segmented):
    # find the convex hull of the segmented hand region
    hullvar = cv2.convexHull(segmented)
    # find the most extreme points in the convex hull
    hull_extreme_top    = tuple(hullvar[hullvar[:, :, 1].argmin()][0])
    hull_extreme_bottom = tuple(hullvar[hullvar[:, :, 1].argmax()][0])
    hull_extreme_left   = tuple(hullvar[hullvar[:, :, 0].argmin()][0])
    hull_extreme_right  = tuple(hullvar[hullvar[:, :, 0].argmax()][0])

    # get the center coordinates of the palm
    X_coord = (hull_extreme_left[0] + hull_extreme_right[0]) // 2
    Y_coord = (hull_extreme_top[1] + hull_extreme_bottom[1]) // 2
    # get the maximum euclidean distance between the center of the convex hull and extreme points of the  hull
    distance = pairwise.euclidean_distances([(X_coord, Y_coord)], Y=[hull_extreme_left, hull_extreme_right, hull_extreme_top, hull_extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]
    
    # consider the radius of the circle considering it as 80% of the max euclidean distance calculated
    radius = int(0.8 * maximum_distance)
    return (X_coord,Y_coord, radius)

#-------------------------------------------------------------------------------
# Function - To get the longest increasing subarray of an array
#-------------------------------------------------------------------------------
def lenOfLongIncSubArr(arr, n) :  
    m = 1 
    l = 1 

    for i in range(1, n) : 
  
        if (arr[i] > arr[i-1]) : 
            l =l + 1 
        else : 
            if (m < l)  : 
                m = l  
      
            l = 1 
    if (m < l) : 
        m = l 
    return m 
#-------------------------------------------------------------------------------
# Function - To get the longest decreasing subarray of an array
#-------------------------------------------------------------------------------
def revlongsubarr(arr,n):
    newlist=arr.copy()
    newlist.reverse()
    return lenOfLongIncSubArr(newlist,n)

    
#-------------------------------------------------------------------------------
# Function - To segment and count the number of fingers in the region of interest
#-------------------------------------------------------------------------------

def countfinger(frame, top, right,bottom, left,num_frames,fingers, frameMode):
    global bg
    global observedIm
  
    accumWeight = 0.5
   
    roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our weighted average model gets calibrated
    if num_frames < 31:
        bg_avg(gray, accumWeight)

        if num_frames == 1:
            print ("Initializing...")
        elif num_frames == 30:
                    print ("initialization successfull...")       
    else:
                #consider the current frame with 0.7 weigh 
        if num_frames%10 != 0:
            if frameMode==0:
                ob_avg(gray, 0.7)
        else:
            hand = None
            
            hand = segment(observedIm.astype("uint8"),bg)

                    # check whether hand region is segmented
            if hand is not None:
                        # if yes, unpack the thresholded image and
                        # segmented region
                (thresholded, segmented) = hand

                        # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                fingers = count(thresholded, segmented)                            
                        # show the thresholded image
                cv2.imshow("Thesholded", thresholded)

            else:
                fingers=0
    return fingers
                
#-------------------------------------------------------------------------------
# Function - To get the direction 
#-------------------------------------------------------------------------------

def direction(frame, top, right,bottom, left,num_frames):
    global bg
    global observedIm
    
    accumWeight = 0.5
    (cX,cY)=(-1,-1)

    roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    #get the initial 30 frames for weighted average background image
    if num_frames < 31:
        bg_avg(gray, accumWeight)

        if num_frames == 1:
            print ("Initializing...")
        elif num_frames == 30:
                    print ("initialization successfull...")       
    else:
                # get the average weighted foregound image
        if num_frames%3 != 0:
                ob_avg(gray, 0.7)

        else:
            hand = None
            hand = segment(observedIm.astype("uint8"),bg)

                    # check whether hand region is segmented
            if hand is not None:
                        # if yes, unpack the thresholded image and
                        # segmented region
                (thresholded, segmented) = hand

                        # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

    

                (cX,cY,r)=centerAndRadiusOfSegmentedConvexHull(thresholded, segmented)

    return (cX,cY)


#-------------------------------------------------------------------------------
# Main function of the program:
# Mode: there are two modes in the program 
#1. for fingers count in hand region for gesture understanding it is a simple 
# algorithm
# 2. for hand gesture direction

# for mode 2 the sensitivity of direction measurement depends upon incCountX 
# and decCountX 
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    mode=1 # please change the mode mode=1/2
    fingers = 0
    num_frames = 0
    coordX=[]
    coordY=[]
    textout="starting"
    #subprocess.call(["/usr/bin/open", "-n", "-a", "/Applications/Microsoft PowerPoint.app"])

    

    # region of interest (ROI) coordinates
    
    calibrated = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame to make mirror view right
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]
        if mode==1:

            # get the ROI
            top, right, bottom, left = 10, 350, 225, 590
            fingers=countfinger(frame, top, right,bottom, left,num_frames,fingers, 0)
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
            textout=str(fingers)

        elif mode==2:
            top, right, bottom = 10, 190, 225
            left =  590
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
            (cX,cY)=direction(frame, top, right,bottom, left,num_frames)
            if cX!= -1 or cY!=-1:
                coordX.append(cX)
                coordY.append(cY)
                cv2.circle(clone,(right+cX,top+cY), 8, (0,0,255), -1)
                # direction depends on the x,y coordinate of aveage image of hand after every 3 frames. if the x count increase then it means
                # hand has moved towards right direction else it is left
                incCountX=lenOfLongIncSubArr(coordX,len(coordX))
                decCountX=revlongsubarr(coordX,len(coordX))
                incCountY=lenOfLongIncSubArr(coordY,len(coordY))
                decCountY=revlongsubarr(coordY,len(coordY))
                
                if incCountX>2:
                    #fingers=111111
                    textout="Right"
                    coordX.clear()
                    coordY.clear()
                    #pyautogui.typewrite(["left"])
                if incCountY>8:
                    textout="Up"
                    coordX.clear()
                    coordY.clear()
                if decCountX>2:
                    textout="Left"
                    coordX.clear()
                    coordY.clear()
                if decCountY>8:
                    textout="Down"
                    coordX.clear()
                    coordY.clear()
            if (num_frames-30)%30 == 0 and num_frames>30:
                coordX.clear()
                coordY.clear()
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)



        # draw the segmented hand
        #cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(clone, textout, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
camera.release()
cv2.destroyAllWindows()





