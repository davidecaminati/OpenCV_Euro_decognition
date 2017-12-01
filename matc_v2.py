# Davide Caminati 2017


# export DISPLAY=:0.0   # if you are on terminal

import cv2
import numpy as np
import time

cap = cv2.VideoCapture(-1)  # UDOO = -1 
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)


template_50_1 = cv2.imread('50_f.png',0)
template_50_2 = cv2.imread('50_r.png',0)
template_50_3 = cv2.imread('50_f_180.png',0)
template_50_4 = cv2.imread('50_r_180.png',0)

template_100_1 = cv2.imread('100_f.png',0)
template_100_2 = cv2.imread('100_r.png',0)
template_100_3 = cv2.imread('100_f_180.png',0)
template_100_4 = cv2.imread('100_r_180.png',0)

template_20_1 = cv2.imread('20_f.png',0)
template_20_2 = cv2.imread('20_r.png',0)
template_20_3 = cv2.imread('20_f_180.png',0)
template_20_4 = cv2.imread('20_r_180.png',0)

template_50 = [template_50_1,template_50_2,template_50_3,template_50_4]
template_20 = [template_20_1,template_20_2,template_20_3,template_20_4]
template_100 = [template_100_1,template_100_2,template_20_3,template_100_4]
#50_r

match = 0
while True:
	cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
	cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
	ret, frame = cap.read()
	if not ret:
	  continue
	img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#img_gray = cv2.blur(img_gray, (3,3))
	threshold = 0.75
	# 50 template
	for t_50 in template_50:
		res = cv2.matchTemplate(img_gray,t_50,cv2.TM_CCOEFF_NORMED)
		w, h = t_50.shape[::-1]
		loc = np.where( res >= threshold)
		for pt in zip(*loc[::-1]):
			cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
			print "50"
			break
	# 20 template
	for t_20 in template_20:
		res = cv2.matchTemplate(img_gray,t_20,cv2.TM_CCOEFF_NORMED)
		w, h = t_20.shape[::-1]
		loc = np.where( res >= threshold)
		for pt in zip(*loc[::-1]):
			cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
			print "20"
			break
	# 100 template
	for t_100 in template_100:
		res = cv2.matchTemplate(img_gray,t_100,cv2.TM_CCOEFF_NORMED)
		w, h = t_100.shape[::-1]
		loc = np.where( res >= threshold)
		for pt in zip(*loc[::-1]):
			cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
			print "100"
			break
	
	#cv2.imwrite('res.png',frame)
	#time.sleep(1000)

	cv2.cv.MoveWindow("ATM", 0, 0)  #place window upper left corner
	cv2.imshow("ATM", frame)
	# Wait for the magic key
	keypress = cv2.waitKey(1) & 0xFF
	if keypress == ord('q') :
		break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

