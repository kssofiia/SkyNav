import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from math import pi, tan
from cv_bridge import CvBridge
from sensor_msgs.msg import Range
rospy.init_node("sdfghjk")


# frame = CvBridge.imgmsg_to_cv2(rospy.wait_for_message('main_camera/image_raw', Image), 'bgr8')
img = cv2.imread("map.jpg")
img = cv2.resize(img,(700,700))[::-1,::-1]
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# sr = cv2.imread('bg.jpg')
bridge = CvBridge()
Ly = [0]
Lx = [0]
fon = np.zeros((700,700))
print("go")
while True:
    dist = rospy.wait_for_message('rangefinder/range', Range).range
    frame = bridge.imgmsg_to_cv2(rospy.wait_for_message('main_camera/image_raw', Image), 'bgr8')
    m = len(frame)//2
    k = tan(pi/4)*2*dist/m
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame,(int(700*dist/100),int(700*dist/100)))
    fon[0:int(700*dist/100),0:int(700*dist/100)]=frame[:,:]
    #cv2.imshow('Camera',frame)
    #cv2.imshow('Original',img)
    ((x,y),K) = cv2.phaseCorrelate(img.astype(np.float32),fon.astype(np.float32))
    '''kx = max(-100,x-Lx[len(Lx)-2])
    ky = max(-100,y-Ly[len(Ly)-2])
    kx = min(100,kx)
    ky = min(100,ky)'''
    Lx.append((x)*k)
    Ly.append((y)*k*2)
    if len(Lx)>=15 and len(Ly)>=15:
        Lx=[sum(Lx)/len(Lx)]
        Ly=[sum(Ly)/len(Ly)]
        print(Lx[0],Ly[0])
	
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows() 
