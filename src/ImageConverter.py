import rospy
from sensor_msgs.msg import Image
from mask_rcnn_ros.msg import Bbox_values
import numpy as np
import cv2

class ImageConverter:

    def __init__(self):
        self.image_pub = rospy.Publisher("bbox_output",Bbox_values, queue_size=10)
        self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)
        self.cv_img = None

    def callback(self,data):
        width = data.width
        height = data.height
        channels = int(len(data.data) / (width * height))

        encoding = None
        if data.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif data.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif data.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        # Have to use a copy as the original image is read-only which will result in an error when
        # trying to modify the image
        self.cv_img = np.ndarray(shape=(data.height, data.width, channels), dtype=encoding, buffer=data.data).copy()

        if data.encoding.lower() == 'mono8':
            self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_RGB2GRAY)
        else:
            self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_RGB2BGR)