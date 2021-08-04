import os
import cv2
import random
import numpy as np

class ExtraFunctions():
    
    def __init__(self, cropped_path = "/home/dylan/Videos/", count=0):
        self.cropped_path = cropped_path
        self.count = 0

    # function for cropping each detection and saving as new image
    def crop_objects(self, image, boxes):

        # Get bbox values
        # Only for 1 target
        y1, x1, y2, x2 = boxes[0]

        # Crop based on detection (take an additional 5 pixels around all edges)
        cropped_image = image[int(y1)-5:int(y2)+5, int(x1)-5:int(x2)+5]

        if self.count<10:
            count_str = f"000{self.count}"
        elif self.count >=10 and self.count <100:
            count_str = f"00{self.count}"
        elif self.count >=100 and self.count <1000:
            count_str = f"0{self.count}"
        else:
            count_str = f"{self.count}"

        # construct image name and join it to path for saving crop properly
        image_name = "target" + '_c006'+ '_' + count_str + '.jpg'
        image_path = os.path.join(self.cropped_path, image_name)
        
        # save image
        cv2.imwrite(image_path, cropped_image)

    def update(self):
        self.count += 1