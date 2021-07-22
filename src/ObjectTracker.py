import numpy as np
import random
import colorsys
import cv2
from Mask_RCNN.scripts.visualize_cv2 import apply_mask

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

class ObjectTracker:

    def __init__(self):
        self.rectangle_colors=(255,0,0)
        self.Text_colors=(255,255,0)

        self.NUM_CLASS = {0: 'BG', 1:'target'}
        self.key_list = list(self.NUM_CLASS.keys()) 
        self.val_list = list(self.NUM_CLASS.values())

        self.max_cosine_distance = 0.7
        self.nn_budget = None

        self.model_filename = '/home/dylan/catkin_ws/src/mask_rcnn_ros/src/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)

    def track_object(self, img, boxes, names, scores, r, fps):
        features = np.array(self.encoder(img, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        self.tracker.predict()
        self.tracker.update(detections)
        fps.stop()
        self.show_tracked_object(img, r, fps)

    def show_tracked_object(self, img, r, fps):
        # Obtain info from the tracks
        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = self.key_list[self.val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
    
        num_classes = len(self.NUM_CLASS)
        image_h, image_w, _ = img.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        #print("hsv_tuples", hsv_tuples)
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        if len(tracked_bboxes) == 1 and np.any(r['masks']):
            bbox = tracked_bboxes[0]
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = self.rectangle_colors if self.rectangle_colors != '' else colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
            mask = r['masks'][:, :, 0]
            masked_image = apply_mask(img, mask, bbox_color)
            
            # put object rectangle
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

            score_str = " {:.2f}".format(score)

            score_str = " "+str(score)

            try:
                label = "{}".format(self.NUM_CLASS[class_ind]) + score_str
            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                    fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(masked_image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(masked_image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, self.Text_colors, bbox_thick, lineType=cv2.LINE_AA)
            
            cv2.putText(masked_image, f"FPS: {fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("Masked Image", masked_image)

        else:
            cv2.putText(img, f"FPS: {fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("Masked Image", img)