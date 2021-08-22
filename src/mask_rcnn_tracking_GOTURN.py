#!/usr/bin/env python3

from FPS import *
from ObjectTracker import *
from ImageConverter import *
from ExtraFunctions import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import cv2

from Mask_RCNN.scripts.visualize_cv2 import model, class_dict, class_names
from tensorflow.python.client import device_lib
import numpy as np

video_path   = "/home/dylan/catkin_ws/src/mask_rcnn_ros/videos/video_2.avi"
output_path   = "/home/dylan/catkin_ws/src/mask_rcnn_ros/videos/video_2_goturn_results_3.mp4"


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        
        # Make detection bbox slightly bigger so that we can compare with tracked bbox
        image = cv2.rectangle(image, (x1-5, y1-5), (x2+5, y2+5), color, 2)
        image = cv2.putText(
            image, caption, (x1-10, y1-25), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image

def main(args):

    print(device_lib.list_local_devices())

    # Create tracker
    # Have to change dir so that the GOTURN caffemodel and prototxt files can be read during creation of tracker
    print(f"Current dir: {os.getcwd()}")
    os.chdir("./GOTURN/")

    print(f"Current dir: {os.getcwd()}")
    tracker = cv2.TrackerGOTURN_create()
    
    os.chdir("../")
    print(f"Current dir: {os.getcwd()}")

    total_fps = FPS()
    detection_fps = FPS()
    tracking_fps = FPS()

    extra = ExtraFunctions(cropped_path = "/home/dylan/Videos/image_train/")

    ##### Have to initiate to read frame by frame from video #####
    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    detected = 0

    print(f"\nVideo fps: {fps}")
    print(f"Video frame count: {total_frame_count}\n")

    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    #################################################################

    while True:

        total_fps.start()
        # Read a new frame
        ok, frame = vid.read()
        if not ok:
            break
        
        frame_count+=1
        print(f"\nFrame: {frame_count}")

        # Start timer
        timer = cv2.getTickCount()

        if detected == 0 or frame_count%30 == 0:
            print("Start detection...\n")
            results = model.detect([frame], verbose=0)

            r = results[0]
            display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

            if len(r['rois']):
                detected +=1 
                bbox = extra.format_bbox(r['rois'])

                 # To format for object tracking
                bbox=(bbox.x, bbox.y, bbox.w, bbox.h)
                print(f"results: {bbox}")

                # Initialize tracker with first frame and bounding box 
                tracker.init(frame, bbox)

        else:
            tracking_fps.start()
            # Update tracker
            _ , bbox = tracker.update(frame)
            tracking_fps.stop()

        # Draw bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        # Calculate Frames per second (FPS)
        total_fps.stop()

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

        # Display result
        cv2.imshow("Output", frame)
    
        out.write(frame)
        
        print(f"Time taken to track: {tracking_fps.elapsed():.2f} ms\n")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break   
    
    vid.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Average total fps: {total_fps.getFPS():.2f} fps")

if __name__ == '__main__':
    main(sys.argv)