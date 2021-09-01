#!/usr/bin/env python3

from functions.FPS import *
from functions.ImageConverter import *
from functions.ExtraFunctions import *

import sys
import rospy
import cv2

from Mask_RCNN.scripts.visualize_cv2 import model
from tensorflow.python.client import device_lib
import numpy as np

import threading as t
from queue import Queue, LifoQueue
import time

##################### Set global variables #####################
# Choose tracker type
tracker = cv2.TrackerCSRT_create()
tracker_init = False # Shared variable will result in readers-writers problem
read_write_lock = t.Lock()

extra = ExtraFunctions(cropped_path = "/home/dylan/Videos/image_train/")

# Queues are thread-safe
frame_queue = LifoQueue(maxsize=1)
detect_queue = Queue(maxsize=10)
display_queue = Queue(maxsize=10)

# count is a global variable in the shared memory between threads.
# Therefore, have to use locks to prevent race condition
count = 0 
count_lock = t.Lock()
#################################################################

def draw_tracked_bbox(curr_frame, bbox, detection=False):
    # Draw tracked bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

    if detection == False:
        cv2.rectangle(curr_frame, p1, p2, (255,0,0), 2)
    else:
        cv2.rectangle(curr_frame, (int(bbox[0])-5, int(bbox[1])-5), (int(bbox[0] + bbox[2])+5, int(bbox[1] + bbox[3])+5), (100, 255, 0), 3)

def verify_tracker(bbox_detected, bbox_tracked):

    # Determine the (x, y)-coordinates of the Intersection rectangle
    xA = max(bbox_detected[0], bbox_tracked[0])
    yA = max(bbox_detected[1], bbox_tracked[1])
    xB = min(bbox_detected[0] + bbox_detected[2], bbox_tracked[0] + bbox_tracked[2])
    yB = min(bbox_detected[1] + bbox_detected[3], bbox_tracked[1] + bbox_tracked[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the detected and tracked bboxes
    bbox_detectedArea = (bbox_detected[2] + 1) * (bbox_detected[3] + 1)
    bbox_trackedArea = (bbox_tracked[2] + 1) * (bbox_tracked[3]  + 1)

    # Compute the IOU by taking the intersection area
    # divided by the sum of detected + tracked areas - the interesection area
    iou = round(interArea / float(bbox_detectedArea + bbox_trackedArea - interArea),2)

    return iou

def display_frames():
    print(f"\n[{t.current_thread().name}] Display thread starting...")

    while not rospy.is_shutdown():

        display = display_queue.get()
        start = time.perf_counter()

        if display is None:
            break
        
        frame = display[0]
        frame_num = display[1]

        # Display result
        cv2.imshow("Output", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        stop = time.perf_counter()
        print(f"[{t.current_thread().name}] Time taken to display frame {frame_num}: {(stop-start)*1000:.2f}ms\n")

    print(f"[{t.current_thread().name}] --- End of Display thread ---\n")

def object_tracking():
    global count, tracker_init
    print(f"[{t.current_thread().name}] Tracking thread starting...")
    total_fps = FPS()

    while not rospy.is_shutdown():
        
        tracker_wrong = False

        # Get curr frame
        curr_frame = frame_queue.get()

        if curr_frame is None:
            display_queue.put(None)
            break 
        
        total_fps.start()
        start = time.perf_counter()

        with count_lock:
            if count >= 100000:
                count = 0

            count+=1
            count_copy = count

        print(f"[{t.current_thread().name}] Got frame from frame_queue - {count_copy}")
        print(f'\n[{t.current_thread().name}] tracker_init: {tracker_init}')
        
        # Run tracking if tracker has been initiated before
        if tracker_init == True:

            # If no detection received in the detect_queue, continue with tracking
            if detect_queue.qsize() == 0:
                
                # Update Tracker
                _ , bbox = tracker.update(curr_frame)

            # If received a detection in the detect_queue, use the detection results
            # to verify if the tracked bbox is still correct
            else:
                detected_results = detect_queue.get() 
            
                # Ends if no more detection
                if detected_results is None:
                    display_queue.put(None)
                    break 
                
                detected_frame = detected_results[0]
                bbox = detected_results[1]
                detected_num = detected_results[2]
                bbox_tracked = detected_results[3]

                # Check if got valid results
                if bbox_tracked is not False:

                    print(f"[{t.current_thread().name}] Detected Frame {detected_num} results: {bbox}")
                    print(f"[{t.current_thread().name}] Tracked Frame {detected_num} results: {bbox_tracked}")

                    # Check IOU >=0.5
                    iou = verify_tracker(bbox, bbox_tracked)
                    print(f"[{t.current_thread().name}] IOU: {iou}")

                    if iou >= 0.5:
                        print(f"[{t.current_thread().name}] Tracker is correct")

                        # Continue updating tracker
                        _ , bbox = tracker.update(curr_frame)

                    else:
                        print(f"[{t.current_thread().name}] Tracker is wrong!")

                        with read_write_lock:
                            tracker_init = False

                        tracker_wrong = True

                else:
                    print(f"[{t.current_thread().name}] Tracker not re-initialized...")
                    
                    # Update Tracker
                    _ , bbox = tracker.update(curr_frame)

            total_fps.stop()

            if tracker_wrong == False:
                cv2.putText(curr_frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
                # Draw tracked bounding box
                draw_tracked_bbox(curr_frame, bbox)
            else:
                cv2.putText(curr_frame, f"FPS: -.--", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(curr_frame, f"Tracker Is Wrong", (7,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            # Display result
            display_queue.put([curr_frame, count_copy])

        else:

            if detect_queue.qsize() == 0:

                display_queue.put([curr_frame, count_copy])

                total_fps.stop()

                # Display FPS on detected frame
                cv2.putText(curr_frame, f"FPS: -.--", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
                
                # Display result
                display_queue.put([curr_frame, count_copy])

            else:

                detected_results = detect_queue.get() 

                # Ends if no more detection
                if detected_results is None:
                    display_queue.put(None)
                    break
 
                detected_frame = detected_results[0]
                bbox = detected_results[1]
                detected_num = detected_results[2]  

                with read_write_lock:
                # Init tracker
                    tracker.init(detected_frame, bbox)
                    tracker_init = True
                
                print(f"[{t.current_thread().name}] Tracker initialized!")
                
                # Draw tracked bounding box
                draw_tracked_bbox(detected_frame, bbox, detection=True)
            
                total_fps.stop()

                # Display FPS on detected frame
                cv2.putText(detected_frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
                
                # Display result
                display_queue.put([detected_frame, detected_num])

        stop = time.perf_counter()
        print(f"[{t.current_thread().name}] Time taken to track: {(stop-start)*1000:.2f}ms\n")

    print(f"[{t.current_thread().name}] --- End of Tracking thread ---\n")

def main(args):

    print(device_lib.list_local_devices())
    rospy.init_node('drone_detector')
    ic = ImageConverter(frame_queue)

    global count, tracker_init

    tracking_thread = t.Thread(target=object_tracking, name="Tracker")
    display_thread = t.Thread(target=display_frames, name="Display")

    tracking_thread.start()
    display_thread.start()

    while not rospy.is_shutdown():

        curr_frame = frame_queue.get()

        if curr_frame is None:
            detect_queue.put(None)
            display_queue.put(None)
            break

        start = time.perf_counter()

        with count_lock:
            count+=1
            count_copy = count

        print(f"[{t.current_thread().name}] Taking from frame_queue - Frame {count_copy}")

        # Have to always keep tracker updated so that we can compare detection and tracked results
        # Placed this before undergoing detection as detection is slow where tracker thread may have updated
        # the tracker for a few frames already

        with read_write_lock: 

            if tracker_init == True:
                _ , bbox_tracked = tracker.update(curr_frame)

            else:
                bbox_tracked = False
        
        # Mask RCNN detection
        results = model.detect([curr_frame], verbose=0)
        r = results[0]

        if len(r['rois']):

            print(f"[{t.current_thread().name}] Got valid detection")

            bbox_detected = extra.format_bbox(r['rois'])

            # To format for object tracking
            bbox_detected=(bbox_detected.x, bbox_detected.y, bbox_detected.w, bbox_detected.h)
            print(f"[{t.current_thread().name}] Frame {count_copy} results: {bbox_detected}")

            detect_queue.put([curr_frame, bbox_detected, count_copy, bbox_tracked])
        
        else:
            print(f"[{t.current_thread().name}] Failed to detect!")

        print(f"[{t.current_thread().name}] Put frame {count_copy} in detect_queue")
        
        stop = time.perf_counter()
        print(f"[{t.current_thread().name}] Time taken to detect: {(stop-start)*1000:.2f}ms\n")   

    tracking_thread.join()
    display_thread.join()

    print(f'[{t.current_thread().name}] frame_queue: {frame_queue.qsize()}\n')
    print(f'[{t.current_thread().name}] detect_queue: {detect_queue.qsize()}')
    print(f'[{t.current_thread().name}] display_queue: {display_queue.qsize()}')
    print(f"[{t.current_thread().name}] ---End---")
 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)