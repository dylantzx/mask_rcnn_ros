#!/usr/bin/env python3

from functions.FPS import *
from functions.ImageConverter import *
from functions.ExtraFunctions import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import cv2

from Mask_RCNN.scripts.visualize_cv2 import model, class_dict, class_names
from tensorflow.python.client import device_lib
import numpy as np

import threading as t
from queue import Queue
import time

##################### Set global variables #####################
# Choose tracker type
tracker = cv2.TrackerCSRT_create()
tracker_init = False # Shared variable will result in readers-writers problem
read_write_lock = t.Lock()

video_path   = "/home/dylan/catkin_ws/src/mask_rcnn_ros/videos/video_3.avi"
output_path   = f"/home/dylan/catkin_ws/src/mask_rcnn_ros/videos/video_3_multithread_CSRT_results_2.mp4"

extra = ExtraFunctions(cropped_path = "/home/dylan/Videos/image_train/")

# Queues are thread-safe
frame_queue = Queue()
detect_queue = Queue()
display_queue = Queue()

# count is a global variable in the shared memory between threads.
# Therefore, have to use locks to prevent race condition
count = 0 
count_lock = t.Lock()
#################################################################

def frame_grabber(vid):

    print(f"[{t.current_thread().name}] Frame grabber thread starting...")
    while True:

        ok, frame = vid.read()
        
        if not ok:
            for i in range(2):
                frame_queue.put(None)
            break

        frame_queue.put(frame)

    print(f"[{t.current_thread().name}] --- End of Frame grabber thread ---\n")

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

def display_frames(out):
    print(f"[{t.current_thread().name}] Display thread starting...")

    while True:
        start = time.perf_counter()
        display = display_queue.get()

        if display is None:
            break
        
        frame = display[0]
        frame_num = display[1]

        # Display result
        cv2.imshow("Output", frame)

        # out.write(frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

        stop = time.perf_counter()
        print(f"[{t.current_thread().name}] Time taken to display frame {frame_num}: {(stop-start)*1000:.2f}ms\n")

    print(f"[{t.current_thread().name}] --- End of Display thread ---\n")

def object_tracking():
    global count, tracker_init
    print(f"[{t.current_thread().name}] Tracking thread starting...")
    total_fps = FPS()

    while True:
        total_fps.start()
        start = time.perf_counter()
        print(f'\n[{t.current_thread().name}] tracker_init: {tracker_init}')
        
        # Run tracking if tracker has been initiated before
        if tracker_init == True:
            
            # Get curr frame
            curr_frame = frame_queue.get()

            if curr_frame is None:
                display_queue.put(None)
                break 

            count_lock.acquire()
            count+=1
            count_copy = count
            count_lock.release()

            print(f"[{t.current_thread().name}] Got frame from frame_queue - {count_copy}")

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
                r = detected_results[1]
                detected_num = detected_results[2]
                bbox_tracked = detected_results[3]

                # Check if got valid results
                if len(r['rois']) and bbox_tracked is not False:

                    bbox = extra.format_bbox(r['rois'])

                    # To format for object tracking
                    bbox=(bbox.x, bbox.y, bbox.w, bbox.h)
                    print(f"[{t.current_thread().name}] Detected Frame {detected_num} results: {bbox}")
                    print(f"[{t.current_thread().name}] Tracked Frame {detected_num} results: {bbox_tracked}")

                    # Check IOU >=0.7
                    iou = verify_tracker(bbox, bbox_tracked)
                    print(f"[{t.current_thread().name}] IOU: {iou}")

                    if iou >= 0.5:
                        print(f"[{t.current_thread().name}] Tracker is correct")

                        # Continue updating tracker
                        _ , bbox = tracker.update(curr_frame)

                    else:
                        print(f"[{t.current_thread().name}] Tracker is wrong!")

                        read_write_lock.acquire()
                        tracker_init = False
                        read_write_lock.release()
                        total_fps.stop()

                        # Display FPS on curr frame
                        cv2.putText(curr_frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
                        cv2.putText(curr_frame, f"Tracker Is Wrong", (7,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                        # Display result
                        display_queue.put([curr_frame, count_copy])
                        stop = time.perf_counter()
                        print(f"[{t.current_thread().name}] Time taken to track: {(stop-start)*1000:.2f}ms\n")
                        continue

                else:
                    print(f"[{t.current_thread().name}] Failed to detect instance from {detected_num}. Updating track on frame {count_copy} instead...")
                    
                    # Update Tracker
                    _ , bbox = tracker.update(curr_frame)
            
            total_fps.stop()

            # Display FPS on curr frame
            cv2.putText(curr_frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

            # Draw tracked bounding box
            draw_tracked_bbox(curr_frame, bbox)
                
            # Display result
            display_queue.put([curr_frame, count_copy])

        else:
            # The get() will block until get first detection
            detected_results = detect_queue.get() 

            # Ends if no more detection
            if detected_results is None:
                display_queue.put(None)
                break
            
            read_write_lock.acquire()

            detected_frame = detected_results[0]
            r = detected_results[1]
            detected_num = detected_results[2]  

            # Check if got valid results
            if len(r['rois']):

                print(f"[{t.current_thread().name}] Got valid detection from detect_queue")

                bbox = extra.format_bbox(r['rois'])

                # To format for object tracking
                bbox=(bbox.x, bbox.y, bbox.w, bbox.h)
                print(f"[{t.current_thread().name}] Frame {detected_num} results: {bbox}")

                # Init tracker
                tracker.init(detected_frame, bbox)
                tracker_init = True
                
                read_write_lock.release()
                
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

    ############## Initialize to read and write video ##############
    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo fps: {fps}")
    print(f"Video frame count: {total_frame_count}\n")

    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    #################################################################
    global count, tracker_init

    frame_grabber_thread = t.Thread(target=frame_grabber, args=(vid,))
    tracking_thread = t.Thread(target=object_tracking)
    display_thread = t.Thread(target=display_frames, args=(out,))

    frame_grabber_thread.start()
    tracking_thread.start()
    # display_thread.start()

    while True:
        start = time.perf_counter()

        curr_frame = frame_queue.get()

        count_lock.acquire()
        count+=1
        count_copy = count
        count_lock.release()

        print(f"[{t.current_thread().name}] Taking from frame_queue - Frame {count_copy}")

        if curr_frame is None:
            detect_queue.put(None)
            display_queue.put(None)
            break

        # Have to always keep tracker updated so that we can compare detection and tracked results
        # Placed this before undergoing detection as detection is slow where tracker thread may have updated
        # the tracker for a few frames already

        read_write_lock.acquire()

        if tracker_init == True:
            _ , bbox_tracked = tracker.update(curr_frame)

        else:
            bbox_tracked = False
        
        read_write_lock.release()
        
        # Mask RCNN detection
        results = model.detect([curr_frame], verbose=0)
        r = results[0]

        detect_queue.put([curr_frame, r, count_copy, bbox_tracked])

        print(f"[{t.current_thread().name}] Put frame {count_copy} in detect_queue")
        
        stop = time.perf_counter()
        print(f"[{t.current_thread().name}] Time taken to detect: {(stop-start)*1000:.2f}ms\n")   

    frame_grabber_thread.join()
    tracking_thread.join()

    display_thread.start()
    display_thread.join()

    print(f'[{t.current_thread().name}] frame_queue: {frame_queue.qsize()}\n')
    print(f'[{t.current_thread().name}] detect_queue: {detect_queue.qsize()}')
    print(f'[{t.current_thread().name}] detect_queue: {list(detect_queue.queue)}\n')
    print(f'[{t.current_thread().name}] display_queue: {display_queue.qsize()}')
    print(f'[{t.current_thread().name}] display_queue: {list(display_queue.queue)}\n')
    print(f"[{t.current_thread().name}] ---End---")
    
    vid.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)