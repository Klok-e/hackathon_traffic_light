import cv2
import numpy as np
import time
from yolo_model import detect
import torch
from yolo_model import sort

outputFrame = None


def capture_images_continually(capture: cv2.VideoCapture, model, classes, img_size, device):
    global outputFrame

    tracked_paths = {}
    track = sort.Sort()
    i = 0
    while True:
        i += 1
        ret, frame = capture.read()

        with torch.no_grad():
            boxes = detect.detect(model, frame, img_size, device=device)

        traffic_lights = boxes[np.isin(boxes[:, 5], [9])] 
        boxes = boxes[np.isin(boxes[:, 5], [2, 3, 5, 7])] # take car, motobuke, bus, truck 

        if not ret:
            print('no video')
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        boxes_no_class = boxes[:, :-1]
        
        tracked_objects = track.update(boxes_no_class)

        font = cv2.FONT_HERSHEY_PLAIN
        for x0, y0, x1, y1, obj_id in tracked_objects:
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            if obj_id not in tracked_paths:
                tracked_paths[obj_id] = []
            path = tracked_paths[obj_id]
            path.append((int((x0 + x1) / 2.), int((y0 + y1) / 2.), time.time()))

            for i in range(len(path) - 1):
                cv2.line(frame, tuple(path[i][:2]), tuple(path[i + 1][:2]), (0, 255, 0), 2)

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 215, 0), 2)
        
        
        for x0, y0, x1, y1, _, __ in traffic_lights:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 215, 0), 2)

        outputFrame = frame

        # clean old paths (older than 30 seconds)
        if i % 1000 == 0:
            tracked_paths = {k: v for k, v in tracked_paths.items() if len(v) != 0}
            for key, val in tracked_paths.items():
                val[:] = [[*a, time_created] for *a, time_created in val if time.time() - time_created < 30]

        # print(f"frame time: {time.time() - t0:.2}")


def generate_image_binary():
    # grab global references to the output frame and lock variables
    global outputFrame
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if outputFrame is None:
            continue
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
