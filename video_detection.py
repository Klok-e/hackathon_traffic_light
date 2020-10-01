import cv2
import numpy as np
import time
from yolo_model import detect
import torch
from yolo_model import sort
from yolo_model.detect import BBox

outputFrame = None


def capture_images_continually(capture: cv2.VideoCapture, model, classes, img_size, device):
    global outputFrame

    track = sort.Sort()
    while True:
        t0 = time.time()

        ret, frame = capture.read()

        with torch.no_grad():
            boxes = detect.detect(model, frame, img_size, device=device)

        boxes = boxes[np.isin(boxes[:, 5], [2, 3, 5, 7])]  # in [2, 3, 5, 7]

        if not ret:
            print('no video')
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        def box_to_arr(box: BBox):
            return np.array([box.x0, box.y0, box.x1, box.y1, box.confidence])

        boxes_no_class = boxes[:, :-1]
        tracked_objects = track.update(boxes_no_class)

        font = cv2.FONT_HERSHEY_PLAIN
        for x0, y0, x1, y1, obj_id in tracked_objects:
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 215, 0), 2)
            cv2.putText(frame, str(obj_id), (x0, y0 + 30), font, 2, (0, 255, 0), 2)

        outputFrame = frame

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
