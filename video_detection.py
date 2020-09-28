import cv2
import numpy as np
import time
from yolo_model import detect
import torch
from yolo_model import sort

outputFrame = None


def capture_images_continually(capture: cv2.VideoCapture, model, classes, img_size, device):
    global outputFrame

    track = sort.Sort()
    while True:
        t0 = time.time()

        ret, frame = capture.read()

        with torch.no_grad():
            boxes = detect.detect(model, frame, img_size, device=device)

        boxes = list(filter(lambda x: x.class_index in [2, 3, 5, 7], boxes))

        font = cv2.FONT_HERSHEY_PLAIN
        for box in boxes:
            label = str(classes[box.class_index])
            x0, y0, x1, y1 = map(int, [box.x0, box.y0, box.x1, box.y1])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 215, 0), 2)
            cv2.putText(frame, label[0], (x0, y0 + 30), font, 2, (0, 255, 0), 2)
            cv2.putText(frame, f"{box.confidence:.2}", (x0 + 50, y0 + 30), font, 2, (0, 255, 0), 2)

        if not ret:
            print('no video')
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        track.update()

        outputFrame = frame

        # print(f"frame time: {time.time() - t0:.2}")


def generate_image_binary():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
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
