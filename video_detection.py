import cv2
import numpy as np
import time
from yolo_model import detect
import torch

outputFrame = None


def capture_images_continually(capture: cv2.VideoCapture, model, classes, img_size):
    global outputFrame

    while True:
        t0 = time.time()

        ret, frame = capture.read()

        with torch.no_grad():
            boxes = detect.detect(model, frame, img_size)

        font = cv2.FONT_HERSHEY_PLAIN
        for box in boxes:
            label = str(classes[box.class_index])
            cv2.rectangle(frame, (box.x0, box.y0), (box.x1, box.y1), (0, 215, 0), 2)
            cv2.putText(frame, label, (box.x0, box.y0 + 30), font, 3, (0, 255, 0), 3)

        # # Detecting objects
        # blob = cv2.dnn.blobFromImage(frame, 1. / 255., (416, 416), (0, 0, 0), True, crop=False)
        # network.setInput(blob)
        # outs = network.forward(unconnected)
        #
        # # Showing informations on the screen
        # class_ids = []
        # confidences = []
        # boxes = []
        # for out in outs:
        #     for detection in out:
        #         scores = detection[5:]
        #         class_id = np.argmax(scores)
        #         confidence = scores[class_id]
        #         if (confidence > 0.5) and (class_id == 2 or class_id == 3 or class_id == 5 or class_id == 7):
        #             # Object detected
        #             center_x = int(detection[0] * width)
        #             center_y = int(detection[1] * height)
        #             w = int(detection[2] * width)
        #             h = int(detection[3] * height)
        #             # Rectangle coordinates
        #             x = int(center_x - w / 2)
        #             y = int(center_y - h / 2)
        #             boxes.append([x, y, w, h])
        #             confidences.append(float(confidence))
        #             class_ids.append(class_id)
        #
        # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        #
        # font = cv2.FONT_HERSHEY_PLAIN
        # for i in range(len(boxes)):
        #     if i in indexes:
        #         x, y, w, h = boxes[i]
        #         label = str(classes[class_ids[i]])
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 215, 0), 2)
        #         cv2.putText(frame, label, (x, y + 30), font, 3, (0, 255, 0), 3)

        if not ret:
            print('no video')
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        outputFrame = frame

        print(f"frame time: {time.time() - t0:.2}")


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
