import cv2
import threading

outputFrame = None
lock = threading.Lock()


def capture_images_continually(capture: cv2.VideoCapture):
    global outputFrame, lock
    while True:
        ret, frame = capture.read()
        if not ret:
            print('no video')
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        with lock:
            outputFrame = frame.copy()


def generate_image_binary():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
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
