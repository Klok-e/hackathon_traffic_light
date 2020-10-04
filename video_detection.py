import cv2
import numpy as np
import time
from yolo_model import detect
import torch
from yolo_model import sort
from collections import Counter

NO_COLOR = 'no color'
RED_OR_YELLOW = 'red or yellow'
GREEN = 'green'

outputFrame = None
LINE_COORD = ((500, 1500), (3350, 1450))  # only for aziz1
LINE_COORD_COLOR = NO_COLOR

DETECT_TRAFFIC_LIGHT = True
TRAFFIC_LIGHT_RECT = ((0, 0), (1, 1))
DETECT_TRAFFIC_LIGHT_COLOR = True

# TODO: make this into config options at startup (like --print_frame_duration --print_encode_duration etc.)
PRINT_FRAME_DURATION = False
PRINT_ENCODE_DURATION = False
DRAW_DETECTION_BOXES = True
DRAW_TRACKING_BOXES = False



def set_detect_traffic_color(val):
    global DETECT_TRAFFIC_LIGHT_COLOR
    DETECT_TRAFFIC_LIGHT_COLOR = val


def set_traffic_color(val):
    global LINE_COORD_COLOR
    LINE_COORD_COLOR = val


def set_detect_traffic_light():
    global DETECT_TRAFFIC_LIGHT
    DETECT_TRAFFIC_LIGHT = True


def set_traffic(x1, y1, x2, y2):
    global TRAFFIC_LIGHT_RECT
    TRAFFIC_LIGHT_RECT = ((x1, y1), (x2, y2))


def set_line(x1, y1, x2, y2):
    global LINE_COORD
    LINE_COORD = ((x1, y1), (x2, y2))


def capture_images_continually(capture: cv2.VideoCapture, model, classes, img_size, device):
    global outputFrame, LINE_COORD_COLOR, DETECT_TRAFFIC_LIGHT, TRAFFIC_LIGHT_RECT

    violations = {}
    tracked_paths = {}
    track = sort.Sort()
    i = 0
    while True:
        i += 1
        ret, frame = capture.read()

        t0 = 0
        if PRINT_FRAME_DURATION:
            t0 = time.time()

        if not ret:
            print('no video')
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        with torch.no_grad():
            boxes = detect.detect(model, frame, img_size, device=device)

        boxes_cars = boxes[np.isin(boxes[:, 5], [2, 3, 5, 7])]  # take car, motobuke, bus, truck

        if DETECT_TRAFFIC_LIGHT:
            DETECT_TRAFFIC_LIGHT = False
            traffic_lights = boxes[np.isin(boxes[:, 5], [9])]
            traffic_lights_coords = traffic_lights[:, :4]
            lights = traffic_color(frame, traffic_lights_coords)

            for color, light in lights:
                if color != NO_COLOR:
                    TRAFFIC_LIGHT_RECT = (light[0], light[1]), (light[2], light[3])
        if DETECT_TRAFFIC_LIGHT_COLOR:
            nptraffic_coords = np.reshape(np.array([*TRAFFIC_LIGHT_RECT[0], *TRAFFIC_LIGHT_RECT[1]]), (1, 4))
            lights = traffic_color(frame, nptraffic_coords)
            detected_tr_lights = Counter(
                list(filter(lambda x: x != NO_COLOR, map(lambda x: x[0], lights)))).most_common()
            if len(detected_tr_lights) > 0:
                LINE_COORD_COLOR = detected_tr_lights[0][0]

            # draw traffic light
            cv2.rectangle(frame, TRAFFIC_LIGHT_RECT[0], TRAFFIC_LIGHT_RECT[1], (255, 215, 0), 2)
            cv2.putText(frame, LINE_COORD_COLOR, (int(TRAFFIC_LIGHT_RECT[0][0]), int(TRAFFIC_LIGHT_RECT[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)

        boxes_no_class = boxes_cars[:, :-1]

        if DRAW_DETECTION_BOXES:
            for x0, y0, x1, y1, confidence in boxes_no_class:
                x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        tracked_objects = track.update(boxes_no_class)

        # process tracking boxes
        for x0, y0, x1, y1, obj_id in tracked_objects:
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            if obj_id not in tracked_paths:
                tracked_paths[obj_id] = []
            path = tracked_paths[obj_id]
            path.append((int((x0 + x1) / 2.), int((y0 + y1) / 2.), time.time()))

            if line_intersection(LINE_COORD, path) and LINE_COORD_COLOR == RED_OR_YELLOW:
                if obj_id not in violations:
                    print("violation!")
                    # TODO: store something useful here, otherwise it's just a hashset
                    violations[obj_id] = None

            if obj_id in violations:
                rect_rgb = (0, 0, 215)
                line_rgb = (0, 0, 215)
            else:
                rect_rgb = (0, 215, 0)
                line_rgb = (0, 255, 0)

            # draw tracking box
            if DRAW_TRACKING_BOXES:
                cv2.rectangle(frame, (x0, y0), (x1, y1), rect_rgb, 2)

            for i in range(len(path) - 1):
                cv2.line(frame, tuple(path[i][:2]), tuple(path[i + 1][:2]), line_rgb, 2)

        cv2.line(frame, LINE_COORD[0], LINE_COORD[1], (255, 0, 0), 2)

        if PRINT_FRAME_DURATION:
            print(f"frame time: {time.time() - t0:.2}")

        outputFrame = frame

        # clean old paths (older than 30 seconds)
        if i % 1000 == 0:
            tracked_paths = {k: v for k, v in tracked_paths.items() if len(v) != 0}
            for key, val in tracked_paths.items():
                val[:] = [[*a, time_created] for *a, time_created in val if time.time() - time_created < 30]


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
        t0 = 0
        if PRINT_ENCODE_DURATION:
            t0 = time.time()
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        if PRINT_ENCODE_DURATION:
            print(f"encode time: {time.time() - t0:.2}")
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def traffic_color(frame, traffic_lights):
    """
    :param frame:
    :param traffic_lights: array of coordinates x0,y0,x1,y1
    :return:
    """
    # select the largest (temp solution)
    result = []
    for element in traffic_lights:
        color = None

        x0, y0, x1, y1 = element
        # create red mask and appy it to the light
        ligth1 = frame[int(x0): int(x1), int(y0): int(y0 + (y1 - y0) / 2)]
        lower_red = np.array([0, 85, 110], dtype="uint8")
        upper_red = np.array([15, 255, 255], dtype="uint8")

        lower_violet = np.array([165, 85, 110], dtype="uint8")
        upper_violet = np.array([180, 255, 255], dtype="uint8")

        red_mask_orange = cv2.inRange(ligth1, lower_red, upper_red)
        red_mask_violet = cv2.inRange(ligth1, lower_violet, upper_violet)

        red_mask_full = red_mask_orange + red_mask_violet

        ligth_red = cv2.bitwise_and(ligth1, ligth1, mask=red_mask_full)
        # check if there is something green (may need to extand color boundaries)
        red_occur = 0
        for x in ligth_red:
            for y in x:
                for el in y:
                    if el != 0:
                        red_occur += 1
        ligth2 = frame[int(x0): int(x1), int(y0 + (y1 - y0) / 2): int(y1)]
        lower_green = np.array([40, 85, 110], dtype="uint8")
        upper_green = np.array([91, 255, 255], dtype="uint8")
        green_mask = cv2.inRange(ligth2, lower_green, upper_green)
        ligth_green = cv2.bitwise_and(ligth2, ligth2, mask=green_mask)
        green_ocur = 0
        for x in ligth_green:
            for y in x:
                for el in y:
                    if el != 0:
                        green_ocur += 1

        if red_occur > (ligth1.size / 2.7):
            color = RED_OR_YELLOW
        elif green_ocur > (ligth2.size / 2.7):
            color = GREEN
        else:
            color = NO_COLOR
        result.append((color, element))

    return result


def line_intersection(line1, line2):
    # stolen from here https://stackoverflow.com/a/62625458
    # assumes line segments are stored in the format [(x0,y0),(x1,y1)]
    def intersects(s0, s1):
        dx0 = s0[1][0] - s0[0][0]
        dx1 = s1[1][0] - s1[0][0]
        dy0 = s0[1][1] - s0[0][1]
        dy1 = s1[1][1] - s1[0][1]
        p0 = dy1 * (s1[1][0] - s0[0][0]) - dx1 * (s1[1][1] - s0[0][1])
        p1 = dy1 * (s1[1][0] - s0[1][0]) - dx1 * (s1[1][1] - s0[1][1])
        p2 = dy0 * (s0[1][0] - s1[0][0]) - dx0 * (s0[1][1] - s1[0][1])
        p3 = dy0 * (s0[1][0] - s1[1][0]) - dx0 * (s0[1][1] - s1[1][1])
        return (p0 * p1 <= 0) and (p2 * p3 <= 0)

    for i in range(len(line1) - 1):
        for k in range(len(line2) - 1):
            l1pos1 = line1[i]
            l1pos2 = line1[i + 1]
            l2pos1 = line2[k]
            l2pos2 = line2[k + 1]

            if intersects((l1pos1, l1pos2), (l2pos1, l2pos2)):
                return True
    return False
