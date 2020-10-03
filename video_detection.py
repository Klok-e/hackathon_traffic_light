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
        
        lights = traffic_color(frame, traffic_lights)

        for color, light in lights:
            if color != 'no color':
                cv2.rectangle(frame, (light[0], light[1]), (light[2], light[3]), (255, 215, 0), 2)
                cv2.putText(frame, color, (int(light[0]), int(light[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)   
        
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




def traffic_color(frame, traffic_lights):
    # select the largest (temp solution)
    result = []
    for element in traffic_lights:
        color = None
        
        x0, y0, x1, y1, _, __ = element
        # create red mask and appy it to the light
        ligth1 = frame[int(x0) : int(x1), int(y0) : int(y0 + (y1 - y0) / 2)]
        lower_red = np.array([0, 85, 110], dtype = "uint8")
        upper_red = np.array([15, 255, 255], dtype = "uint8")
        
    
        lower_violet = np.array([165, 85, 110], dtype = "uint8")
        upper_violet = np.array([180, 255, 255], dtype = "uint8")
        
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
        ligth2 = frame[int(x0) : int(x1), int(y0 + (y1 - y0) / 2) : int(y1)]
        lower_green = np.array([40, 85, 110], dtype = "uint8")
        upper_green = np.array([91, 255, 255], dtype = "uint8")    
        green_mask = cv2.inRange(ligth2, lower_green, upper_green)
        ligth_green = cv2.bitwise_and(ligth2, ligth2, mask=green_mask)
        green_ocur = 0
        for x in ligth_green:
            for y in x:
                for el in y:
                    if el != 0:
                        green_ocur += 1                

        if red_occur > (ligth1.size / 2.7):
            color = 'red or yellow'
        elif green_ocur > (ligth2.size / 2.7):
            color = 'green'
        else:
            color = 'no color'
        result.append((color, element))
         
    return result
    



        

       
        