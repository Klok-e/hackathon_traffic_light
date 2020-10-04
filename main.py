from flask import Flask
from flask import request, jsonify
import flask
import video_detection

ONLY_SERVER = False


app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


@app.route('/')
def root_redirect():
    return flask.redirect("/index.html")


@app.route('/index.html')
def default_route():
    return flask.render_template("index.html",
                                 line_coords=video_detection.LINE_COORD,
                                 traffic_coords=video_detection.TRAFFIC_LIGHT_RECT,
                                 traffic_color_detect=video_detection.DETECT_TRAFFIC_LIGHT_COLOR,
                                 traffic_color=video_detection.LINE_COORD_COLOR)


@app.route("/video_feed")
def video_feed():
    return flask.Response(video_detection.generate_image_binary(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/coordinates", methods=['POST'])
def coordinates():
    x1 = request.form['line_x1']
    y1 = request.form['line_y1']
    x2 = request.form['line_x2']
    y2 = request.form['line_y2']
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    video_detection.set_line(x1, y1, x2, y2)

    return flask.redirect("/index.html")


@app.route("/traffic_light_coordinates", methods=['POST'])
def traffic_light_coordinates():
    x1 = request.form['traffic_x1']
    y1 = request.form['traffic_y1']
    x2 = request.form['traffic_x2']
    y2 = request.form['traffic_y2']
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    video_detection.set_traffic(x1, y1, x2, y2)

    return flask.redirect("/index.html")


@app.route("/detect_traffic_lights", methods=['POST'])
def detect_traffic_lights():
    video_detection.set_detect_traffic_light()
    return flask.redirect("/index.html")


@app.route("/toggle_traffic_light_detect", methods=['POST'])
def toggle_traffic_light_detect():
    video_detection.set_detect_traffic_color(not video_detection.DETECT_TRAFFIC_LIGHT_COLOR)
    return flask.redirect("/index.html")


@app.route("/toggle_traffic_light_color", methods=['POST'])
def toggle_traffic_light_color():
    video_detection.set_traffic_color(video_detection.GREEN
                                      if video_detection.LINE_COORD_COLOR == video_detection.RED_OR_YELLOW
                                      else video_detection.RED_OR_YELLOW)
    return flask.redirect("/index.html")


        


def main():
    import threading
    import cv2
    import numpy as np
    from yolo_model import detect
    

    np.random.seed(42)
    capture = cv2.VideoCapture("D:/Me/hackaton/hackathon_traffic_light/aziz1.mp4")
    VID_SIZE = (int(capture.get(3)), int(capture.get(4)))     
    if not ONLY_SERVER:
        img_size = 512
        model, device = detect.create_model(
            "D:/Me/hackaton/hackathon_traffic_light/yolo_model/cfg/yolov3-spp.cfg",
            "D:/Me/hackaton/hackathon_traffic_light/yolo/yolov3-spp-ultralytics.pt", img_size,
            device="")
        classes = detect.load_classes("D:/Me/hackaton/hackathon_traffic_light/yolo_model/data/coco.names")

        t = threading.Thread(target=video_detection.capture_images_continually,
                             kwargs={"capture": capture, "model": model, "classes": classes,
                                     "img_size": img_size, "device": device},
                             daemon=True)
        t.start()
    else:
        ret, frame = capture.read()
        video_detection.outputFrame = frame

    app.run()
    capture.release()


if __name__ == "__main__":
    main()
