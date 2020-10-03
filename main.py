from flask import Flask
from flask import request
import flask
import video_detection

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


@app.route('/')
def root_redirect():
    return flask.redirect("/index.html")


@app.route('/index.html')
def default_route():
    return flask.render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return flask.Response(video_detection.generate_image_binary(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/coordinates", methods=['POST'])
def coordinates():
    x = request.form['x']
    print("x=" + x)
    y = request.form['y']
    print("y=" + y)
    return flask.render_template("index.html")


@app.route("/rtl", methods=['POST'])
def rtl():
    pass
    return flask.render_template("index.html")


def main():
    import threading
    import cv2
    import numpy as np
    from yolo_model import detect

    np.random.seed(42)
    capture = cv2.VideoCapture("C:/Python/Test_test/project/hackathon_traffic_light/out32.mp4")
    img_size = 512
    model, device = detect.create_model(
        "C:/Python/Test_test/project/hackathon_traffic_light/yolo_model/cfg/yolov3-spp.cfg",
        "C:/Python/Test_test/project/hackathon_traffic_light/yolo/yolov3-spp-ultralytics.pt", img_size,
        device="")
    classes = detect.load_classes("C:/Python/Test_test/project/hackathon_traffic_light/yolo_model/data/coco.names")

    t = threading.Thread(target=video_detection.capture_images_continually,
                         kwargs={"capture": capture, "model": model, "classes": classes,
                                 "img_size": img_size, "device": device},
                         daemon=True)
    t.start()

    app.run()
    capture.release()


if __name__ == "__main__":
    main()
