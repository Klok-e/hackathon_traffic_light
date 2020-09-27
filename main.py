from flask import Flask
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


def main():
    import threading
    import cv2
    import numpy as np
    from yolo_model import detect

    np.random.seed(42)
    capture = cv2.VideoCapture("out.mp4")
    img_size = 512
    model, device = detect.create_model("yolo_model/cfg/yolov3-spp.cfg", "yolo/yolov3-spp-ultralytics.pt", img_size,
                                        device="")
    classes = detect.load_classes("yolo_model/data/coco.names")

    t = threading.Thread(target=video_detection.capture_images_continually,
                         kwargs={"capture": capture, "model": model, "classes": classes,
                                 "img_size": img_size, "device": device},
                         daemon=True)
    t.start()

    app.run()
    capture.release()


if __name__ == "__main__":
    main()
