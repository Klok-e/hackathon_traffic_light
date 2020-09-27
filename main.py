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

    capture = cv2.VideoCapture("out.mp4")

    t = threading.Thread(target=video_detection.capture_images_continually, args=[capture, ], daemon=True)
    t.start()

    app.run()


if __name__ == "__main__":
    main()
