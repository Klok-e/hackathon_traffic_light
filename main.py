import cv2 as cv
from flask import Flask
import flask

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


@app.route('/')
def hello_world():
    return flask.redirect("/index.html")


def main():
    app.run()


if __name__ == "__main__":
    main()
