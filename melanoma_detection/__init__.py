import os

from flask import Flask

from melanoma_detection import views

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app.add_url_rule('/', view_func=views.Overview.as_view('/'))
    app.add_url_rule('/overview', view_func=views.Overview.as_view('/overview'))
    app.add_url_rule('/about', view_func=views.About.as_view('/about'))
    app.add_url_rule('/performance', view_func=views.Performance.as_view('/performance'))
    app.add_url_rule('/prediction', view_func=views.Prediction.as_view('/prediction'))

    return app

