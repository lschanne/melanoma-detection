from flask import render_template, Response
from flask.views import View

class Index(View):
    methods = ['GET']

    def dispatch_request(self) -> Response:
        return render_template('index.html')
