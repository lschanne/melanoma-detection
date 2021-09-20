from flask import render_template, request, Response
from flask.views import View

from .forms import PatientDataForm
from .utils import get_prediction_results, process_form

class Index(View):
    methods = ['GET', 'POST']

    def dispatch_request(self, *args, **kwargs) -> Response:
        form = PatientDataForm(request.form)
        if request.method == 'GET':
            return self.get(form)
        return self.post(form)
    
    def get(self, form: PatientDataForm) -> Response:
        return render_template('index.html', form=form)
    
    def post(self, form: PatientDataForm) -> Response:
        if not form.validate():
            return render_template('index.html', form=form)

        model_inputs = process_form(request, form)
        context = get_prediction_results(**model_inputs)

        return render_template('results.html', **context)
