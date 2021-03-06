from flask import render_template, request, Response
from flask.views import View

from .constants import AUTHORS
from .forms import PatientData
from .models import Model

# only initialize it once because it takes a while to load the model
model = Model()

class RenderTemplateView(View):
    '''
    Base View class for accepting a GET request and rendering a page.
    '''
    methods = ['GET']
    context = {}

    @property
    def template(self):
        raise NotImplementedError(
            f'Class {self.__class__} does not have a template set.'
        )

    def dispatch_request(self) -> Response:
        return render_template(self.template, **self.context)

class Overview(RenderTemplateView):
    template = 'overview.html'

class About(RenderTemplateView):
    template = 'about.html'
    context = {'authors': AUTHORS}

class Performance(RenderTemplateView):
    template = 'performance.html'

class Prediction(View):
    '''
    On GET, have user fill out PatientData form to make a prediction.
    On POST, return model prediction results and relevant metrics.
    '''
    methods = ['GET', 'POST']

    def dispatch_request(self, *args, **kwargs) -> Response:
        form = PatientData(request.form)
        if request.method == 'GET':
            return self.get(form)
        return self.post(form)
    
    def get(self, form: PatientData) -> Response:
        return render_template('prediction.html', form=form)
    
    def post(self, form: PatientData) -> Response:
        if not form.validate():
            return render_template('prediction.html', form=form)
        context = model.predict(request, form)
        return render_template('results.html', **context)
