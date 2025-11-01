# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
from io import StringIO
from omegaconf import OmegaConf
import torch
import numpy as np
from samplers import ablation_sampler
import flask
import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None
    input_shape = None
    config = None   
                 # Where we keep the model when it's loaded
    def sample_random_batch(EHR_task, sampling_shape, sampler, device, n_classes=None):
    # make_dir(path)
        x = torch.randn(sampling_shape, device=device)
        if n_classes is not None:
            y = torch.randint(n_classes, size=(
                sampling_shape[0],), dtype=torch.int32, device=device)
        else:
            y = None

        x = sampler(x, y).cpu()
        x = x.numpy()

        if EHR_task == 'binary':
            x = np.rint(np.clip(x, 0, 1))
        elif EHR_task == 'continuous':
            x = np.clip(x, 0, 1)

        return x
    
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            state = torch.load(os.path.join(model_path,"final_checkpoint.pth"), map_location="cpu")
            config = OmegaConf.load("./configs/mimic/health_forge_train_edm.yaml")
           
            cls.input_shape = state['input_shape']
            cls.model = state['model']
            cls.model.load_state_dict(state['ema'].state_dict(), strict=False)
            cls.config = config

            
            
        return cls.model

    @classmethod
    def predict(cls):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()

        def sampler(x, y=None):
            return ablation_sampler(x, y, clf, **cls.config.sampler)


        EHR_task = 'binary'

        return cls.sample_random_batch(EHR_task, cls.input_shape, sampler, 
                                                     cls.config.setup.device, cls.config.data.n_classes)
    
    

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    
    # Data is not needed for now, but will keep this csv parsing portion for future
    # data = None

    # # Convert from CSV to pandas
    # if flask.request.content_type == 'text/csv':
    #     print(flask.request.data)
    #     data = flask.request.data.decode('utf-8')
    #     print(data)
    #     s = StringIO(data)
    #     print(s)
    #     data = pd.read_csv(s, header=None)
    # else:
    #     return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    # print('Invoked with {} records'.format(data.shape[0]))

    print('generating a new piece of EHR Data')
    # Do the prediction
    predictions = ScoringService.predict()
    print(predictions)
    # Convert from numpy back to CSV
    out = StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
