import json
# import xgboost
from xgboost import XGBClassifier
import joblib
import pandas as pd
from flask import Flask, request
# from models.config import *

flask_app = Flask(__name__)

params = {'max_depth': 3,
          'gamma': 0,
          'min_child_weight': 0.3,
          'max_delta_step': 1,
          'subsample': 1,
          'colsample_bytree': 0.6,
          'colsample_bylevel': 0.6,
          'random_state': 0,
          'seed': 10,
          'n_estimators': 10,
          'missing': None,
          'eval_metric': 'logloss'}

clf = XGBClassifier(**params)
clf = clf.load_model('models/base_model.json')
main_model = 'base'


@flask_app.route('/', methods=['GET'])
def index_page():
    return_data = {
        "error": "0",
        "message": "Successful"
    }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')


@flask_app.route('/train', methods=['POST'])
def train_model():
    try:
        payload = request.get_json(force=True)
        y_test = payload['label']
        payload.drop('label', axis=1, inplace=True)

        clf.fit(payload, y_test,
                xgb_model='app/models/base_model.json',
                verbose=True,
                early_stopping_rounds=5
                )

        global main_model
        main_model = 'trained'

        return_data = {
                "error": '0',
                "message": 'New trees were added'
            }

    except Exception as e:
        return_data = {
            'error': '2',
            "message": str(e)
        }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')


@flask_app.route('/predict', methods=['GET'])
def model_deploy():
    try:
        payload = request.get_json(force=True)
        data = pd.DataFrame(payload).head(1)
        prediction = clf.predict_proba(data)[0][1]

        return_data = {
                "error": '0',
                "message": 'Prediction',
                "model type": main_model,
                "prediction": float(prediction)
            }

    except Exception as e:
        return_data = {
            'error': '2',
            "message": str(e)
        }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')


if __name__ == "__main__":
    flask_app.run(host='0.0.0.0', port=8080, debug=False)


# build image $docker build -t ecoplant_ml_model:latest .
# run image $docker run -p 8080:8080 ecoplant_ml_model:latest
# run image with constraints $docker run -p 8080:8080 --memory=750m --cpus=0.5 ecoplant_ml_model:latest

