# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


import configparser
import pandas as pd
import pickle

import sys
from xgboost import XGBClassifier

import xeasy_ml as xml
import warnings

warnings.filterwarnings('ignore')

sys.path.append("../../")

BASE_CONFIG = "base_config"
RESULT_PATH = "result_path"
data_file_name = "./data/test.txt"

def predict(conf = './config/demo/feature_enginnering.conf', model_path = "./result/v1/model/demo.m", model_name='sklearn_xgb'):
    '''

    Parameters
    ----------
    conf: str
        Config file path for feature_enginnering.conf.
    model_path: str
        The storage path of the trained model.
    model_name：str
        Name of your model.

    Returns
    -------
    predict_res: list
        Predict results.
    '''
    print("start predict...")
    # Load feature processing configuration file.
    ml_config = configparser.ConfigParser()
    ml_config.read(conf)

    # Initialize the prediction class.
    xeasy_log_path = None
    ml = xml.prediction_ml.PredictionML(xeasy_log_path=xeasy_log_path)

    # Load model
    if model_name == 'sklearn_xgb':
        ml._model = XGBClassifier()
        ml._model.load_model(model_path)
    else:
        ml._model = pickle.load(open(model_path, xml.global_pre.Global.FILE_READ))

    # Load data
    ml.set_data(pd.read_csv(data_file_name))

    # Data preprocessing
    ml._feature_processor = xml.data_processor.DataProcessor(conf=ml_config)
    ml._feature_processor.init()
    ml._feature_processor.test_data = ml._data
    ml._feature_processor.execute()

    # Test data
    test_feature = ml._feature_processor.test_data_feature
    test_feature = test_feature.astype("float64", errors='ignore')

    # Predict result
    predict_res = ml._model.predict(test_feature)

    return predict_res

def train():
    print("start ml...")
    ml_config = configparser.ConfigParser()
    ml_config.read("./config/demo/ml.conf")
    xml.global_pre.RES_PATH = ml_config.get(BASE_CONFIG, RESULT_PATH)

    xeasy_log_path = None

    ml = xml.train_model_ml.TrainML(conf=ml_config, xeasy_log_path=xeasy_log_path)

    if ml.init() == xml.runstatus.RunStatus.FAILED:
        print("ml init failed")
    if ml.set_data(pd.read_csv("./data/sample.txt")) == xml.runstatus.RunStatus.FAILED:
        print("ml set data failed")
    if ml.start() == xml.runstatus.RunStatus.FAILED:
        print("ml train failed")
    print("finish ml!")


if __name__ == '__main__':
    train()
    result = predict()
    print(result)
