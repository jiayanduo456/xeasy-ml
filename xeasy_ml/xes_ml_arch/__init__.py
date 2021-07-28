# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT



#from xeasy-ml.xes_ml_arch.project_init import create_new_demo
from .src.analysis import analysis
from .src.constance import predict_constance
from .src.cross_validation import cross_validation, data_split
from .src.feature_enginnering import data_processor, data_sampler, data_washer, feature_discretizer, feature_filter, pre_feature_utils, xes_onehot_encoder
from .src.ml import base_ml, prediction_ml, train_model_ml
from .src.ml_utils import configmanager, feature_processor, global_pre, jsonmanager, pre_utils, runstatus
from .src.model import base_model, desion_tree, linear, lr, model_factory, my_lightgbm, my_xgb, rf, sklearn_xgb, sklearn_xgb_reg, svr
from .src.optimizing import optimizing
from .src.systemlog import accesslog, syserrorlog, syslog, sysmanagerlog
from .src.schema import accessdictmodel, data_set
from .__main__ import train, predict