# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from . import base_model
from . import lr
from . import my_xgb
from . import rf
from . import linear
from . import sklearn_xgb
from . import sklearn_xgb_reg
from . import desion_tree
from . import my_lightgbm
from . import lgb_category
import configparser
from ..ml_utils import runstatus


class ModelFactory(object):
    """Model factory.Create a model based on the parameters passed in.

    Parameters
    --------
    config: configparser.ConfigParser()
         Configuration file of model.
    model_name: str
         Name of model.

    Examples
    --------
    >>> from xes_ml_arch.src.model import model_factory
    >>> import configparser
    >>> import pandas as pd
    >>> import numpy as np
    >>> import random
    >>> config = configparser.ConfigParser()
    >>> config.read("model_online.conf")
    >>> x = pd.read_csv('data.txt', sep=',')
    >>> y = np.array([int(random.random() * 100) for _x in range(99)])
    >>> ins = model_factory.ModelFactory()
    >>> my_model = ins.create_model(config=config, model_name='lr')
    >>> my_model.train(x, y)
    ...
    ...
    """
    @staticmethod
    def create_model(config, model_name=None,log_path = None):
        """Create a model.

        Parameters
        --------
        config: configparser.ConfigParser
            Config object.
        model_name: str
            Name of model.

        Returns
        --------
        :return: model object or None

        Notes
        --------
        This function may raise exception include TypeError, RuntimeError
        """
        model = None

        # get model name
        if not isinstance(config, configparser.ConfigParser):
            raise TypeError("config object is not instance of ConfigParser")

        if model_name is None:
            try:
                model_name = config.get(base_model.BaseModel.BASE_CONF,
                                        base_model.BaseModel.MODEL_NAME)
            except configparser.Error:
                raise RuntimeError("config has no section named %s, or has no option named %s" % (
                    base_model.BaseModel.BASE_CONF, base_model.BaseModel.MODEL_NAME))

        # create a model
        if model_name == base_model.BaseModel.MODEL_XGB:
            model = my_xgb.MyXgb(config=config,log_path = log_path)
        elif model_name == base_model.BaseModel.MODEL_LR:
            model = lr.LR(config=config,log_path = log_path)
        elif model_name == base_model.BaseModel.MODEL_RF:
            model = rf.RF(config=config,log_path = log_path)
        elif model_name == base_model.BaseModel.MODEL_LINE_REG:
            model = linear.Liner(config=config,log_path = log_path)
        elif model_name == base_model.BaseModel.MODEL_SKLEARN_XGB:
            model = sklearn_xgb.SklearnXGB(config=config,log_path = log_path)
        elif model_name == base_model.BaseModel.MODEL_SKLEARN_XGB_REG:
            model = sklearn_xgb_reg.SklearnXGBReg(config=config,log_path = log_path)
        elif model_name == base_model.BaseModel.MODEL_DST:
            model = desion_tree.MyDesionTree(config=config,log_path = log_path)
        elif model_name == base_model.BaseModel.MODEL_LIGHTGBM:
            model = my_lightgbm.MyLightGBM(config=config,log_path = log_path)
        elif model_name == base_model.BaseModel.MODEL_CATE_LIGHTGBM:
            model = lgb_category.Lgbcf(config=config, log_path = log_path)
        else:
            pass

        if model is None:
            raise RuntimeError("can not create a model named: %s" % (model_name))

        # Initialize model
        if runstatus.RunStatus.FAILED == model.init():
            raise RuntimeError("model init faild")

        return model
