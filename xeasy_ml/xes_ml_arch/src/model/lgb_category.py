# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

import time
import traceback
import numpy as np
import gc
from ..ml_utils import runstatus
try:
    import configparser
    from .base_model import BaseModel
except:
    import ConfigParser as configparser
    from base_model import BaseModel
try:
    import lightgbm as lgb
except:
    pass


class Lgbcf(BaseModel):
    GBM_N_ESTIMATORS = "num_boost_round"
    MODEL_ID = BaseModel.MODEL_CATE_LIGHTGBM

    def __init__(self, config=None, log_path = None):
        super(Lgbcf, self).__init__(config=config, log_path = log_path)
        self._lgb_param_ = []
        self.cate = []
        self.cf = False

    def _init_model(self):
        u'''
        init model
        :return:
        '''
        self._init_config()
        if 'categorical_feature' in self._model_params.keys():
            if len(self._model_params['categorical_feature']) > 0:
                self.cate = self._model_params['categorical_feature'].copy()
                self.cf = True
            del self._model_params['categorical_feature']
        # self._model = lgb.LGBMClassifier(**self._model_params)

        gc.collect()

    def _init_config(self):
        pass

    def train(self, x, y):
        u'''
        model train
        :param x: sample data
        :param y: label
        :return: bool
            True: suceess
            False: faild
        '''
        try:
            t_start = time.time()
            self.managerlogger.logger.info("start lightGBM..")
            if type(x) == tuple:
                if self.cf and self.cate in x[0].columns:
                    train_data = lgb.Dataset(x[0], label = y[0], categorical_feature = self.cate, free_raw_data = False)
                    valid_data = lgb.Dataset(x[1], label = y[1], categorical_feature = self.cate, free_raw_data = False)
                    self._model = lgb.train(self._model_params, train_data, valid_sets = valid_data,
                                    early_stopping_rounds = 100, verbose_eval = 100)
                else:
                    train_data = lgb.Dataset(x[0], label=y[0])
                    valid_data = lgb.Dataset(x[1], label=y[1])
                    self._model = lgb.train(self._model_params, train_data, valid_sets = valid_data,
                              early_stopping_rounds=100, verbose_eval = 100)
            else:
                if self.cf and self.cate in x.columns:
                    train_data = lgb.Dataset(x, label = y, categorical_feature = self.cate, free_raw_data=False)
                    self._model = lgb.train(self._model_params, train_data, valid_sets = train_data,
                                            early_stopping_rounds=100, verbose_eval = 100)
                else:
                    train_data = lgb.Dataset(x, label = y)
                    self._model = lgb.train(self._model_params, train_data, valid_sets = train_data,
                                            early_stopping_rounds=100, verbose_eval = 100)

            t_end = time.time()
            self.managerlogger.logger.info("finished lightGBM!")
            self.managerlogger.logger.info("lightGBM train time: %s" % (t_end - t_start))

            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("lightGBM train error: %s " % err)
            self.errorlogger.logger.error("lightGBM train error:\n %s " % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def predict(self, x):
        '''
        calculate data predict value
        :param x: sample data
        :param thresh: label classification threshold
        :return: result list, prediction label
        '''
        if self._model is None:
            return None
        #x_data = lgb.Dataset(x)
        thresh = 0.5
        res = self._model.predict(x)
        res=[0 if x < thresh else 1 for x in res]
        return np.array(res)

    def predict_proba(self, x):
        '''
        calculate data predict probability value
        :param x: sample data
        :return: list of probability value [[ ]]
        '''
        if self._model is None:
            return None
        # x_data = lgb.Dataset(x)
        res = self._model.predict(x)
        return res

    def get_feature_importance(self, feature):
        '''
        Get feature weights of user data
        :return: type:list, [[feature, score]]
        '''
        try:
            # feature_importance = []
            # importance = self._model.get_score(importance_type='gain')
            # for key in importance:
            #     feature_importance.append([importance[key], key])
            res = list(zip(self._model.feature_importance(importance_type='split'), self._model.feature_name()))
            #res = zip(self._model.feature_importance(), feature)
            return res

        except Exception as err:
            self.managerlogger.logger.error("lightGBM get feature importance error: %s" % err)
            self.errorlogger.logger.error("lightGBM get feature importance error:\n %s" % traceback.format_exc())
            return None