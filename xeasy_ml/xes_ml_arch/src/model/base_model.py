# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from sklearn.linear_model import LogisticRegression
import pickle
import os
import traceback
from ..ml_utils import runstatus
from ..ml_utils import global_pre
from ..systemlog import sysmanagerlog, syserrorlog

try:
    import xgboost as xgb
except:
    pass


class BaseModel(object):
    """This basic class encapsulates the methods of model definition, model training, model prediction,
        model saving, and model loading, etc.

    Parameters
    --------
    config: configparser.ConfigParser(), default = None
        Model configuration file.

    Attributes
    ----------
    _model : ml model object
        used which model

    _model_params : json or dict = {key: word}
        used model params.
            eg.{
                 "bst:max_depth": 10,
                 "bst:eta": 1,
                 "silent": 1,
                 "objective":"binary:logistic",
                 "learning_rate": 0.1,
                 "n_estimators": 3000,
                 "num_boost_round": 100,
                 "min_child_weight": 1,
                 "subsample": 0.9,
                 "colsample_bytree": 0.9,
                 "nthread": 4
                }

    _config : ConfigParser.ConfigParser(),
        The config of model.

    _is_trained : bool, True or False
        flag of train

    _mode_file : file with path
        store trained model to _mode_file

    _feature_weight_file : int
        The number of outputs when ``fit`` is performed.

    _log : system log

    _logerror: error debug log
    """

    MODEL_XGB = "xgb"
    MODEL_SKLEARN_XGB = "sklearn_xgb"
    MODEL_LIGHTGBM = "lightgbm"
    MODEL_CATE_LIGHTGBM = 'lightgbm_cf'
    MODEL_SKLEARN_XGB_REG = "sklearn_xgb_reg"
    MODEL_SVM = "svm"
    MODEL_SVR = "svr"
    MODEL_SCV = "svc"
    MODEL_RF = "rf"
    MODEL_LR = "lr"
    MODEL_DST = "dst"
    MODEL_GBDT = "gbdt"
    MODEL_LINE_REG = "line_reg"
    MODEL_PARAM = "train_param"
    MODEL_NAME = "model_name"
    MODEL_FILE = "model_file"
    FEATURE_WEIGHT_FILE = "feature_weight_file"
    BASE_CONF = "base_conf"
    PARAMS = "params"

    MODEL_ID = MODEL_LR

    def __init__(self, config=None, log_path = None):
        # used model
        self._model = None
        # flag of train
        self._is_trained = False
        # config
        self._config = config
        # model params
        self._model_params = {}

        self.xeasy_log_path = log_path
        # model file
        self._mode_file = None

        # feature importance
        self._feature_weight_file = None

        # system log
        self.managerlogger = sysmanagerlog.SysManagerLog(__file__,self.xeasy_log_path)
        self.errorlogger = syserrorlog.SysErrorLog(__file__,self.xeasy_log_path)

    def _init_model(self):
        """Define the initial model, default LR."""

        self._model = LogisticRegression()

    def init(self):
        """Initialize model, enclude init params and new model object.

        Returns
        --------
        :return: bool
            True: suceess
            False: faild
        """
        try:
            self._init_params()
            self._init_model()
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("init model error: %s" % err)
            self.errorlogger.logger.error("init model error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def train(self, x, y):
        """Model train, call the methods in the subclass.

        Parameters
        --------
        x: pandas.DataFrame of shape (n_sample, n_features)
            sample data
        y: array-like of shape (n_samples,)
            Target vector relative to X.
        """

        self._is_trained = True

    def predict(self, x):
        """Data predict, call the methods in the subclass, if subclass don't have methods, run this

        Parameters
        --------
        x: pandas.DataFrame of shape (n_sample, n_features)
            test data.

        Returns
        --------
        :return: predict results
        """
        pass

    def predict_proba(self, x):
        """Calculate data predict probability value
        You can call the methods in the subclass, if subclass don't have methods, run this function.

        Parameters
        --------
        x: pandas.DataFrame of shape (n_sample, n_features)
            test data

        Returns
        --------
        :return: predict probability value
        """
        pass

    def get_feature_importance(self, feature):
        """Get user feature weights.
        You can call the methods in the subclass, if subclass don't have methods, run this.

        Parameters
        --------
        feature: list
            All features list.

        Returns
        --------
        :return: list, [[feature, score]]
        """
        pass

    def store_model(self, model_file=None):
        """Store model file.

        Parameters
        --------
        model_file: str
            Name of stored model file.

        Returns
        --------
        :return: bool
            True: suceess
            False: faild
        """
        try:
            if model_file is None:
                model_file = self._mode_file
            if isinstance(self._model, xgb.sklearn.XGBClassifier):
                self._model.save_model(self._mode_file)
            else:
                pickle.dump(self._model, open(model_file, global_pre.Global.FILE_WRITE))
            self.managerlogger.logger.info("store model succeed!")
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("store model error: %s" % err)
            self.errorlogger.logger.error("store model error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def load_model(self, model_file=None):
        """Load model from model file.

        Parameters
        --------
        model_file: str
            Name of exist model file.

        Returns
        --------
        :return: bool
            True: suceess
            False: faild
        """
        try:
            if model_file is None:
                model_file = self._mode_file
            if isinstance(self._model, xgb.sklearn.XGBClassifier):
                self._model.load_model(self._mode_file)
            else:
                self._model = pickle.load(open(model_file, global_pre.Global.FILE_READ))
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("load model error: %s" % err)
            self.errorlogger.logger.error("load model error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def store_feature_importance(self, feature):
        """Store feature importance to local file.

        Parameters
        --------
        feature: list
            List of feature_name.

        Returns
        --------
        :return: bool
            True: suceess
            False: faild
        """
        score = self.get_feature_importance(feature)
        try:
            score = sorted(score, key=lambda x: x[0], reverse=True)
            with open(self._feature_weight_file, "w") as file_handle:
                for data in score:
                    file_handle.write(str(data[1]) + "\t" + str(data[0]) + "\n")
            self.managerlogger.logger.info("store_feature_importance succeed")
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("store_feature_importance error: %s" % err)
            self.errorlogger.logger.error("store_feature_importance error:\n %s" % traceback.format_exc())
        return runstatus.RunStatus.FAILED

    def _init_params(self):
        """Initialize model params, get model params from config file."""

        # if not config model params, use default
        if self.MODEL_ID not in self._config.sections():
            self._model_params = {}
            self.managerlogger.logger.info("%s model use default params")
        else:
            self._model_params = eval(self._config.get(self.MODEL_ID, self.PARAMS))

        self._mode_file = self._config.get(self.BASE_CONF, self.MODEL_FILE)
        self._feature_weight_file = self._config.get(self.BASE_CONF, self.FEATURE_WEIGHT_FILE)
        self._mode_file = os.path.join(global_pre.RES_PATH, self._mode_file)
        self._feature_weight_file = os.path.join(global_pre.RES_PATH, self._feature_weight_file)
        if not os.path.isdir(os.path.dirname(self._mode_file)):
            os.mkdir(os.path.dirname(self._mode_file))
        if not os.path.isdir(os.path.dirname(self._feature_weight_file)):
            os.mkdir(os.path.dirname(self._feature_weight_file))


class SklearnModel(BaseModel):
    """A model based on SklearnModel.

    Parameters
    ----------
    config: configparser.ConfigParser()
        config should contains model param, otherwise use the default
    """

    def __init__(self, config=None, log_path = None):
        super(SklearnModel, self).__init__(config=config, log_path = log_path)

    def predict(self, x):
        """Calculate data predict result.

        Parameters
        --------
        x: pandas.DataFrame with shape (n_samples, n_features)
            Features of test data.

        Returns
        --------
        :return: array like
            Predict result.
        """
        # if self._model is not None:
        try:
            return self._model.predict(x)
        except:
            self.errorlogger.logger.error("predict error:\n %s" % traceback.format_exc())
            return None

    def predict_proba(self, x):
        """Calculate data predict probability value.

        Parameters
        --------
        x: pandas.DataFrame with shape (n_samples, n_features)
            Features of test data.

        Returns
        --------
        res: list
            List of probability value belonging to a certain class.
        """
        # if self._model is not None:
        try:
            res = self._model.predict_proba(x)
            # return [x[1] for x in res]
            return res
        except:
            self.errorlogger.logger.error('predict_proba error:\n %s' % traceback.format_exc())
            return None

    def get_feature_importance(self, feature):
        """Get weights of user feature.

        Parameters
        --------
        feature: list
            List of feature_name.

        Returns
        --------
        res: list
            Feature importance list [[feature, score]].
        """
        try:
            res = zip(self._model.coef_[0], feature)
            return res
        except Exception as e:
            self.managerlogger.logger.error("get_feature_importance error:\n %s" % e)
            self.errorlogger.logger.error("get_feature_importance error:\n %s " % traceback.format_exc())
            return None
