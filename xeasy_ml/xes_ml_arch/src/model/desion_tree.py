# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT

from sklearn.tree import DecisionTreeClassifier
from . import base_model
import traceback
from sklearn import tree
import pydotplus
from ..ml_utils import runstatus


class MyDesionTree(base_model.SklearnModel):
    """
        A decision tree classifier. The encapsulation form is sklearn model.

        Parameters
        --------
        config: configparser.ConfigParser()
            Configuration file for model initialization.

        Examples
        --------
        >>> from xes_ml_arch.src.model import desion_tree
        >>> import configparser
        >>> from sklearn.datasets import load_iris
        >>> config = configparser.ConfigParser()
        >>> config.read("desion_ress.conf")
        >>> clf = desion_tree.MyDesionTree(config=config)
        >>> iris = load_iris()
        >>> clf.train(iris.data, iris.target)
        ...
        ...
        """

    MODEL_ID = base_model.SklearnModel.MODEL_DST

    def __init__(self, config=None, log_path = None):
        super(MyDesionTree, self).__init__(config=config, log_path = log_path)

    def _init_model(self):
        """Initialize model.

        Returns
        --------
        :return: bool
            True: suceess
            False: faild
        """
        try:
            self._model = DecisionTreeClassifier(**self._model_params)
            return True
        except Exception as e:
            self.managerlogger.logger.error("init model error:\n %s" % e)
            self.errorlogger.logger.error("init model error:\n %s" % traceback.format_exc())
            return False

    def train(self, x, y):
        """Train a model use a decision tree.

        Parameters
        --------
        x: pandas.DataFrame of shape (n_sample, n_features)
            sample data
        y: array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        --------
        :return: bool
            True: suceess
            False: faild
        """
        try:
            self._model.fit(x, y)
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("train model error: %s" % (err))
            self.errorlogger.logger.error("train model error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def show_dot(self, file, feature):
        """Create decision tree visualization file, and save it as file.

        Parameters
        ------
        file: str
            file to store dot data and pdf
        feature: list
            List of feature names.

        Returns
        --------
        :return: bool, runstatus.RunStatus
            runstatus.RunStatus.SUCC: success
            runstatus.RunStatus.Faild: Faild
        """
        try:
            dot_data = tree.export_graphviz(self._model, feature_names=[str(x) for x in feature],
                                            class_names=[str(x) for x in self._model.classes_],
                                            out_file=None)

            # store bin tree, data an pdf
            with open(file + ".txt", "w") as f:
                f.write(dot_data)
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_pdf(file + ".pdf")
            self.managerlogger.logger.info("show dot succeed! ")
            return runstatus.RunStatus.SUCC
        except Exception as err:
            self.managerlogger.logger.error("show dot error: %s" % err)
            self.errorlogger.logger.error("show dot error:\n %s" % traceback.format_exc())
            return runstatus.RunStatus.FAILED

    def get_feature_importance(self, feature):
        """Get weights of user features.

        Parameters
        --------
        feature: list
            List of feature names.

        Returns
        --------
        res: dict
            Feature importance, like : dict([feature_importance, feature name])

        """
        try:
            res = zip(self._model.feature_importances_, feature)
            return res
        except Exception as e:
            self.managerlogger.logger.error("get_feature_importance error: %s" % e)
            self.errorlogger.logger.error('get_feature_importance error:\n %s' % traceback.format_exc())
            return None
