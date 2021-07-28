# -*-coding:utf-8-*-
# @version: 0.0.1
# License: MIT


from collections import Iterable

from sklearn import metrics
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np


class PredictUtils(object):
    """Model prediction Class Initialization."""

    @staticmethod
    def store_feature_importance(result, store_file):
        """Store feature weight to file and sort.

        Parameters
        ----------
        result: Feature weights(float of list).
        store_file: path of store

        Returns
        -------
        sort of result.
        """
        if result is None:
            raise AttributeError("get feature weight exception: result is None")
        result.sort(key=lambda x: x[0], reverse=True)
        with open(store_file, "w") as file_handle:
            for data in result:
                file_handle.write(str(data[1]) + "\t" + str(data[0]) + "\n")
        return result

    @staticmethod
    def get_precision(label, pre_label):
        """Precision prediction score.

        In prediction, this function computes subset precision:
        the set of labels predicted for a sample must *exactly* match the corresponding set of labels in true.

        Parameters
        ----------
        label: 1d array-like, or label indicator array / sparse matrix Ground truth (correct) labels.
        pre_label: 1d array-like, or label indicator array / sparse matrix Predicted labels, as returned by a classifier.

        Returns
        -------
        Jaccard Distance score : float.
        """

        if not isinstance(label, list):
            try:
                label = list(label)
            except Exception:
                raise TypeError("label type is %s" % (type(label)))

        if not isinstance(pre_label, Iterable):
            try:
                pre_label = list(pre_label)
            except Exception:
                raise TypeError("pre_label type is %s" % (type(pre_label)))

        if len(label) != len(pre_label):
            raise ValueError("length of label and pre_label is not equal")

        return metrics.jaccard_similarity_score(label, pre_label)

    @staticmethod
    def get_model_score(target_true, target_pred):
        """Model evaluation index;Different indicators can measure the model from different perspectives.

        Parameters
        ----------
        target_true: 1d array-like, or label indicator array / sparse matrix Ground truth (correct) labels in test.
        target_pred: 1d array-like, or label indicator array / sparse matrix Predicted labels, as returned by a classifier in test.

        Returns
        -------
        Array:[Confusion metrics,Recall,Precision,F1-socre,auc]
        """

        if not isinstance(target_true, list):
            try:
                target_true = list(target_true)
            except Exception:
                raise TypeError("label type is %s" % (type(target_true)))

        if not isinstance(target_pred, list):
            try:
                target_pred = list(target_pred)
            except Exception:
                raise TypeError("pre_label type is %s" % (type(target_pred)))

        if len(target_true) != len(target_pred):
            raise ValueError("length of label and pre_label is not equal")

        try:
            confuseion = metrics.confusion_matrix(target_true, target_pred)

            # judge is binary-class or multiclass
            # if only a few samples, maybe all the labels are 0/1
            if len(set(target_true)) <= 2:
                class_type = "binary"
            else:
                class_type = "macro"
            recall = metrics.recall_score(target_true, target_pred, average=class_type)
            precision = metrics.precision_score(target_true, target_pred, average=class_type)
            f1_score = metrics.f1_score(target_true, target_pred, average=class_type)
            # jaccard = metrics.jaccard_similarity_score(target_true, target_pred)
            jaccard = metrics.jaccard_score(target_true, target_pred, average="weighted")
        except:
            confuseion = None
            recall = None
            precision=None
            f1_score=None
            jaccard=None

        try:
            msr = metrics.mean_squared_error(target_true, target_pred)
        except:
            msr = None

        res = dict(confuseion=confuseion, recall=recall, precision=precision, f1_score=f1_score,
                   jaccard=jaccard, msr=msr)

        return res


    @staticmethod
    def get_roc_score(label, prob_label):
        """Get mode auc in dateset.AUC (area under curve) is defined as the area under the ROC curve. We often use AUC
        value as the evaluation standard of the model, because often the ROC curve can not clearly indicate which
        classifier is better, but as a value, the classifier with larger AUC is better.

        Parameters
        ----------
        label: 1d array-like, or label indicator array / sparse matrix Ground truth (correct) labels.
        target_pred: 1d array-like, or label indicator array / sparse matrix Predicted labels, as returned by a classifier in test.
        prob_label

        Returns
        -------
        AUC_socre: float
        """

        try:
            if not isinstance(label, list):
                try:
                    label = list(label)
                except Exception:
                    raise TypeError("label type is %s" % (type(label)))

            if not isinstance(prob_label, list):
                try:
                    prob_label = list(prob_label)
                except Exception:
                    raise TypeError("pre_label type is %s" % (type(prob_label)))

            if len(label) != len(prob_label):
                raise ValueError("length of label and pre_label is not equal")

            if not isinstance(prob_label[0],Iterable):
                try:
                    prob_label = list(map(lambda x: np.array([1.0-abs(x),abs(x)]), prob_label))
                except Exception:
                    raise TypeError("pre_label type is %s" % (type(prob_label)))


            n_class = len(set(label))
            # if only a few samples, maybe all the labels are 0/1, then len(set(label)) ==1 ，
            #  this may occur exception in roc_auc_score. Set label[0] as a different value to avoid.
            if n_class == 1:
                label[0] = 1.0 - label[0]
                n_class = len(set(label))
            label_bin = label_binarize(label, classes=list(set(label)))

            if len(prob_label[0]) != n_class:
                raise ValueError(
                    "class num is %s, but prob dimensionality is %s" % (
                        n_class, len(prob_label[0])))

            # judge is binary-class or multiclass
            if n_class == 2:
                auc = metrics.roc_auc_score(label_bin, [x[1] for x in prob_label])
            else:
                auc = metrics.roc_auc_score(label_bin, prob_label)

            return dict(auc=auc)
        except:
            return dict(auc=-1)

    @staticmethod
    def valid_pandas_data(data):
        """Valid data type detection, is pandas DataFrame, shape is not zero.

        Parameters
        ----------
        data: input date;

        Returns
        -------
        Boolean: if input date is Dataframe and shape is not zero then True else False.
        """

        if data is None:
            return False
        if not isinstance(data, pd.DataFrame):
            return False
        if data.shape[0] == 0:
            return False
        return True
