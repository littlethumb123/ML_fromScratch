####### ML evaluation metrics #######
# 1. Accuracy
# 2. Precision
# 3. Recall
# 4. F1 Score
# 5. ROC Curve
# 6. AUC
# Confusion matrix
# 7. TPR = True positive/True positive + False negative
# 8. TNR = True negative/True negative + False positive
# 9. FPR = False positive/False positive + True negative
# 10. FNR = False negative/False negative + True positive
import numpy as np
import pandas as pd

class EvalMetrics:
    def __type_check(self, y_true, y_pred):

        if len(y_true) != len(y_pred):
            raise ValueError("Length of y_true and y_pred must be the same.")
        if not isinstance(y_true, (np.ndarray, pd.Series)):
            y_ = np.array(y_true)
        if not isinstance(y_pred, (np.ndarray, pd.Series)):
            y_pred = np.array(y_pred)
        return y_true, y_pred
    
    def accuracy(self, y_true, y_pred):
        y_true, y_pred = self.__type_check(y_true, y_pred)

        return (y_true == y_pred).mean()
    
    def precision(self, y_true, y_pred):
        """
        Precision = true_positive / true_positive + false_positive
        """
        return np.sum(y_true == 1 & y_pred == 1)/np.sum(y_pred)

    def recall(self, y_true, y_pred):
        """
        Recall = true_positive / true_positive + false_negative
        """
        return np.sum(y_true == 1 & y_pred == 1)/np.sum(y_true)

class TestCase:
    def test_accuracy(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
        em = EvalMetrics()
        print(em.accuracy(y_true, y_pred))
    def test_precision(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
        em = EvalMetrics()
        print(em.precision(y_true, y_pred))
    def test_recall(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
        em = EvalMetrics()
        print(em.recall(y_true, y_pred))

if __name__ == '__main__':
    tc = TestCase()
    tc.test_accuracy()
    tc.test_precision()
    tc.test_recall()