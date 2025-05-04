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
        True positive rate
        Recall = true_positive / true_positive + false_negative
        """
        return np.sum(y_true == 1 & y_pred == 1)/np.sum(y_true)
    
    def specificity(self, y_true, y_pred):
        """
        True negative rate
        Specificity = true_negative / true_negative + false_positive
        """
        return np.sum(y_true == 0 & y_pred == 0)/np.sum(y_true == 0)

    def roc_auc(self, y_true, y_prob):
        """
        True positive (recall) and false positive rates (1 - specificity)
        """
        threshold = np.arange(0, 1.1, 0.1)
        tpr = []
        fpr = []
        
        for t in threshold:
            # True positive rate = TP / (TP + FN)
            y_pred = (y_prob >= t).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            tpr.append(tp/(tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp/(fp + tn) if (fp + tn) > 0 else 0)

        # sort by fpr make sure the list ranked ascendingly
        sort_index = np.argsort(fpr)
        fpr = fpr[sort_index]
        tpr = tpr[sort_index]
        return np.trapz(tpr, fpr)

        



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