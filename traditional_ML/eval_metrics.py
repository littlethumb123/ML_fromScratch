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
        return 

    def recall(self, y_true, y_pred):
        """
        Recall = true_positive / true_positive + false_negative
        """
        tp = np.sum((y_true == 1)&(y_pred == 1))
        fn = np.sum((y_true == 1)&(y_pred == 0))
    
        return tp/(tp+fn)
    
    def f1_score(self, y_score, y_true)
    
    def specificity(self, y_true, y_pred):
        tn = np.sum((y_true == 0)&(y_pred = 0))
        fp = np.sum((y_true == 0)&(y_pred = 1))
        return tn/(tn + fp)
    def roc_auc(self, y_score, y_true):
        """
        Calculate true positive and false positive rate
        True positive rate = TP/TP+FN (Recall)
        False positive rate = FP/FP+TN (Specificity)
        """
        y_score = np.array(y_score)
        y_true = np.array(y_true)
        
        threshold = list(range(0, 1.01, 0.05))
        tpr_val = []
        fpr_val = []

        for i in threshold:
            y_pred = np.where(y_score > i, 1, 0)
            tpr = self.recall(y_true, y_pred)
            fpr = self.specificity(y_true, y_pred)
            tpr_val.append(tpr)
            fpr_val.append(fpr)

        # Sort by increasing false positive rate
        sorted_indices = np.argsort(tpr_val)
        tpr_val = np.array(tpr_val)[sorted_indices]
        fpr_val = np.array(fpr_val)[sorted_indices]

        auc_score = np.trapz(tpr_val, fpr_val)


        return auc_score


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