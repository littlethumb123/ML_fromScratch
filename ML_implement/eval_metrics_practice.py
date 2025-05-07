### calculation of evaluation metrics
# Tips
# 1. Remember to add np.sum() to the end of the calculation to get the sum of the array.
# 2. np.where to get the y_pred through y_prob
# 3. sort the fpr array to get the correct order, using np.argsort()
# 4. np.trapz() to calculate the area under the curve (AUC) using the trapezoidal rule.
# 5. np.arange(0, 1.01, 0.01) to create an array of evenly spaced values within a given interval.
class EvalMetrics: 
    def accuracy(self, y_true, y_pred):
        """
        Accuracy = 
        TP + TN/(TP + TN + FP + FN)
        """
        tp = np.sum((y_true == 1)&(y_pred == 1).astype(int))
        tn = np.sum((y_true == 0)&(y_pred == 0).astype(int))
        fp = np.sum((y_true == 0)&(y_pred == 1).astype(int))
        fn = np.sum((y_true == 1)&(y_pred == 0).astype(int))
        total = len(y_true)
        return (tp + tn)/(total)
    
    def precision(self, y_true, y_pred):
        """
        precision = TP/(TP + FP)
        """
        tp = np.sum((y_true == 1)&(y_pred == 1).astype(int))
        fp = np.sum((y_true == 0)&(y_pred == 1).astype(int))
        if tp + fp == 0: return None
        return tp/(tp + fp)
    
    def recall(self, y_true, y_pred):
        """
        recall: tp/tp + fn
        """
        tp = np.sum((y_true == 1)&(y_pred == 1).astype(int))
        fn = np.sum((y_true == 1)&(y_pred == 0).astype(int))
        if tp + fn == 0: return None
        return tp/(tp + fn)
    
    def specificity(self, y_true, y_pred):
        """
        Specificity: tn/(tn + fp)
        """
        tn = np.sum((y_true == 0)&(y_pred == 0).astype(int))
        fp = np.sum((y_true == 0)&(y_pred == 1).astype(int))
        if tn + fp == 0: return None
        return tn/(tn + fp)
    
    def roc_auc(self, y_true, y_prob):
        """
        y = recall
        x = 1 - specificity
        summary metrics for classifier capability
        out of all positive cases, its ability the psoitve
        out of all negative cases, its ability test the positve
        """
        threshold = np.arange(0, 1.01, 0.01)
        tprs = []
        fprs = []
        for i in threshold:
            # y_pred = int(y_prob > i)
            y_pred = np.where(y_prob > i, 1, 0)
            tpr = self.recall(y_true, y_pred)
            fpr = 1 - self.specificity(y_true, y_pred)
            tprs.append(tpr)
            fprs.append(fpr)
        sort_index = np.argsort(fprs)
        tprs = np.array(tprs)[sort_index]
        fprs = np.array(fprs)[sort_index]
        auc = np.trapz(tprs, fprs, axis=0)
        return auc




if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix

    sample_size = 100
    y_true = np.random.randint(0, 2, sample_size)
    y_prob = np.random.rand(sample_size)
    y_pred = np.where(y_prob > 0.5, 1, 0)

    # calculate confusion matrix

    # calculate precision, recall, specificity, accuracy, roc_auc
    precision = EvalMetrics().precision(y_true, y_pred)
    recall = EvalMetrics().recall(y_true, y_pred)
    specificity = EvalMetrics().specificity(y_true, y_pred)
    accuracy = EvalMetrics().accuracy(y_true, y_pred)
    roc_auc = EvalMetrics().roc_auc(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    print(
        f"""
        Precision: {precision}, correct: {precision_score(y_true, y_pred)},
        Recall: {recall}, correct: {recall_score(y_true, y_pred)},
        Specificity: {specificity}, correct: {tn / (tn+fp)},
        Accuracy: {accuracy}, correct: {accuracy_score(y_true, y_pred)}, 
        ROC AUC: {roc_auc}, correct: {roc_auc_score(y_true, y_prob)}
        """
    )


