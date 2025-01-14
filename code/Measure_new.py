from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,roc_auc_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric

def measure_final_score(dataset_orig_test, dataset_orig_predict,privileged_groups,unprivileged_groups):

    y_test = dataset_orig_test.labels
    y_pred = dataset_orig_predict.labels

    accuracy = accuracy_score(y_test, y_pred)
    recall1 = recall_score(y_test, y_pred, pos_label=1)
    recall0 = recall_score(y_test, y_pred, pos_label=0)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision1 = precision_score(y_test, y_pred, pos_label=1)
    precision0 = precision_score(y_test, y_pred, pos_label=0)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score1 = f1_score(y_test, y_pred, pos_label=1)
    f1score0 = f1_score(y_test, y_pred, pos_label=0)
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    #This line is important for the following code completion.
    classified_metric_pred = ClassificationMetric(dataset_orig_test, dataset_orig_predict,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)

    # To do begin
    # Read the IBM AIF360 document and use the AIF360 fairness metrics calculation function to fill up the following three blanks.
    spd = to_do_1
    aod = to_do_2
    eod = to_do_3
    # To do end



    erd = abs(classified_metric_pred.error_rate_difference())

    return accuracy, recall1, recall0, recall_macro, precision1, precision0, precision_macro, f1score1, f1score0, f1score_macro, mcc, spd,aod,eod,erd
