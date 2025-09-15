from sklearn.metrics import precision_score, recall_score, make_scorer

def weighted_recall_precision(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return 0.55 * recall + 0.45 * precision

weighted_scorer = make_scorer(weighted_recall_precision, greater_is_better=True)