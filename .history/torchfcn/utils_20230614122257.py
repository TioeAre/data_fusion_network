import numpy as np


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    fwavacc = 0
    mean_iu = 0
    acc = 0
    acc_cls = 0
    for i in range(np.shape(label_trues)[1]):
        acc += np.sub
    return acc, acc_cls, mean_iu, fwavacc