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
    true_np = label_trues.numpy() # np.array(label_trues.cpu())
    pre_np = label_preds.numpy() # np.array(label_preds.cpu())
    if len(np.shape(true_np)) == 4:
        # for i in range(np.shape(label_trues)[1]):
        acc += (true_np[:,:,:,0:3] - pre_np[:,:,:,0:3]).mean()
        acc_cls = acc / n_class
    else:
        # for i in range(np.shape(label_trues)[0]):
        #     for j in range(np.shape(label_trues[i])):
        acc += (true_np[:,:,:,:,0:3] - pre_np[:,:,:,:,0:3]).mean()
        acc_cls = acc / (n_class*np.shape(true_np)[0])
    return acc, acc_cls, mean_iu, fwavacc
