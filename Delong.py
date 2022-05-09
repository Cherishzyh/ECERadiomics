#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
import scipy.stats
from scipy import stats


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
   """Computes midranks.
   Args:
      x - a 1D numpy array
   Returns:
      array of midranks
   """
   J = np.argsort(x)
   Z = x[J]
   N = len(x)
   T = np.zeros(N, dtype=np.float32)
   i = 0
   while i < N:
       j = i
       while j < N and Z[j] == Z[i]:
           j += 1
       T[i:j] = 0.5*(i + j - 1)
       i = j
   T2 = np.empty(N, dtype=np.float32)
   # Note(kazeevn) +1 is due to Python using 0-based indexing
   # instead of 1-based in the AUC formula in the paper
   T2[J] = T + 1
   return T2


def compute_midrank_weight(x, sample_weight):
   """Computes midranks.
   Args:
      x - a 1D numpy array
   Returns:
      array of midranks
   """
   J = np.argsort(x)
   Z = x[J]
   cumulative_weight = np.cumsum(sample_weight[J])
   N = len(x)
   T = np.zeros(N, dtype=np.float)
   i = 0
   while i < N:
       j = i
       while j < N and Z[j] == Z[i]:
           j += 1
       T[i:j] = cumulative_weight[i:j].mean()
       i = j
   T2 = np.empty(N, dtype=np.float)
   T2[J] = T
   return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight=None):

   if sample_weight is None:

       return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)

   else:

       return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):

   """

   The fast version of DeLong's method for computing the covariance of

   unadjusted AUC.

   Args:

      predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]

         sorted such as the examples with label "1" are first

   Returns:

      (AUC value, DeLong covariance)

   Reference:

    @article{sun2014fast,

      title={Fast Implementation of DeLong's Algorithm for

             Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},

      author={Xu Sun and Weichao Xu},

      journal={IEEE Signal Processing Letters},

      volume={21},

      number={11},

      pages={1389--1393},

      year={2014},

      publisher={IEEE}

    }

   """

   # Short variables are named as they are in the paper

   m = label_1_count

   n = predictions_sorted_transposed.shape[1] - m

   positive_examples = predictions_sorted_transposed[:, :m]

   negative_examples = predictions_sorted_transposed[:, m:]

   k = predictions_sorted_transposed.shape[0]



   tx = np.empty([k, m], dtype=np.float)

   ty = np.empty([k, n], dtype=np.float)

   tz = np.empty([k, m + n], dtype=np.float)

   for r in range(k):

       tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])

       ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])

       tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)

   total_positive_weights = sample_weight[:m].sum()

   total_negative_weights = sample_weight[m:].sum()

   pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])

   total_pair_weights = pair_weights.sum()

   aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights

   v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights

   v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights

   sx = np.cov(v01)

   sy = np.cov(v10)

   delongcov = sx / m + sy / n

   return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):

   """

   The fast version of DeLong's method for computing the covariance of

   unadjusted AUC.

   Args:

      predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]

         sorted such as the examples with label "1" are first

   Returns:

      (AUC value, DeLong covariance)

   Reference:

    @article{sun2014fast,

      title={Fast Implementation of DeLong's Algorithm for

             Comparing the Areas Under Correlated Receiver Oerating

             Characteristic Curves},

      author={Xu Sun and Weichao Xu},

      journal={IEEE Signal Processing Letters},

      volume={21},

      number={11},

      pages={1389--1393},

      year={2014},

      publisher={IEEE}

    }

   """

   # Short variables are named as they are in the paper

   m = label_1_count
   n = predictions_sorted_transposed.shape[1] - m
   positive_examples = predictions_sorted_transposed[:, :m]
   negative_examples = predictions_sorted_transposed[:, m:]
   k = predictions_sorted_transposed.shape[0]



   tx = np.empty([k, m], dtype=np.float32)
   ty = np.empty([k, n], dtype=np.float32)
   tz = np.empty([k, m + n], dtype=np.float32)

   for r in range(k):

       tx[r, :] = compute_midrank(positive_examples[r, :])
       ty[r, :] = compute_midrank(negative_examples[r, :])
       tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

   aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
   v01 = (tz[:, :m] - tx[:, :]) / n
   v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

   sx = np.cov(v01)
   sy = np.cov(v10)
   delongcov = sx / m + sy / n

   return aucs, delongcov


def calc_pvalue(aucs, sigma):
   """Computes log(10) of p-values.
   Args:
      aucs: 1D array of AUCs
      sigma: AUC DeLong covariances
   Returns:
      log10(pvalue)

   """

   l = np.array([[1, -1]])

   z = np.abs(np.diff(aucs)) / (np.sqrt(np.dot(np.dot(l, sigma), l.T)) + 1e-8)
   pvalue = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
   return pvalue


def compute_ground_truth_statistics(ground_truth, sample_weight=None):
   assert np.array_equal(np.unique(ground_truth), [0, 1])
   order = (-ground_truth).argsort()
   label_1_count = int(ground_truth.sum())
   if sample_weight is None:
       ordered_sample_weight = None
   else:
       ordered_sample_weight = sample_weight[order]

   return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
   """
   Computes ROC AUC variance for a single set of predictions
   Args:
      ground_truth: np.array of 0 and 1
      predictions: np.array of floats of the probability of being class 1
   """
   sample_weight = None
   order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
       ground_truth, sample_weight)
   predictions_sorted_transposed = predictions[np.newaxis, order]
   aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)

   assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
   return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
   """
   Computes log(p-value) for hypothesis that two ROC AUCs are different
   Args:
      ground_truth: np.array of 0 and 1
      predictions_one: predictions of the first model,
         np.array of floats of the probability of being class 1
      predictions_two: predictions of the second model,
         np.array of floats of the probability of being class 1
   """
   sample_weight = None
   order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(ground_truth)
   predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
   aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count,sample_weight)

   return calc_pvalue(aucs, delongcov)


def delong_roc_ci(y_true, y_pred):
   aucs, auc_cov = delong_roc_variance(y_true, y_pred)
   auc_std = np.sqrt(auc_cov)
   lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
   ci = stats.norm.ppf(
       lower_upper_q,
       loc=aucs,
       scale=auc_std)
   ci[ci > 1] = 1
   return aucs, ci


if __name__ == '__main__':
    #examples 具体用法

    alpha = .95

    # train_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\train_prediction.csv'
    # train_df = pd.read_csv(train_csv, index_col=0)
    # train_list_1 = train_df['PredPro'].values.tolist()
    # train_list_2 = train_df['PredPCa'].values.tolist()
    # train_list_3 = train_df['PredAtten'].values.tolist()
    # train_list_4 = train_df['PredRegion'].values.tolist()
    # train_label_list = train_df['Label'].astype(int).values.tolist()
    #
    # internal_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\internal_prediction.csv'
    # internal_df = pd.read_csv(internal_csv, index_col=0)
    # internal_list_1 = internal_df['PredPro'].values.tolist()
    # internal_list_2 = internal_df['PredPCa'].values.tolist()
    # internal_list_3 = internal_df['PredAtten'].values.tolist()
    # internal_list_4 = internal_df['PredRegion'].values.tolist()
    # internal_label_list = internal_df['Label'].astype(int).values.tolist()
    #
    # external_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\external_prediction.csv'
    # external_df = pd.read_csv(external_csv, index_col=0)
    # external_list_1 = external_df['PredPro'].values.tolist()
    # external_list_2 = external_df['PredPCa'].values.tolist()
    # external_list_3 = external_df['PredAtten'].values.tolist()
    # external_list_4 = external_df['PredRegion'].values.tolist()
    # external_label_list = external_df['Label'].astype(int).values.tolist()
    #
    # pvalue_1 = delong_roc_test(np.array(train_label_list), np.array(train_list_1), np.array(train_list_2))
    # pvalue_2 = delong_roc_test(np.array(train_label_list), np.array(train_list_1), np.array(train_list_3))
    # pvalue_3 = delong_roc_test(np.array(train_label_list), np.array(train_list_1), np.array(train_list_4))
    # pvalue_4 = delong_roc_test(np.array(train_label_list), np.array(train_list_2), np.array(train_list_3))
    # pvalue_5 = delong_roc_test(np.array(train_label_list), np.array(train_list_2), np.array(train_list_4))
    # pvalue_6 = delong_roc_test(np.array(train_label_list), np.array(train_list_3), np.array(train_list_4))
    # print('Train')
    # print(pvalue_1)
    # print(pvalue_2)
    # print(pvalue_3)
    # print(pvalue_4)
    # print(pvalue_5)
    # print(pvalue_6)
    #
    # pvalue_1 = delong_roc_test(np.array(internal_label_list), np.array(internal_list_1), np.array(internal_list_2))
    # pvalue_2 = delong_roc_test(np.array(internal_label_list), np.array(internal_list_1), np.array(internal_list_3))
    # pvalue_3 = delong_roc_test(np.array(internal_label_list), np.array(internal_list_1), np.array(internal_list_4))
    # pvalue_4 = delong_roc_test(np.array(internal_label_list), np.array(internal_list_2), np.array(internal_list_3))
    # pvalue_5 = delong_roc_test(np.array(internal_label_list), np.array(internal_list_2), np.array(internal_list_4))
    # pvalue_6 = delong_roc_test(np.array(internal_label_list), np.array(internal_list_3), np.array(internal_list_4))
    # print('Internal')
    # print(pvalue_1)
    # print(pvalue_2)
    # print(pvalue_3)
    # print(pvalue_4)
    # print(pvalue_5)
    # print(pvalue_6)
    #
    # pvalue_1 = delong_roc_test(np.array(external_label_list), np.array(external_list_1), np.array(external_list_2))
    # pvalue_2 = delong_roc_test(np.array(external_label_list), np.array(external_list_1), np.array(external_list_3))
    # pvalue_3 = delong_roc_test(np.array(external_label_list), np.array(external_list_1), np.array(external_list_4))
    # pvalue_4 = delong_roc_test(np.array(external_label_list), np.array(external_list_2), np.array(external_list_3))
    # pvalue_5 = delong_roc_test(np.array(external_label_list), np.array(external_list_2), np.array(external_list_4))
    # pvalue_6 = delong_roc_test(np.array(external_label_list), np.array(external_list_3), np.array(external_list_4))
    # print('External')
    # print(pvalue_1)
    # print(pvalue_2)
    # print(pvalue_3)
    # print(pvalue_4)
    # print(pvalue_5)
    # print(pvalue_6)



    # data_folder = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\atten.csv'
    # df = pd.read_csv(data_folder, index_col='CaseName')
    # case_list = df.index.tolist()
    # pred1_list = df['Pred0.1'].values.tolist()
    # pred2_list = df['Pred0.2'].values.tolist()
    # pred3_list = df['Pred0.3'].values.tolist()
    # pred4_list = df['Pred0.4'].values.tolist()
    # pred5_list = df['Pred0.5'].values.tolist()
    # label_list = df['Label'].values.astype(int).tolist()
    #
    #
    #
    # pvalue_1 = delong_roc_test(np.array(label_list), np.array(pred1_list), np.array(pred2_list))
    # pvalue_2 = delong_roc_test(np.array(label_list), np.array(pred1_list), np.array(pred3_list))
    # pvalue_3 = delong_roc_test(np.array(label_list), np.array(pred1_list), np.array(pred4_list))
    # pvalue_4 = delong_roc_test(np.array(label_list), np.array(pred1_list), np.array(pred5_list))
    # pvalue_5 = delong_roc_test(np.array(label_list), np.array(pred2_list), np.array(pred3_list))
    # pvalue_6 = delong_roc_test(np.array(label_list), np.array(pred2_list), np.array(pred4_list))
    # pvalue_7 = delong_roc_test(np.array(label_list), np.array(pred2_list), np.array(pred5_list))
    # pvalue_8 = delong_roc_test(np.array(label_list), np.array(pred3_list), np.array(pred4_list))
    # pvalue_9 = delong_roc_test(np.array(label_list), np.array(pred3_list), np.array(pred5_list))
    # pvalue_10 = delong_roc_test(np.array(label_list), np.array(pred4_list), np.array(pred5_list))
    #
    # print(pvalue_1)
    # print(pvalue_2)
    # print(pvalue_3)
    # print(pvalue_4)
    # print(pvalue_5)
    # print(pvalue_6)
    # print(pvalue_7)
    # print(pvalue_8)
    # print(pvalue_9)
    # print(pvalue_10)

    from scipy.stats import wilcoxon

    # resnet_train = pd.read_csv(r'Y:\RenJi\TVT\Model\ResNet_1123_mask\test.csv', index_col='CaseName')
    # label = resnet_train['Label'].values.astype(int).tolist()
    # preds_norm = resnet_train['Pred'].values.tolist()
    #
    # ch2_train = pd.read_csv(r'Y:\RenJi\TVT\Model\ResNet_1210_2CH_mask\test.csv', index_col='CaseName')
    # label_ch2 = ch2_train['Label'].values.astype(int).tolist()
    # preds_ch2 = ch2_train['Pred'].values.tolist()
    #
    # ch3_train = pd.read_csv(r'Y:\RenJi\TVT\Model\ResNet_1210_3CH_mask\test.csv', index_col='CaseName')
    # label_ch3 = ch3_train['Label'].values.astype(int).tolist()
    # preds_ch3 = ch3_train['Pred'].values.tolist()
    #
    # seg_train = pd.read_csv(r'Y:\RenJi\TVT\Model\ResNet_1210_mask_SliceBySeg\test.csv', index_col='CaseName')
    # label_seg = seg_train['Label'].values.astype(int).tolist()
    # preds_seg = seg_train['Pred'].values.tolist()
    #
    # dimension3_train = pd.read_csv(r'Y:\RenJi\TVT\Model\ResNet3D_1123_mask\test.csv', index_col='CaseName')
    # label_3D = dimension3_train['Label'].values.astype(int).tolist()
    # preds_3D = dimension3_train['Pred'].values.tolist()
    #
    # assert (label_seg == label_ch3 == label_ch2 == label == label_3D)

    # pvalue_1 = delong_roc_test(np.array(label), np.array(preds_norm), np.array(preds_3D))
    # pvalue_2 = delong_roc_test(np.array(label), np.array(preds_norm), np.array(preds_seg))
    # pvalue_3 = delong_roc_test(np.array(label), np.array(preds_norm), np.array(preds_ch2))
    # pvalue_4 = delong_roc_test(np.array(label), np.array(preds_norm), np.array(preds_ch3))
    #
    # print(pvalue_1)
    # print(pvalue_2)
    # print(pvalue_3)
    # print(pvalue_4)
    # model_name = ['ResNet_1123_mask', 'ResNet_1210_2CH_mask', 'ResNet_1210_3CH_mask', 'ResNet_1210_mask_SliceBySeg',
    #               'ResNet3D_1123_mask']
    # from sklearn.metrics import roc_auc_score, roc_curve
    # for model in model_name:
    #     model_folder = os.path.join(r'Y:\RenJi\TVT\Model', model)
    #     external = pd.read_csv(os.path.join(model_folder, 'external.csv'), index_col='CaseName')
    #     external_revision = pd.read_csv(os.path.join(model_folder, 'external_revision.csv'), index_col='CaseName')
    #     preds = external['Pred'].values.tolist()
    #     preds_revision = external_revision['Pred'].values.tolist()
    #     label = external['Label'].values.tolist()
    #     fpn, sen, the = roc_curve(label, preds_revision)
    #     auc = roc_auc_score(label, preds_revision)
    #     pvalue = delong_roc_test(np.array(label), np.array(preds), np.array(preds_revision))
    #     print(model, auc, pvalue)

    df = pd.read_csv(r'C:\Users\ZhangYihong\Documents\WeChat Files\wxid_dh6xwqxfg7ny21\FileStorage\File\2022-03\deep learning 1year 5 year ROC alltrain model multiply 3.csv', index_col='CaseName')
    # train_list = pd.read_csv(r'Y:\RenJi\alltrain_name.csv', index_col='CaseName').index.tolist()
    test_list = pd.read_csv(r'Y:\RenJi\SuccessfulModel\ResNet_1123\test.csv', index_col='CaseName').index.tolist()
    label = df.loc[test_list, 'Label'].tolist()
    model = df.loc[test_list, 'Pred'].tolist()
    exp1 = df.loc[test_list, '1 year experience'].tolist()
    exp2 = df.loc[test_list, '5 year experience'].tolist()
    pvalue1 = delong_roc_test(np.array(label), np.array(model), np.array(exp1))
    pvalue2 = delong_roc_test(np.array(label), np.array(model), np.array(exp2))
    pvalue3 = delong_roc_test(np.array(label), np.array(exp1), np.array(exp2))
    print(pvalue1, pvalue2, pvalue3)
    #
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import roc_auc_score, roc_curve
    #
    # plt.figure(0, figsize=(6, 6))
    # plt.plot([0, 1], [0, 1], 'k--')
    # linewidth = 2
    #
    # fpn, sen, _ = roc_curve(label, model)
    # auc = roc_auc_score(label, model)
    # plt.plot(fpn, sen, label='CNN: {:.3f}'.format(auc), linewidth=linewidth)
    # fpn, sen, _ = roc_curve(label, exp1)
    # auc = roc_auc_score(label, exp1)
    # plt.plot(fpn, sen, label='Expert 1: {:.3f}'.format(auc), linewidth=linewidth)
    # fpn, sen, _ = roc_curve(label, exp2)
    # auc = roc_auc_score(label, exp2)
    # plt.plot(fpn, sen, label='Expert 2: {:.3f}'.format(auc), linewidth=linewidth)
    #
    # plt.xlabel('1 - specificity', fontsize=16)
    # plt.ylabel('sensitivity', fontsize=16)
    # plt.legend(loc='lower right', fontsize=16)
    #
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(r'C:\Users\ZhangYihong\Desktop\train_roc.jpg', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # # plt.show()









