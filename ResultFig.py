import os
from copy import deepcopy
import numpy as np
import SimpleITK as sitk
import pandas as pd
from sklearn import metrics
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *

from MeDIT.Statistics import BinaryClassification

color_patten = sns.color_palette()


def Auc(y_true, y_pred, ci_index=0.95):
    single_auc = metrics.roc_auc_score(y_true, y_pred)

    bootstrapped_scores = []

    np.random.seed(42)  # control reproducibility
    seed_index = np.random.randint(0, 65535, 1000)
    for seed in seed_index.tolist():
        np.random.seed(seed)
        pred_one_sample = np.random.choice(y_pred, size=y_pred.size, replace=True)
        np.random.seed(seed)
        label_one_sample = np.random.choice(y_true, size=y_pred.size, replace=True)

        if len(np.unique(label_one_sample)) < 2:
            continue

        score = metrics.roc_auc_score(label_one_sample, pred_one_sample)
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    std_auc, mean_auc = np.std(sorted_scores), np.mean(sorted_scores)

    ci = stats.norm.interval(ci_index, loc=mean_auc, scale=std_auc)
    return single_auc, sorted_scores, ci


def ROC4DiffTh(save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    atten_1 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.1\Result\Mean\PCC\RFE_3\LDA\train_prediction.csv'
    atten_2 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.2\Result\Mean\PCC\RFE_2\SVM\train_prediction.csv'
    atten_3 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.3\Result\Mean\PCC\RFE_13\LR\train_prediction.csv'
    atten_4 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.4\Result\Zscore\PCC\RFE_12\LR\train_prediction.csv'
    atten_5 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.5\Result\Mean\PCC\RFE_13\SVM\train_prediction.csv'

    df_1 = pd.read_csv(atten_1, index_col=0)
    label_1 = df_1['Label'].values.astype(int)
    pred_1 = df_1['Pred'].values

    df_2 = pd.read_csv(atten_2, index_col=0)
    label_2 = df_2['Label'].values.astype(int)
    pred_2 = df_2['Pred'].values

    df_3 = pd.read_csv(atten_3, index_col=0)
    label_3 = df_3['Label'].values.astype(int)
    pred_3 = df_3['Pred'].values

    df_4 = pd.read_csv(atten_4, index_col=0)
    label_4 = df_4['Label'].values.astype(int)
    pred_4 = df_4['Pred'].values

    df_5 = pd.read_csv(atten_5, index_col=0)
    label_5 = df_5['Label'].values.astype(int)
    pred_5 = df_5['Pred'].values


    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')
    linewidth = 2

    fpn, sen, the = roc_curve(label_1, pred_1)
    auc = roc_auc_score(label_1, pred_1)
    plt.plot(fpn, sen, label='阈值=0.1: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(label_2, pred_2)
    auc = roc_auc_score(label_2, pred_2)
    plt.plot(fpn, sen, label='阈值=0.2: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(label_3, pred_3)
    auc = roc_auc_score(label_3, pred_3)
    plt.plot(fpn, sen, label='阈值=0.3: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(label_4, pred_4)
    auc = roc_auc_score(label_4, pred_4)
    plt.plot(fpn, sen, label='阈值=0.4: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(label_5, pred_5)
    auc = roc_auc_score(label_5, pred_5)
    plt.plot(fpn, sen, label='阈值=0.5: {:.3f}'.format(auc), linewidth=linewidth)


    plt.xlabel('1 - 特异度', fontsize=16)
    plt.ylabel('敏感度', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(save_path + '.jpg', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # plt.show()
# ROC4DiffTh(r'C:\Users\ZhangYihong\Desktop\diffth')


def ROC4DiffDataset(save_path, type='train'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    model = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\{}_prediction.csv'.format(type)

    df = pd.read_csv(model, index_col=0)

    pro = df['PredPro'].values.tolist()
    pca = df['PredPCa'].values.tolist()
    atten = df['PredAtten'].values.tolist()
    region = df['PredRegion'].values.tolist()
    label = df['Label'].values.tolist()


    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')
    linewidth = 2

    fpn, sen, the = roc_curve(label, pro)
    auc = roc_auc_score(label, pro)
    plt.plot(fpn, sen, label='腺体: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(label, pca)
    auc = roc_auc_score(label, pca)
    plt.plot(fpn, sen, label='癌灶: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(label, atten)
    auc = roc_auc_score(label, atten)
    plt.plot(fpn, sen, label='注意力: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(label, region)
    auc = roc_auc_score(label, region)
    plt.plot(fpn, sen, label='子区域: {:.3f}'.format(auc), linewidth=linewidth)


    plt.xlabel('1 - 特异度', fontsize=16)
    plt.ylabel('敏感度', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(save_path + '.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    # plt.show()
# ROC4DiffModel(type='train')
ROC4DiffDataset(save_path=r'C:\Users\ZhangYihong\Desktop\diffmodel', type='internal')


def ROC4DiffModel(save_path, model='PredPro'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    train_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\train_prediction.csv'
    internal_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\internal_prediction.csv'
    external_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\external_prediction.csv'

    train_df = pd.read_csv(train_csv, index_col=0)
    internal_df = pd.read_csv(internal_csv, index_col=0)
    external_df = pd.read_csv(external_csv, index_col=0)

    train_list = train_df[model].values.tolist()
    internal_list = internal_df[model].values.tolist()
    external_list = external_df[model].values.tolist()

    train_label_list = train_df['Label'].values.tolist()
    internal_label_list = internal_df['Label'].values.tolist()
    external_label_list = external_df['Label'].values.tolist()

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')
    linewidth = 2

    fpn, sen, the = roc_curve(train_label_list, train_list)
    auc = roc_auc_score(train_label_list, train_list)
    plt.plot(fpn, sen, label='train: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(internal_label_list, internal_list)
    auc = roc_auc_score(internal_label_list, internal_list)
    plt.plot(fpn, sen, label='internal test: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(external_label_list, external_list)
    auc = roc_auc_score(external_label_list, external_list)
    plt.plot(fpn, sen, label='external test: {:.3f}'.format(auc), linewidth=linewidth)

    plt.title(model, fontsize=16)
    plt.xlabel('1 - specificity', fontsize=16)
    plt.ylabel('sensitivity', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(save_path, '{}.jpg'.format(model)), dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.close()

# ROC4DiffModel(save_path=r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Fig of lunwen', model='PredPro')
# ROC4DiffModel(save_path=r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Fig of lunwen', model='PredPCa')
# ROC4DiffModel(save_path=r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Fig of lunwen', model='PredAtten')
# ROC4DiffModel(save_path=r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Fig of lunwen', model='PredRegion')


def FeatureWeights(save_path):
    # 形状特征去掉T2和灰阶去掉first
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    weights_path = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result\Mean\PCC\RFE_14\SVM\SVM_coef.csv' # SubRegion
    # weights_path = r'X:\FAEFormatData\ECE\ZYH\ProWithoutGLCM\Result\Mean\PCC\KW_16\SVM\SVM_coef.csv'  # Prostate
    # weights_path = r'X:\FAEFormatData\ECE\ZYH\PcaWithoutGLCM\Result\Zscore\PCC\KW_12\LR\LR_coef.csv'  # Pca
    # weights_path = r'X:\FAEFormatData\ECE\ZYH0521\atten0.2\Result\Mean\PCC\RFE_2\SVM\SVM_coef.csv'  # Atten
    weights_df = pd.read_csv(weights_path, index_col=0)
    weights_df = weights_df.sort_values(by='Coef')

    features = weights_df.index.values.tolist()
    weights = weights_df['Coef'].values.tolist()

    new_features = []
    color_list = []
    for feature in features:
        part = feature.split('_')
        if len(part) == 5:
            if 'pca' in feature:
                if 'shape' in feature:
                    new_features.append('Shape_{}_{}'.format(part[-2], 'PCa'))
                else:
                    new_features.append('{}_{}_{}'.format(part[0].upper(), part[-2], 'PCa'))
                color_list.append(color_patten[0])
            elif 'pro' in feature:
                if 'shape' in feature:
                    new_features.append('Shape_{}_{}'.format(part[-2], 'Pro'))
                else:
                    new_features.append('{}_{}_{}'.format(part[0].upper(), part[-2], 'Pro'))
                color_list.append(color_patten[1])
            elif 'bg' in feature:
                if 'shape' in feature:
                    new_features.append('Shape_{}_{}'.format(part[-2], 'BG'))
                else:
                    new_features.append('{}_{}_{}'.format(part[0].upper(), part[-2], 'BG'))
                color_list.append(color_patten[2])
            else:
                print(feature)
        else:
            if 'shape' in feature:
                new_features.append('Shape_{}'.format(part[-1]))
            elif 'firstorder' in feature:
                new_features.append('{}_{}'.format(part[0].upper(), part[-1]))
            color_list.append(color_patten[3])


    y_pos = np.arange(len(features))

    plt.figure(0, figsize=(6, 5))
    plt.barh(y_pos, weights, align='center', color=color_list)
    # plt.barh(y_pos, weights, align='center')
    for index, weight in enumerate(weights):
        if weights[index] < 0.:
            plt.text(0, y_pos[index], '{:.3f}'.format(weight), color="black", ha='left', va="center", fontsize=14)
        else:
            plt.text(0, y_pos[index], '{:.3f}'.format(weight), color="black", ha='right', va="center", fontsize=14)

    plt.yticks(y_pos, new_features, fontsize=12)
    plt.xlabel('feature weights of model SubRegion', fontsize=14)
    plt.xticks(fontsize=12)
    plt.savefig(save_path + '.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    # plt.show()
# FeatureWeights(save_path=r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Fig of lunwen\weights_region')


def BoxPlot(save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    atten1_path = r'X:\FAEFormatData\ECE\ZYH0521\atten0.1\Result\Mean\PCC\RFE_3\LDA\train_prediction.csv'
    atten2_path = r'X:\FAEFormatData\ECE\ZYH0521\atten0.2\Result\Mean\PCC\RFE_2\SVM\train_prediction.csv'
    atten3_path = r'X:\FAEFormatData\ECE\ZYH0521\atten0.3\Result\Mean\PCC\RFE_13\LR\train_prediction.csv'
    atten4_path = r'X:\FAEFormatData\ECE\ZYH0521\atten0.4\Result\Zscore\PCC\RFE_12\LR\train_prediction.csv'
    atten5_path = r'X:\FAEFormatData\ECE\ZYH0521\atten0.5\Result\Mean\PCC\RFE_13\SVM\train_prediction.csv'

    atten1_df = pd.read_csv(atten1_path, index_col=0)
    atten2_df = pd.read_csv(atten2_path, index_col=0)
    atten3_df = pd.read_csv(atten3_path, index_col=0)
    atten4_df = pd.read_csv(atten4_path, index_col=0)
    atten5_df = pd.read_csv(atten5_path, index_col=0)

    atten1 = atten1_df['Pred'].values.tolist()
    atten2 = atten2_df['Pred'].values.tolist()
    atten3 = atten3_df['Pred'].values.tolist()
    atten4 = atten4_df['Pred'].values.tolist()
    atten5 = atten5_df['Pred'].values.tolist()
    assert (atten5_df['Label'].values.tolist() == atten4_df['Label'].values.tolist()
            == atten3_df['Label'].values.tolist()
            == atten2_df['Label'].values.tolist()
            == atten1_df['Label'].values.tolist())

    label = atten5_df['Label'].values.tolist()
    auc_1, atten1_auc_list, _ = Auc(np.asarray(label), np.asarray(atten1), ci_index=0.95)
    auc_2, atten2_auc_list, _ = Auc(np.asarray(label), np.asarray(atten2), ci_index=0.95)
    auc_3, atten3_auc_list, _ = Auc(np.asarray(label), np.asarray(atten3), ci_index=0.95)
    auc_4, atten4_auc_list, _ = Auc(np.asarray(label), np.asarray(atten4), ci_index=0.95)
    auc_5, atten5_auc_list, _ = Auc(np.asarray(label), np.asarray(atten5), ci_index=0.95)
    print(auc_1, auc_2, auc_3, auc_4, auc_5)

    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.boxplot([atten1_auc_list, atten2_auc_list, atten3_auc_list, atten4_auc_list, atten5_auc_list],
                labels=['0.1', '0.2', '0.3', '0.4', '0.5'],
                showfliers=False)
    ax1.set_ylabel('AUC', fontsize=16)
    ax1.set_xlabel('阈值', fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot([1, 2, 3, 4, 5], [3, 2, 13, 12, 13], c='g')
    ax2.set_ylabel('特征数', fontsize=16)
    #

    ax1.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    plt.savefig(save_path + '.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    # plt.show()
# BoxPlot(save_path=r'C:\Users\ZhangYihong\Desktop\box')
#

def VolinPlot():
    df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\test_prediction.csv', index_col=0)
    label = df['Label'].to_numpy().squeeze().tolist()
    new_label = label + label + label + label
    new_pred = df['PredPro'].to_numpy().squeeze().tolist() + df['PredPCa'].to_numpy().squeeze().tolist()  + df[
        'PredAtten'].to_numpy().squeeze().tolist() + df['PredRegion'].to_numpy().squeeze().tolist()
    new_name = ['Prostate' for _ in range(len(label))] + ['PCa' for _ in range(len(label))] + ['Attention' for _ in range(
        len(label))] + ['Sub Region' for _ in range(len(label))]
    new_df = pd.DataFrame({
                'Prediction': new_pred,
                'Label': new_label,
                'Distribution': new_name
            })
    sns.violinplot(x='Distribution', y='Prediction', hue='Label',
                           data=new_df, split=True, scale='area', inner="quartile")
    plt.show()
# VolinPlot()


def CompareCSV():
    # train_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultSub\train_prediction.csv', index_col='CaseName')
    # internal_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultSub\test_prediction.csv', index_col='CaseName')
    # external_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultSub\external_prediction.csv', index_col='CaseName')

    train_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultPCa\train_prediction.csv', index_col='CaseName')
    internal_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultPCa\test_prediction.csv', index_col='CaseName')
    external_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultPCa\external_prediction.csv', index_col='CaseName')

    train = train_df['Pred'].values.tolist()
    internal = internal_df['Pred'].values.tolist()
    external = external_df['Pred'].values.tolist()

    train_label = train_df['Label'].values.tolist()
    internal_label = internal_df['Label'].values.tolist()
    external_label = external_df['Label'].values.tolist()


    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')
    linewidth = 2

    fpn, sen, the = roc_curve(train_label, train)
    auc = roc_auc_score(train_label, train)
    plt.plot(fpn, sen, label='train: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(internal_label, internal)
    auc = roc_auc_score(internal_label, internal)
    plt.plot(fpn, sen, label='internal: {:.3f}'.format(auc), linewidth=linewidth)
    fpn, sen, the = roc_curve(external_label, external)
    auc = roc_auc_score(external_label, external)
    plt.plot(fpn, sen, label='external: {:.3f}'.format(auc), linewidth=linewidth)


    plt.xlabel('1 - Specificity', fontsize=16)
    plt.ylabel('Sensitivity', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\AUC_pca' + '.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.05)

# CompareCSV()

def Statistics():
    from Delong import delong_roc_test
    train_df_sub = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultSub\train_prediction.csv', index_col='CaseName')
    internal_df_sub = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultSub\test_prediction.csv', index_col='CaseName')
    external_df_sub = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultSub\external_prediction.csv', index_col='CaseName')

    train_sub = train_df_sub['Pred'].values.tolist()
    internal_sub = internal_df_sub['Pred'].values.tolist()
    external_sub = external_df_sub['Pred'].values.tolist()

    train_sub_label = train_df_sub['Label'].values.tolist()
    internal_sub_label = internal_df_sub['Label'].values.tolist()
    external_sub_label = external_df_sub['Label'].values.tolist()


    train_df_pca = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultPCa\train_prediction.csv', index_col='CaseName')
    internal_df_pca = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultPCa\test_prediction.csv', index_col='CaseName')
    external_df_pca = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultPCa\external_prediction.csv', index_col='CaseName')

    train_pca = train_df_pca['Pred'].values.tolist()
    internal_pca = internal_df_pca['Pred'].values.tolist()
    external_pca = external_df_pca['Pred'].values.tolist()

    train_pca_label = train_df_pca['Label'].values.tolist()
    internal_pca_label = internal_df_pca['Label'].values.tolist()
    external_pca_label = external_df_pca['Label'].values.tolist()

    # bc = BinaryClassification()
    # print('train')
    # bc.Run(train_sub, train_sub_label)
    # bc.Run(train_pca, train_pca_label)
    # print('internal')
    # bc.Run(internal_sub, internal_sub_label)
    # bc.Run(internal_pca, internal_pca_label)
    # print('external')
    # bc.Run(external_sub, external_sub_label)
    # bc.Run(external_pca, external_pca_label)

    assert (train_pca_label == train_sub_label)
    assert (internal_pca_label == internal_sub_label)
    assert (external_pca_label == external_sub_label)

    pvalue_1 = delong_roc_test(np.array(train_sub_label), np.array(train_sub), np.array(train_pca))
    pvalue_2 = delong_roc_test(np.array(internal_sub_label), np.array(internal_sub), np.array(internal_pca))
    pvalue_3 = delong_roc_test(np.array(external_sub_label), np.array(external_sub), np.array(external_pca))

    print('p_value_train:', pvalue_1)
    print('p_value_internal:', pvalue_2)
    print('p_value_external:', pvalue_3)
# Statistics()

# train_df = pd.read_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result\Mean\PCC\RFE_14\SVM\cv_train_prediction.csv', index_col='CaseName')
# test_df = pd.read_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result\Mean\PCC\RFE_14\SVM\cv_val_prediction.csv', index_col='CaseName')
# test_df = pd.read_csv(r'X:\FAEFormatData\ECE\ZYH\PCaWithoutGLCM\Result\Zscore\PCC\KW_12\LR\cv_val_prediction.csv', index_col='CaseName')
#
#
# train = []
# label = []
# case_list = []
# for case in list(set(test_df.index)):
#     train.append(test_df.loc[case, 'Pred'])
#     label.append(test_df.loc[case, 'Label'])
#     case_list.append(case)
#
# new_df = pd.DataFrame({'CaseName': case_list, 'Pred': train, 'Label': label})
# new_df.to_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\ResultPCa\cv_val.csv', index=False)









