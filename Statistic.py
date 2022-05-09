import os

import pandas as pd
import numpy as np

from scipy.stats import wilcoxon, mannwhitneyu, friedmanchisquare, chi2_contingency

from MeDIT.Statistics import BinaryClassification


def TTestAge(train_age, test_age):
    print("train:", np.mean(train_age), np.std(train_age),
          np.quantile(train_age, 0.25, interpolation='lower'), np.quantile(train_age, 0.75, interpolation='higher'))
    print("test:", np.mean(test_age), np.std(test_age),
          np.quantile(test_age, 0.25, interpolation='lower'), np.quantile(test_age, 0.75, interpolation='higher'))
    print(mannwhitneyu(train_age, test_age, alternative='two-sided'))


def CountGrade(Gs_list, grade_list):
    num_list = []
    for grade in grade_list:
        if grade == grade_list[-1]:
            num_list.append(len([case for case in Gs_list if case >= grade]))
        else:
            num_list.append(len([case for case in Gs_list if case == grade]))
    print(num_list)
    return num_list


def Countpsa(train_psa, test_psa):
    print("train:", np.mean(train_psa), np.std(train_psa),
          np.quantile(train_psa, 0.25, interpolation='lower'), np.quantile(train_psa, 0.75, interpolation='higher'))
    print("test:", np.mean(test_psa), np.std(test_psa),
          np.quantile(test_psa, 0.25, interpolation='lower'), np.quantile(test_psa, 0.75, interpolation='higher'))
    print(mannwhitneyu(train_psa, test_psa, alternative='two-sided'))


def CountNP(data_list):
    positive = np.sum(data_list)
    negative = len(data_list) - np.sum(data_list)
    print(positive, negative)
    return positive, negative


def ClinicalInfo():
    ece_csv_path = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ECE-ROI.csv'
    case_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    case_list = os.listdir(case_folder)
    test_ref = r'X:\FAEFormatData\ECE\ZYH0521\test_ref_right.csv'
    test_ref_df = pd.read_csv(test_ref, index_col='CaseName')
    del_path = r'X:\FAEFormatData\ECE\ZYH0521\drop_data.csv'
    del_list = np.squeeze(pd.read_csv(del_path).values)

    test_list = test_ref_df.index.tolist()
    train_list = [case for case in case_list if case not in test_list]
    train_list = [case for case in train_list if case not in del_list]

    clinical_info = pd.read_csv(ece_csv_path, encoding='gbk', index_col='case')
    age_train_list, age_test_list = [], []
    psa_train_list, psa_test_list = [], []
    GS_train_list, GS_test_list = [], []
    ECE_train_list, ECE_test_list = [], []

    for case in test_list:
        age_train_list.append(int(clinical_info.loc[case]['age']))
        psa_train_list.append(float(clinical_info.loc[case]['psa']))
        GS_train_list.append(int(clinical_info.loc[case]['bGs']))
        ECE_train_list.append(int(clinical_info.loc[case]['pECE']))
    for case in train_list:
        age_test_list.append(int(clinical_info.loc[case]['age']))
        psa_test_list.append(float(clinical_info.loc[case]['psa']))
        GS_test_list.append(int(clinical_info.loc[case]['bGs']))
        ECE_test_list.append(int(clinical_info.loc[case]['pECE']))
    TTestAge(age_train_list, age_test_list)
    print()
    Countpsa(psa_train_list, psa_test_list)
    print()

    train = CountGrade(GS_train_list, [1, 2, 3, 4])
    test = CountGrade(GS_test_list, [1, 2, 3, 4])
    print((train[0])/sum(train), train[1]/sum(train), train[2]/sum(train), train[3]/sum(train))
    print((test[0])/sum(test), test[1]/sum(test), test[2]/sum(test), test[3]/sum(test))


def ROC4DiffTh():
    # atten_1 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.1\Result\Mean\PCC\RFE_3\LDA\cv_val_prediction.csv'
    # atten_2 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.2\Result\Mean\PCC\RFE_2\SVM\cv_val_prediction.csv'
    # atten_3 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.3\Result\Mean\PCC\RFE_13\LR\cv_val_prediction.csv'
    # atten_4 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.4\Result\Zscore\PCC\RFE_12\LR\cv_val_prediction.csv'
    # atten_5 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.5\Result\Mean\PCC\RFE_13\SVM\cv_val_prediction.csv'

    atten_1 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.1\Result\Mean\PCC\RFE_3\LDA\train_prediction.csv'
    atten_2 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.2\Result\Mean\PCC\RFE_2\SVM\train_prediction.csv'
    atten_3 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.3\Result\Mean\PCC\RFE_13\LR\train_prediction.csv'
    atten_4 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.4\Result\Zscore\PCC\RFE_12\LR\train_prediction.csv'
    atten_5 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.5\Result\Mean\PCC\RFE_13\SVM\train_prediction.csv'

    df_1 = pd.read_csv(atten_1, index_col=0)
    label_1 = df_1['Label'].values.astype(int).tolist()
    pred_1 = df_1['Pred'].values.tolist()

    df_2 = pd.read_csv(atten_2, index_col=0)
    label_2 = df_2['Label'].values.astype(int).tolist()
    pred_2 = df_2['Pred'].values.tolist()

    df_3 = pd.read_csv(atten_3, index_col=0)
    label_3 = df_3['Label'].values.astype(int).tolist()
    pred_3 = df_3['Pred'].values.tolist()

    df_4 = pd.read_csv(atten_4, index_col=0)
    label_4 = df_4['Label'].values.astype(int).tolist()
    pred_4 = df_4['Pred'].values.tolist()

    df_5 = pd.read_csv(atten_5, index_col=0)
    label_5 = df_5['Label'].values.astype(int).tolist()
    pred_5 = df_5['Pred'].values.tolist()

    # state, p = friedmanchisquare(pred_1, pred_2, pred_3, pred_4, pred_5)
    # print(p)
    #
    # state, p = friedmanchisquare(pred_1, pred_2, pred_3, pred_5)
    # print(p)
    #
    # state, p = friedmanchisquare(pred_1, pred_2, pred_4, pred_5)
    # print(p)
    #
    # state, p = friedmanchisquare(pred_1, pred_3, pred_4, pred_5)
    # print(p)
    #
    # state, p = friedmanchisquare(pred_3, pred_4, pred_5)
    # print(p)

    print(wilcoxon(pred_1, pred_2))

    # alpha = 0.05
    # if p > alpha:
    #     print('Same distributions (fail to reject H0)')
    # else:
    #     print('Different distributions (reject H0)')
# ROC4DiffTh()


def AUCDiff(type='train'):
    if type == 'train':
        model_1 = r'X:\FAEFormatData\ECE\ZYH\ProWithoutGLCM\Result\Mean\PCC\KW_16\SVM\train_prediction.csv'
        model_2 = r'X:\FAEFormatData\ECE\ZYH\PcaWithoutGLCM\Result\Zscore\PCC\KW_12\LR\train_prediction.csv'
        model_3 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.2\Result\Mean\PCC\RFE_2\SVM\train_prediction.csv'
        model_4 = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result\Mean\PCC\RFE_14\SVM\train_prediction.csv'
    else:
        model_1 = r'X:\FAEFormatData\ECE\ZYH\ProWithoutGLCM\Result\Mean\PCC\KW_16\SVM\test_prediction.csv'
        model_2 = r'X:\FAEFormatData\ECE\ZYH\PcaWithoutGLCM\Result\Zscore\PCC\KW_12\LR\test_prediction.csv'
        model_3 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.2\Result\Mean\PCC\RFE_2\SVM\test_prediction.csv'
        model_4 = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result\Mean\PCC\RFE_14\SVM\test_prediction.csv'

    pro_df = pd.read_csv(model_1, index_col=0)
    pca_df = pd.read_csv(model_2, index_col=0)
    atten_df = pd.read_csv(model_3, index_col=0)
    region_df = pd.read_csv(model_4, index_col=0)

    pro = pro_df['Pred'].values.tolist()
    pca = pca_df['Pred'].values.tolist()
    atten = atten_df['Pred'].values.tolist()
    region = region_df['Pred'].values.tolist()

    print(wilcoxon(pro, region))
    print(wilcoxon(pca, region))
    print(wilcoxon(atten, region))


def ResultCSV(type='train'):
    if type == 'train':
        model_1 = r'X:\FAEFormatData\ECE\ZYH\ProWithoutGLCM\Result\Mean\PCC\KW_16\SVM\train_prediction.csv'
        model_2 = r'X:\FAEFormatData\ECE\ZYH\PcaWithoutGLCM\Result\Zscore\PCC\KW_12\LR\train_prediction.csv'
        model_3 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.2\Result\Mean\PCC\RFE_2\SVM\train_prediction.csv'
        model_4 = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result\Mean\PCC\RFE_14\SVM\train_prediction.csv'
        save_path = r'C:\Users\ZhangYihong\Desktop\train_prediction.csv'
    else:
        model_1 = r'X:\FAEFormatData\ECE\ZYH\ProWithoutGLCM\Result\Mean\PCC\KW_16\SVM\test_prediction.csv'
        model_2 = r'X:\FAEFormatData\ECE\ZYH\PcaWithoutGLCM\Result\Zscore\PCC\KW_12\LR\test_prediction.csv'
        model_3 = r'X:\FAEFormatData\ECE\ZYH0521\atten0.2\Result\Mean\PCC\RFE_2\SVM\test_prediction.csv'
        model_4 = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result\Mean\PCC\RFE_14\SVM\test_prediction.csv'
        save_path = r'C:\Users\ZhangYihong\Desktop\test_prediction.csv'

    pro_df = pd.read_csv(model_1, index_col=0)
    pca_df = pd.read_csv(model_2, index_col=0)
    atten_df = pd.read_csv(model_3, index_col=0)
    region_df = pd.read_csv(model_4, index_col=0)


    case_name, label = [], []
    list_pro, list_pca, list_att, list_reg = [], [], [], []
    for case in pro_df.index:
        case_name.append(case)
        label.append(pro_df.loc[case]['Label'])
        list_pro.append(pro_df.loc[case]['Pred'])
        list_pca.append(pca_df.loc[case]['Pred'])
        list_att.append(atten_df.loc[case]['Pred'])
        list_reg.append(region_df.loc[case]['Pred'])
    df = pd.DataFrame({'CaseName': case_name, 'Label': label, 'PredPro': list_pro, 'PredPCa': list_pca, 'PredAtten': list_att,
                       'PredRegion': list_reg})
    df.to_csv(save_path, index=False)


def ComputeMetric(type='train'):
    if type == 'train':
        model = r'C:\Users\ZhangYihong\Desktop\train_prediction.csv'
    else:
        model = r'C:\Users\ZhangYihong\Desktop\test_prediction.csv'

    df = pd.read_csv(model, index_col=0)

    pro = df['PredPro'].values.tolist()
    pca = df['PredPCa'].values.tolist()
    atten = df['PredAtten'].values.tolist()
    region = df['PredRegion'].values.tolist()
    label = df['Label'].values.tolist()

    bc = BinaryClassification()
    # bc.Run(pro, label)
    bc.Run(pca, label)
    # bc.Run(atten, label)
    # bc.Run(region, label)


if __name__ == '__main__':
    # AUCDiff(type='test')
    # ResultCSV(type='test')
    # ComputeMetric(type='test')
    # ROC4DiffTh()
    # ClinicalInfo()
    # gs_array = np.array([[132, 124, 149, 169], [39, 31, 43, 31]])
    # print(chi2_contingency(gs_array))
    ece_array = np.array([[151, 323], [40, 104]])
    print(chi2_contingency(ece_array))
    # print(chi2_contingency(ECE_train_list, ECE_test_list))


















