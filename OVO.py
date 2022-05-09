import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, friedmanchisquare, chi2_contingency

from MeDIT.Normalize import Normalize01
from MeDIT.Visualization import Imshow3DArray

from collections import Counter

# sta_adc = sitk.ReadImage(r'X:\PrcoessedData\ProstateCancerECE_SUH\CHEN REN CAI\t2.nii')
# print(sta_adc.GetSpacing())

# t2 = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\PrcoessedData\ProstateCancerECE_SUH\CAI XUE QI\t2_5x5.nii'))
# pro = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\PrcoessedData\ProstateCancerECE_SUH\CAI XUE QI\prostate_roi_5x5.nii.gz'))
# pca = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\PrcoessedData\ProstateCancerECE_SUH\CAI XUE QI\pca_roi_5x5.nii.gz'))
# att_roi = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\PrcoessedData\ProstateCancerECE_SUH\CAI XUE QI\3D_AttentionMap_binary_1021_0.2.nii'))
# att = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\PrcoessedData\ProstateCancerECE_SUH\CAI XUE QI\3D_AttentionMap_1021.nii'))

# t2 = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\StoreFormatData\ProstateCancerECE\ResampleData\BI JUN\t2.nii'))
# pro = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\StoreFormatData\ProstateCancerECE\ResampleData\BI JUN\ProstateROI_TrumpetNet.nii.gz'))
# pca = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\StoreFormatData\ProstateCancerECE\ResampleData\BI JUN\roi.nii'))
# att_roi = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\StoreFormatData\ProstateCancerECE\ResampleData\BI JUN\3D_AttentionMap_binary_0.2.nii'))
# att = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\StoreFormatData\ProstateCancerECE\ResampleData\BI JUN\3D_AttentionMap_0521.nii'))



# Imshow3DArray(np.concatenate([Normalize01(t2.transpose(1, 2, 0)), Normalize01(att.transpose(1, 2, 0))], axis=1),
#               roi=[np.concatenate([Normalize01(pro.transpose(1, 2, 0)), Normalize01(att_roi.transpose(1, 2, 0))], axis=1),
#                    np.concatenate([Normalize01(pca.transpose(1, 2, 0)), Normalize01(att_roi.transpose(1, 2, 0))], axis=1)])


# for case in os.listdir(r'Z:\RenJi\3CHHeathy20211012\3CH'):
#     case_folder = os.path.join(r'Z:\RenJi\3CHHeathy20211012\3CH', case)
#     if os.path.exists(os.path.join(case_folder, 'mask_3ch.nii.gz')):
#
#         data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'resize_3ch.nii.gz')))
#         roi = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'mask_3ch.nii.gz')))
#
#         plt.imshow(data[0], cmap='gray')
#         plt.contour(roi[0], color='r')
#         plt.savefig(os.path.join(r'Z:\RenJi\3CHHeathy20211012\Image', '{}.jpg'.format(case.split('.npy')[0])))
#         plt.close()


# train_name = pd.read_csv(r'D:\ZYH\ECE\ECE组学\ECERadiomics\train_prediction.csv', index_col='CaseName').index.tolist()
# df = pd.DataFrame({'CaseName': train_name})
# df.to_csv(r'D:\ZYH\ECE\ECE组学\ECERadiomics\train_name.csv', index=False)
#
# train_name = pd.read_csv(r'D:\ZYH\ECE\ECE组学\ECERadiomics\test_prediction.csv', index_col='CaseName').index.tolist()
# df = pd.DataFrame({'CaseName': train_name})
# df.to_csv(r'D:\ZYH\ECE\ECE组学\ECERadiomics\test_name.csv', index=False)



def ECEStatistics():
    case_list = os.listdir(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\AdcSlice')
    case_list = [case.split('_-_')[0] for case in case_list]
    train_list = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\alltrain-name.csv').values.tolist()[0]
    train_list = [case for case in train_list if case in case_list]
    internal_list = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\test-name.csv').values.tolist()[0]
    internal_list = [case for case in internal_list if case in case_list]

    jsph_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ECE-ROI.csv', index_col='case', encoding='gbk')
    suh_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\SUH_Dwi1500\SUH_ECE_clinical-report.csv', index_col='case', encoding='gbk')
    external_list = suh_df.index.tolist()

    train_age = np.array(jsph_df.loc[train_list, 'age'].values.tolist(), dtype=np.float32)
    internal_age = np.array(jsph_df.loc[internal_list, 'age'].values.tolist(), dtype=np.float32)
    external_age = np.array(suh_df.loc[external_list, 'age'].values.tolist(), dtype=np.float32)

    print('age')
    # print(np.mean(train_age), np.std(train_age), np.quantile(train_age, 0.25), np.quantile(train_age, 0.75))
    # print(np.mean(internal_age), np.std(internal_age), np.quantile(internal_age, 0.25), np.quantile(internal_age, 0.75))
    # print(np.mean(external_age), np.std(external_age), np.quantile(external_age, 0.25), np.quantile(external_age, 0.75))
    print(mannwhitneyu(train_age, internal_age))
    print(mannwhitneyu(train_age, external_age))
    print(mannwhitneyu(external_age, internal_age))

    train_psa = np.array(jsph_df.loc[train_list, 'psa'].values.tolist(), dtype=np.float32)
    internal_psa = np.array(jsph_df.loc[internal_list, 'psa'].values.tolist(), dtype=np.float32)
    external_psa = np.array(suh_df.loc[external_list, 'PSA'].values.tolist(), dtype=np.float32)
    print('PSA')
    # print(np.mean(train_psa), np.std(train_psa), np.quantile(train_psa, 0.25), np.quantile(train_psa, 0.75))
    # print(np.mean(internal_psa), np.std(internal_psa), np.quantile(internal_psa, 0.25), np.quantile(internal_psa, 0.75))
    # print(np.mean(external_psa), np.std(external_psa), np.quantile(external_psa, 0.25), np.quantile(external_psa, 0.75))
    print(mannwhitneyu(train_psa, internal_psa))
    print(mannwhitneyu(train_psa, external_psa))
    print(mannwhitneyu(external_psa, internal_psa))

    train_pGs = np.array(jsph_df.loc[train_list, 'bGs'].values.tolist(), dtype=np.float32)
    internal_pGs = np.array(jsph_df.loc[internal_list, 'bGs'].values.tolist(), dtype=np.float32)
    external_pGs = np.array(suh_df.loc[external_list, '手术GS grade'].values.tolist(), dtype=np.float32)
    print('pGs')
    # print('{}/({:.1f}), {}/({:.1f}), {}/({:.1f}), {}/({:.1f})'.format(sum(train_pGs == 1), sum(train_pGs == 1)/596*100,
    #                                                                   sum(train_pGs == 2), sum(train_pGs == 2)/596*100,
    #                                                                   sum(train_pGs == 3), sum(train_pGs == 3)/596*100,
    #                                                                   sum(train_pGs > 3), sum(train_pGs > 3)/596*100))
    # print('{}/({:.1f}), {}/({:.1f}), {}/({:.1f}), {}/({:.1f})'.format(sum(internal_pGs == 1), sum(internal_pGs == 1)/150*100,
    #                                                                   sum(internal_pGs == 2), sum(internal_pGs == 2)/150*100,
    #                                                                   sum(internal_pGs == 3), sum(internal_pGs == 3)/150*100,
    #                                                                   sum(internal_pGs > 3), sum(internal_pGs > 3)/150*100))
    # print('{}/({:.1f}), {}/({:.1f}), {}/({:.1f}), {}/({:.1f})'.format(sum(external_pGs == 1), sum(external_pGs == 1)/146*100,
    #                                                                   sum(external_pGs == 2), sum(external_pGs == 2)/146*100,
    #                                                                   sum(external_pGs == 3), sum(external_pGs == 3)/146*100,
    #                                                                   sum(external_pGs > 3), sum(external_pGs > 3)/146*100))
    train_pGs[train_pGs > 3] = 3
    internal_pGs[internal_pGs > 3] = 3
    external_pGs[external_pGs > 3] = 3
    train = dict(sorted(dict(Counter(train_pGs).most_common()).items(), key=lambda x: x[0])).values()
    internal = dict(sorted(dict(Counter(internal_pGs).most_common()).items(), key=lambda x: x[0])).values()
    external = dict(sorted(dict(Counter(external_pGs).most_common()).items(), key=lambda x: x[0])).values()
    print(chi2_contingency(np.stack([list(train), list(internal)], axis=0)))
    print(chi2_contingency(np.stack([list(train), list(external)], axis=0)))
    print(chi2_contingency(np.stack([list(internal), list(external)], axis=0)))

    train_pece = np.array(jsph_df.loc[train_list, 'pECE'].values.tolist(), dtype=np.float32)
    internal_pece = np.array(jsph_df.loc[internal_list, 'pECE'].values.tolist(), dtype=np.float32)
    external_pece = np.array(suh_df.loc[external_list, '包膜突破'].values.tolist(), dtype=np.float32)
    print('ece')
    # print('{}/({:.1f}), {}/({:.1f})'.format(sum(train_pece == 1), sum(train_pece == 1) / 596 * 100,
    #                                         sum(train_pece == 0), sum(train_pece == 0) / 596 * 100))
    # print('{}/({:.1f}), {}/({:.1f}),'.format(sum(internal_pece == 1), sum(internal_pece == 1) / 150 * 100,
    #                                          sum(internal_pece == 0), sum(internal_pece == 0) / 150 * 100))
    # print('{}/({:.1f}), {}/({:.1f})'.format(sum(external_pece == 1), sum(external_pece == 1) / 146 * 100,
    #                                         sum(external_pece == 0), sum(external_pece == 0) / 146 * 100))
    train = dict(sorted(dict(Counter(train_pece).most_common()).items(), key=lambda x: x[0])).values()
    internal = dict(sorted(dict(Counter(internal_pece).most_common()).items(), key=lambda x: x[0])).values()
    external = dict(sorted(dict(Counter(external_pece).most_common()).items(), key=lambda x: x[0])).values()
    print(chi2_contingency(np.stack([list(train), list(internal)], axis=0)))
    print(chi2_contingency(np.stack([list(train), list(external)], axis=0)))
    print(chi2_contingency(np.stack([list(internal), list(external)], axis=0)))
# ECEStatistics()


def ComputeMetric(model='PredPro'):
    from MeDIT.Statistics import BinaryClassification
    bc = BinaryClassification()
    train_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\train_prediction.csv'
    internal_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\internal_prediction.csv'
    external_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\external_prediction.csv'

    train_df = pd.read_csv(train_csv, index_col=0)
    internal_df = pd.read_csv(internal_csv, index_col=0)
    external_df = pd.read_csv(external_csv, index_col=0)

    train_list = train_df[model].values.tolist()
    internal_list = internal_df[model].values.tolist()
    external_list = external_df[model].values.tolist()

    train_label_list = train_df['Label'].astype(int).values.tolist()
    internal_label_list = internal_df['Label'].astype(int).values.tolist()
    external_label_list = external_df['Label'].astype(int).values.tolist()

    bc.Run(train_list, train_label_list)
    bc.Run(internal_list, internal_label_list)
    bc.Run(external_list, external_label_list)
# ComputeMetric(model='PredPro')


ComputeMetric(model='PredPCa')
print()
ComputeMetric(model='PredAtten')
print()
ComputeMetric(model='PredRegion')