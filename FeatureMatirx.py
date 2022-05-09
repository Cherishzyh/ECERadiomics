import os
import cv2
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import transform


def GenerateFeatureMatrix():
    feature_csv = r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\JSPH\features.csv'
    label_csv = r'X:\StoreFormatData\ProstateCancerECE\ECE-ROI.csv'

    label_df = pd.read_csv(label_csv, encoding='gbk', index_col='case')
    feature_df = pd.read_csv(feature_csv)

    label_list = []
    for index in feature_df.index:
        case = feature_df.loc[index]['CaseName']
        label_list.append(label_df.loc[case]['pECE'])

    feature_df.insert(loc=1, column='label', value=label_list)
    feature_df.to_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\JSPH\features_matrix.csv', index=False)
# GenerateFeatureMatrix()


def GenerateClinicalMatrix():
    radiomics_csv = r'C:\Users\ZhangYihong\Desktop\ECE_Radiomics\feature_matrix\features_matrix.csv'
    clinical_csv = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ECE-JSPH-clinical_report.csv'

    clinical_df = pd.read_csv(clinical_csv)
    radiomics_df = pd.read_csv(radiomics_csv)

    new_clinical_df = clinical_df

    case_list = radiomics_df['CaseName'].tolist()
    for index in clinical_df.index:
        case = clinical_df.loc[index]['case']
        if case in case_list:
            continue
        else:
            new_clinical_df.drop(index=[index], inplace=True)

    new_clinical_df.to_csv(r'C:\Users\ZhangYihong\Desktop\ECE_Radiomics\clinical_features\clinical_features.csv')
# GenerateClinicalMatrix()


def CombineFeatures():
    # t2_feature = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\External\Attention\features_t2_shape.csv')
    # adc_feature = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features.csv')
    t2_feature = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features.csv')
    adc_feature = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\SUH\label.csv')
    new_feature = pd.merge(adc_feature, t2_feature, on='CaseName')
    new_feature.to_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features.csv', index=False)
# CombineFeatures()
#


# adc_pro_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features_pro_adc.csv', index_col='CaseName')
# adc_pca_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features_pca_adc.csv', index_col='CaseName')
# adc_bg_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features_bg_adc.csv', index_col='CaseName')
# t2_pro_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features_pro_t2.csv', index_col='CaseName')
# t2_pca_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features_pca_t2.csv', index_col='CaseName')
# t2_bg_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features_bg_t2.csv', index_col='CaseName')
# label_feature = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\SUH\label.csv')
# #
# adc_pro_df.rename(columns=dict(zip(adc_pro_df.columns, ['{}_pro'.format(column) for column in adc_pro_df.columns])), inplace=True)
# adc_pca_df.rename(columns=dict(zip(adc_pca_df.columns, ['{}_pca'.format(column) for column in adc_pca_df.columns])), inplace=True)
# adc_bg_df.rename(columns=dict(zip(adc_bg_df.columns, ['{}_bg'.format(column) for column in adc_bg_df.columns])), inplace=True)
# t2_pro_df.rename(columns=dict(zip(t2_pro_df.columns, ['{}_pro'.format(column) for column in t2_pro_df.columns])), inplace=True)
# t2_pca_df.rename(columns=dict(zip(t2_pca_df.columns, ['{}_pca'.format(column) for column in t2_pca_df.columns])), inplace=True)
# t2_bg_df.rename(columns=dict(zip(t2_bg_df.columns, ['{}_bg'.format(column) for column in t2_bg_df.columns])), inplace=True)
#
# new_df = pd.merge(adc_bg_df, t2_bg_df, on='CaseName')
# new_df = pd.merge(new_df, adc_pro_df, on='CaseName')
# new_df = pd.merge(new_df, t2_pro_df, on='CaseName')
# new_df = pd.merge(new_df, adc_pca_df, on='CaseName')
# new_df = pd.merge(new_df, t2_pca_df, on='CaseName')
#
# # new_df = pd.merge(label_feature, new_df, on='CaseName')
# new_df.to_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features.csv')

print()

# adc_pro_df = pd.read_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\feature\firstorder_pro_right.csv', index_col='CaseName')
# adc_pca_df = pd.read_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\feature\firstorder_pca_right.csv', index_col='CaseName')
# adc_bg_df = pd.read_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\feature\firstorder_bg_right.csv', index_col='CaseName')
# t2_pro_df = pd.read_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\feature\shape_t2_right.csv', index_col='CaseName')
# label_df = pd.read_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\features_with_label.csv', index_col='CaseName', usecols=['label'])
#
#
# adc_pro_df.rename(columns=dict(zip(adc_pro_df.columns, ['{}_pro'.format(column) for column in adc_pro_df.columns])), inplace=True)
# adc_pca_df.rename(columns=dict(zip(adc_pca_df.columns, ['{}_pca'.format(column) for column in adc_pca_df.columns])), inplace=True)
# adc_bg_df.rename(columns=dict(zip(adc_bg_df.columns, ['{}_bg'.format(column) for column in adc_bg_df.columns])), inplace=True)
# t2_pro_df.rename(columns=dict(zip(t2_pro_df.columns, ['{}_pro'.format(column) for column in t2_pro_df.columns])), inplace=True)
#
# new_df = pd.merge(adc_pro_df, adc_pca_df, on='CaseName')
# new_df = pd.merge(new_df, adc_bg_df, on='CaseName')
# new_df = pd.merge(new_df, t2_pro_df, on='CaseName')


# new_df = pd.merge(label_df, new_df, on='CaseName')
# new_df.to_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\features.csv', index=False)
#
# print()
#
df_1 = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\ECERadomicSupp\Region\features.csv', index_col='CaseName')
df_2 = pd.read_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\train_numeric_feature.csv', index_col='CaseName')
[print(column) for column in df_2.columns]

print([column for column in df_2.columns if column not in df_1.columns])
print([column for column in df_1.columns if column not in df_2.columns])
print(df_1.columns == df_2.columns)