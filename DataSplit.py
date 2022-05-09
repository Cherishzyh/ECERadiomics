import os
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import transform

from BasicTool.MeDIT.SaveAndLoad import LoadImage
from BasicTool.MeDIT.Visualization import Imshow3DArray
from BasicTool.MeDIT.Normalize import Normalize01


def DataSpliter():
    csv_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide'
    test_df = pd.read_csv(os.path.join(csv_folder, 'test-name.csv'))
    test_list = test_df.values.tolist()[0]

    feature_matirx_df = r'C:\Users\ZhangYihong\Desktop\ECE_Radiomics\matrix\features_matrix.csv'
    feature_df = pd.read_csv(feature_matirx_df)
    for index in feature_df.index:
        case = feature_df.loc[index]['CaseName']
        if case not in test_list:
            continue
        else:
            feature_df.drop(labels=index, inplace=True)
    feature_df.to_csv(r'C:\Users\ZhangYihong\Desktop\ECE_Radiomics\split\train_numeric_feature.csv', index=False)
    print()


def GenerateCSV():
    path = r'X:\StoreFormatData\ProstateCancerECE\ECE-ROI.csv'
    feature_df = pd.read_csv(path, index_col='case', encoding='gbk')
    case_list = []
    label_list = []

    for case in feature_df.index:
        case_list.append(case)
        label_list.append(feature_df.loc[case]['pECE'])

    new_feature = pd.DataFrame({'case': case_list, 'label': label_list})
    new_feature.to_csv(r'C:\Users\ZhangYihong\Desktop\test\ece.csv', index=False)


def ConcatCSV():
    feature_csv_path_1 = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.3\shape_t2.csv'
    feature_csv_path_2 = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.3\firstorder_bg.csv'
    feature_csv_path_3 = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.3\firstorder_pro.csv'
    feature_csv_path_4 = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.3\firstorder_pca.csv'
    # feature_csv_path_5 = r'X:\FAEFormatData\ECE\ZYH0514\sub_region_0.2_glcm\glcm_bg_t2\features_right.csv'
    # feature_csv_path_6 = r'X:\FAEFormatData\ECE\ZYH0514\sub_region_0.2_glcm\glcm_pro_t2\features_right.csv'
    # feature_csv_path_7 = r'X:\FAEFormatData\ECE\ZYH0514\sub_region_0.2_glcm\glcm_pca_t2\features_right.csv'

    label_csv_path = r'X:\FAEFormatData\ECE\ece.csv'

    feature_df_1 = pd.read_csv(feature_csv_path_1, index_col='CaseName')
    feature_df_2 = pd.read_csv(feature_csv_path_2, index_col='CaseName')
    feature_df_3 = pd.read_csv(feature_csv_path_3, index_col='CaseName')
    feature_df_4 = pd.read_csv(feature_csv_path_4, index_col='CaseName')
    # feature_df_5 = pd.read_csv(feature_csv_path_5, index_col='CaseName')
    # feature_df_6 = pd.read_csv(feature_csv_path_6, index_col='CaseName')
    # feature_df_7 = pd.read_csv(feature_csv_path_7, index_col='CaseName')

    label_df = pd.read_csv(label_csv_path, index_col='case')

    column_list = []
    for column in feature_df_2.columns:
        column_list.append('{}_bg'.format(column))
    feature_df_2.columns = column_list

    column_list = []
    for column in feature_df_3.columns:
        column_list.append('{}_pro'.format(column))
    feature_df_3.columns = column_list

    column_list = []
    for column in feature_df_4.columns:
        column_list.append('{}_pca'.format(column))
    feature_df_4.columns = column_list

    # column_list = []
    # for column in feature_df_5.columns:
    #     column_list.append('{}_bg'.format(column))
    # feature_df_5.columns = column_list
    #
    # column_list = []
    # for column in feature_df_6.columns:
    #     column_list.append('{}_pro'.format(column))
    # feature_df_6.columns = column_list
    #
    # column_list = []
    # for column in feature_df_7.columns:
    #     column_list.append('{}_pca'.format(column))
    # feature_df_7.columns = column_list

    new_df = feature_df_1
    new_df = pd.concat([new_df, feature_df_2], axis=1)
    new_df = pd.concat([new_df, feature_df_3], axis=1)
    new_df = pd.concat([new_df, feature_df_4], axis=1)
    # new_df = pd.concat([new_df, feature_df_5], axis=1)
    # new_df = pd.concat([new_df, feature_df_6], axis=1)
    # new_df = pd.concat([new_df, feature_df_7], axis=1)

    label_list = []
    case_list = []
    for case in feature_df_4.index:
        label = label_df.loc[case].values
        label_list.append(int(label))
        case_list.append(case)

    df = pd.DataFrame({'label': label_list}, index=case_list)

    new_df.insert(loc=0, column='label', value=df)
    new_df.to_csv(r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.3\features_with_label.csv')


def AddLabel(feature_csv_path, label_csv_path, save_csv_path):
    feature_df = pd.read_csv(feature_csv_path, index_col='CaseName')
    label_df = pd.read_csv(label_csv_path, index_col='case')
    label_list = []
    case_list = []
    for case in feature_df.index:
        label = label_df.loc[case].values
        label_list.append(int(label))
        case_list.append(case)

    df = pd.DataFrame({'label': label_list}, index=case_list)
    feature_df.insert(loc=0, column='label', value=df)
    feature_df.to_csv(save_csv_path)


def DelectCasefromCSV(del_case, feature_path):
    '''
    delect the cases that when th >= 0.9, the values of pixels in the binary attention map are all zero
    :return:
    '''
    feature_matrix_df = pd.read_csv(os.path.join(feature_path, 'features_label.csv'), index_col='CaseName')
    for case in feature_matrix_df.index:
        if case in del_case:
            feature_matrix_df.drop(case, axis=0, inplace=True)
    feature_matrix_df.to_csv(os.path.join(feature_path, 'features_right.csv'))
    # feature_matrix_df = pd.read_csv(os.path.join(feature_path, 'test_ref.csv'), index_col='CaseName')
    # for case in feature_matrix_df.index:
    #     if case in del_case:
    #         feature_matrix_df.drop(case, axis=0, inplace=True)
    # feature_matrix_df.to_csv(os.path.join(feature_path, 'test_ref_right.csv'))


def testDelectCasefromCSV():
    th_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for th in th_list:
        feature_csv_path = r'X:\FAEFormatData\ECE\ZYH0514\atten{}'.format(th)
        del_path = r'X:\FAEFormatData\ECE\ZYH0514\atten0.9\features_error.csv'
        del_list = np.squeeze(pd.read_csv(del_path).values)
        DelectCasefromCSV(del_list, feature_csv_path)


def DelectFeature(feature_path):
    '''
    delect the features of shape and GLCM of adc
    :return:
    '''
    feature_matrix_df = pd.read_csv(os.path.join(feature_path, 'features_right.csv'), index_col='CaseName')
    for column in feature_matrix_df.columns:
        if 'adc' in column:
            if 'firstorder' in column:
                continue
            else:
                feature_matrix_df.drop(column, axis=1, inplace=True)
    feature_matrix_df.to_csv(os.path.join(feature_path, 'features_right_right.csv'))


def testDelectFeature():
    th_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for th in th_list:
        feature_csv_path = r'X:\FAEFormatData\ECE\ZYH0523\atten{}'.format(th)
        DelectFeature(feature_csv_path)


def SaveNoLabelCase():
    data_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    des_folder = r'X:\StoreFormatData\ProstateCancerECE\NoLabel'
    case_list = []
    for case in os.listdir(data_folder):
        case_folder = os.path.join(data_folder, case)
        des_case_folder = os.path.join(des_folder, case)
        atten_folder = os.path.join(case_folder, '3D_AttentionMap_0520.nii')
        if not os.path.exists(atten_folder):
            shutil.move(case_folder, des_case_folder)
            case_list.append(case)

    df = pd.DataFrame({'CaseName': case_list})
    df.to_csv(r'X:\FAEFormatData\ECE\ZYH0520\no_label.csv', index=False)


def CheckDelectCase():
    'case that has no label(label all equal to zero) may have nonECE'
    del_path = r'X:\FAEFormatData\ECE\ZYH0521\atten0.7\features_error.csv'
    del_list = np.squeeze(pd.read_csv(del_path).values)
    label_df = pd.read_csv(r'X:\FAEFormatData\ECE\ece.csv', index_col='case')
    label_list = []
    case_list = []
    for case in label_df.index:
        if case in del_list:
            case_list.append(case)
            label_list.append(label_df.loc[case]['label'])
    df = pd.DataFrame({'case': case_list, 'label': label_list})
    df.to_csv(r'X:\FAEFormatData\ECE\ZYH0521\DropCase&Label.csv', index=False)


if __name__ == '__main__':
    # th_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # # th_list = [0.7]
    # for th in th_list:
    #     feature_csv_path = r'X:\FAEFormatData\ECE\ZYH0521\atten{}\features.csv'.format(th)
    #     label_csv_path = r'X:\FAEFormatData\ECE\ece.csv'
    #     save_csv_path = r'X:\FAEFormatData\ECE\ZYH0521\atten{}\features_label.csv'.format(th)
    #     AddLabel(feature_csv_path, label_csv_path, save_csv_path)
    #
    #     feature_csv_path = r'X:\FAEFormatData\ECE\ZYH0521\atten{}'.format(th)
    #     del_path = r'X:\FAEFormatData\ECE\ZYH0521\atten0.7\features_error.csv'
    #     del_list = np.squeeze(pd.read_csv(del_path).values)
    #     DelectCasefromCSV(del_list, feature_csv_path)

    feature_csv_path = r'X:\FAEFormatData\ECE\ZYH\PCaWithoutGLCM'
    del_path = r'X:\FAEFormatData\ECE\ZYH0521\drop_data.csv'
    del_list = np.squeeze(pd.read_csv(del_path).values)
    feature_matrix_df = pd.read_csv(os.path.join(feature_csv_path, 'test_numeric_feature.csv'), index_col='CaseName')
    for case in feature_matrix_df.index:
        if case in del_list:
            feature_matrix_df.drop(case, axis=0, inplace=True)
    feature_matrix_df.to_csv(os.path.join(feature_csv_path, 'test_numeric_feature_right.csv'))

    # testDelectFeature()
    # ConcatCSV()
    # SaveNoLabelCase()
    # CheckDelectCase()