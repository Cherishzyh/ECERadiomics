import os
import filecmp
import numpy as np
import pandas as pd
from filecmp import dircmp


from pandas.testing import assert_frame_equal


def CheckFile():
    zj_result = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\zj_result0524'
    zyh_result = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result0524'
    cp = dircmp(zj_result, zyh_result)
    cp.report_full_closure()
# CheckFile()


def CheckCSV():
    zj_csv = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\zj_result0524\SMOTE_features.csv'
    zyh_csv = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result0524\SMOTE_features.csv'

    zj_df = pd.read_csv(zj_csv, index_col=0)
    zyh_df = pd.read_csv(zyh_csv, index_col=0)
    print(zyh_df.equals(zj_df))
    assert_frame_equal(zj_df, zyh_df)

    for case in zj_df.index:
        if zj_df.loc[case]['t2_original_shape_Elongation'] == zyh_df.loc[case]['t2_original_shape_Elongation']:
            continue
        else:
            print('{} in zj result is {}, but is {} in zyh result'.
                  format(case, zj_df.loc[case]['t2_original_shape_Elongation'],
                         zyh_df.loc[case]['t2_original_shape_Elongation']))
# CheckCSV()


def CompareFolder():
    '''
    compare two result folder,
    :return:
    '''
    zj_result = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\zj_result0524'
    zyh_result = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result0524'

    zj_csv = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\zj_result0524\SMOTE_features.csv'
    zyh_csv = r'X:\FAEFormatData\ECE\ZYH0521\sub_region_0.2\Result0524\SMOTE_features.csv'

    from csv_diff import load_csv, compare
    diff = compare(
        load_csv(open(zj_csv)),
        load_csv(open(zyh_csv))
    )
    # for root, dirs, files in os.walk(zj_result, topdown=True):
    #     print(root)
    #     print(dirs)
    #     print(files)

        # for name in files:
        #     print(os.path.join(root, name))
        # for name in dirs:
        #     print(os.path.join(root, name))
    print(diff)
CompareFolder()