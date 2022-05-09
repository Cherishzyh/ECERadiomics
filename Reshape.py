import os
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import transform

from MeDIT.SaveAndLoad import LoadImage
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01


def CheckECEData():
    data_path = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    error_csv = r'C:\Users\ZhangYihong\Desktop\ECE_Radiomics\feature\features_normal_error.csv'
    error_case = pd.read_csv(error_csv, header=None).values.tolist()
    for case in error_case:
        print(case[0])
        case_path = os.path.join(data_path, case[0])

        t2_path = os.path.join(case_path, 't2.nii')
        adc_path = os.path.join(case_path, 'adc_Reg.nii')
        dwi_path = os.path.join(case_path, 'dwi_Reg.nii')
        roi_path = os.path.join(case_path, 'roi.nii')

        _ = LoadImage(t2_path, is_show_info=True)
        _ = LoadImage(adc_path, is_show_info=True)
        _ = LoadImage(dwi_path, is_show_info=True)
        _ = LoadImage(roi_path, is_show_info=True)
        print()
# CheckECEData()


def DataResize():
    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01
    data_path = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    error_csv = r'C:\Users\ZhangYihong\Desktop\ECE_Radiomics\feature\features_normal_error.csv'
    error_case = pd.read_csv(error_csv, header=None).values.tolist()
    for case in error_case:
        print(case[0])
        case_path = os.path.join(data_path, case[0])

        t2_path = os.path.join(case_path, 't2.nii')
        roi_path = os.path.join(case_path, 'roi.nii')

        t2_image, t2_arr, _ = LoadImage(t2_path, is_show_info=False)
        roi_image, roi_arr, _ = LoadImage(roi_path, is_show_info=False)

        shape = t2_image.GetSize()
        roi_arr_resize = transform.resize(roi_arr, shape, order=3)
        for slice in range(roi_arr.shape[-1]):
            plt.subplot(121)
            plt.imshow(roi_arr[..., slice], cmap='gray')
            plt.subplot(122)
            plt.imshow(roi_arr_resize[..., slice], cmap='gray')
            plt.show()


def DataReshape():
    '''
    e.g.
        before: t2.shape = (360, 360, 23), roi.shape = (360, 359, 23)
        np.concat([(360, 359, 23), (360, 1, 23)], axis=1)
    :return: roi.shape = (360, 360, 23)
    '''
    data_path = r'X:\StoreFormatData\ProstateCancerECE\Radiomics'
    error_csv = r'C:\Users\ZhangYihong\Desktop\ECE_Radiomics\radiomics_feature\features_normal_error.csv'
    save_folder = r'X:\StoreFormatData\ProstateCancerECE\FailedData\Radiomics'
    error_case = pd.read_csv(error_csv, header=None).values.tolist()
    for case in error_case:
        print(case[0])

        save_path = os.path.join(save_folder, case[0])
        case_path = os.path.join(data_path, case[0])
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        t2_path = os.path.join(case_path, 't2.nii')
        roi_path = os.path.join(case_path, 'roi.nii')

        # t2_image, t2_arr, _ = LoadImage(t2_path, is_show_info=False)
        # roi_image, roi_arr, _ = LoadImage(roi_path, is_show_info=False)
        t2_image = sitk.ReadImage(t2_path)
        roi_image = sitk.ReadImage(roi_path)
        t2_arr = sitk.GetArrayFromImage(t2_image)
        roi_arr = sitk.GetArrayFromImage(roi_image)

        new_roi_arr = roi_arr
        for axis in range(len(t2_arr.shape)):
            if t2_arr.shape[axis] == new_roi_arr.shape[axis]:
                pass
            elif t2_arr.shape[axis] > new_roi_arr.shape[axis]:
                shape = [new_roi_arr.shape[0], new_roi_arr.shape[1], new_roi_arr.shape[2]]
                shape[axis] = int(t2_arr.shape[axis] - new_roi_arr.shape[axis])
                roi_add = np.zeros(shape=shape, dtype=np.int)
                new_roi_arr = np.concatenate([new_roi_arr, roi_add], axis=axis)
            else:

                del_index = np.arange(new_roi_arr.shape[axis] - t2_arr.shape[axis]).tolist()
                del_index = [index + t2_arr.shape[axis] for index in del_index]
                new_roi_arr = np.delete(new_roi_arr, del_index, axis)

        # print(new_roi_arr.shape)
        print(new_roi_arr.shape)
        new_roi = sitk.GetImageFromArray(new_roi_arr)
        new_roi.SetDirection(t2_image.GetDirection())
        new_roi.SetOrigin(t2_image.GetOrigin())
        new_roi.SetSpacing(t2_image.GetSpacing())
        sitk.WriteImage(new_roi, os.path.join(save_path, 'resize_roi.nii'))
# DataReshape()


def CopyData():
    scr_folder = r'X:\StoreFormatData\ProstateCancerECE\FailedData\Radiomics'
    des_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    for case in os.listdir(scr_folder):
        # 合成roi的case文件夹
        src_case_path = os.path.join(scr_folder, case)

        # 合成的roi的路径
        src_roi_path = os.path.join(src_case_path, 'resize_roi.nii')

        # 合成roi的保存文件夹
        des_case_path = os.path.join(des_folder, case)
        # 原roi的路径
        des_roi_path = os.path.join(des_case_path, 'roi.nii')

        if os.path.exists(des_case_path):
            # 如果roi.nii 存在，为了防止生成的错误而正确的没有，将原roi copy到FailedData\Radiomics里
            # shutil.copy(src, dst)
            shutil.copy(des_roi_path, os.path.join(src_case_path, 'roi.nii'))

        # 将合成好的roi copy到原路径
        shutil.copy(src_roi_path, des_roi_path)
# CopyData()
# scr_folder = r'X:\StoreFormatData\ProstateCancerECE\FailedData\Radiomics'
# print(os.listdir(scr_folder))