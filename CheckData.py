import os
import shutil
import numpy as np
import SimpleITK as sitk
import pandas as pd
from MeDIT.SaveAndLoad import LoadImage

raw_data_folder = r'Y:\Imaging'

def CheckSpacing():
    for path, dirs, files in sorted(os.walk(raw_data_folder, topdown=True)):
        if len(dirs) > 0: continue
        if not os.path.exists(os.path.join(path, 'prostate.nii.gz')):
            print(path, '!')
        try:
            t2, _, _ = LoadImage(os.path.join(path, 't2.nii'), dtype=np.float32, is_show_info=False)
            adc, _, _ = LoadImage(os.path.join(path, 'adc_Reg.nii'), dtype=np.float32, is_show_info=False)
            dwi, _, _ = LoadImage(os.path.join(path, 'dwi_Reg.nii'), dtype=np.float32, is_show_info=False)
            mask, _, _ = LoadImage(os.path.join(path, 'mask_resample.nii.gz'), dtype=np.float32, is_show_info=False)
            prostate, _, _ = LoadImage(os.path.join(path, 'prostate.nii.gz'), dtype=np.float32, is_show_info=False)
            if t2.GetSpacing() == adc.GetSpacing() == dwi.GetSpacing() == mask.GetSpacing() == prostate.GetSpacing():
                continue
            else:
                print(path.split('\\')[-1], 'need align.')
                print(t2.GetSpacing())
                print(dwi.GetSpacing())
                print(adc.GetSpacing())
                print(prostate.GetSpacing())
                print(mask.GetSpacing())


        except Exception as e:
            print(path.split('\\')[-1], '\n', e)
# CheckSpacing()


def NormalizeT2():
    from MeDIT.Normalize import NormalizeZ
    for path, dirs, files in sorted(os.walk(raw_data_folder, topdown=True)):
        if len(dirs) > 0: continue
        if not os.path.exists(os.path.join(path, 'prostate.nii.gz')):
            print(path, '!')

        t2 = sitk.ReadImage(os.path.join(path, 't2.nii'))
        t2_array = sitk.GetArrayFromImage(t2)
        t2_array = NormalizeZ(t2_array)
        t2_norm = sitk.GetImageFromArray(t2_array)
        t2_norm.CopyInformation(t2)
        sitk.WriteImage(t2_norm, os.path.join(path, 't2_norm.nii'))
# NormalizeT2()


def CopyFile():
    for path, dirs, files in sorted(os.walk(raw_data_folder, topdown=True)):
        if len(dirs) > 0: continue
        if 'Y:\Imaging\AllData' in path: continue
        folder, name = path.split('\\')[-2], path.split('\\')[-1]
        case_folder = os.path.join(r'Y:\Imaging\AllData', '{}_{}'.format(folder, name))
        if os.path.exists(case_folder): continue
        else: print(path)
        # os.mkdir(case_folder)
        # try:
        #     shutil.copytree(path, case_folder)
        #     print('successful move {} {}'.format(folder, name))
        # except Exception as e:
        #     print('cannot move {} {} for {}'.format(folder, name, e))

# CopyFile()



def CheckNoLabel():
    from MeDIT.Normalize import Normalize01
    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.SaveAndLoad import LoadImage
    raw_folder = r'Y:\Imaging\AllData'
    label_df = pd.read_csv(r'Y:\Imaging\ALL_formal.csv')
    for case in os.listdir(raw_folder):
        case_folder = os.path.join(raw_folder, case)
        if os.path.exists(os.path.join(case_folder, 'AttentionMap.nii')): continue
        Institute, ID = case.split('_')[0], int(case.split('_')[1])
        case_series = label_df.loc[(label_df['Institute'] == Institute) & (label_df['ID'] == ID)]
        try:
            label = case_series.loc[:, 'ECE'].values[0]
        except Exception as e:
            print(case, 'not in label csv')
            continue
        print(case, label, end=' ')
        t2, t2_array, _ = LoadImage(os.path.join(case_folder, 't2_norm.nii'))
        adc, adc_array, _ = LoadImage(os.path.join(case_folder, 'adc_Reg.nii'))
        pro, pro_array, _ = LoadImage(os.path.join(case_folder, 'prostate.nii.gz'))
        pca, pca_array, _ = LoadImage(os.path.join(case_folder, 'mask_resample.nii.gz'))
        if (pca_array == np.zeros_like(pca_array)).all():
            print('pca roi = 0')
        else:
            print()
            # Imshow3DArray(np.concatenate([Normalize01(t2_array), Normalize01(adc_array)], axis=1),
            #               roi=[np.concatenate([Normalize01(pca_array), Normalize01(pca_array)], axis=1),
            #                    np.concatenate([Normalize01(pro_array), Normalize01(pro_array)], axis=1)])
# CheckNoLabel()


raw_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData\WANG HUAI ZHONG'
t2_image, t2_array, _ = LoadImage(os.path.join(raw_folder, 'adc_Reg.nii.gz'))
pro_image, pro_array, _ = LoadImage(os.path.join(raw_folder, 'ProstateROI_TrumpetNet.nii.gz'))
pca_image, pca_array, _ = LoadImage(os.path.join(raw_folder, 'roi.nii'))
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01
Imshow3DArray(Normalize01(t2_array), roi=[Normalize01(pro_array), Normalize01(pca_array)])

