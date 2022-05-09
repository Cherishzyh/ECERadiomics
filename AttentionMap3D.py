import numpy as np
import os
import shutil
from copy import deepcopy
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter, median_filter, convolve
from scipy.ndimage import binary_dilation

from MeDIT.SaveAndLoad import LoadImage, SaveNiiImage
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01


def InterSliceFilter(attention, diff_value, kernel_size=(3, 1, 1)):
    raw_attention = deepcopy(attention)
    # w = np.array([[[0.25]], [[0.5]], [[0.25]]])

    while True:
        new_attention = maximum_filter(raw_attention, size=kernel_size)
        new_attention[new_attention > raw_attention] -= diff_value
        new_attention[new_attention < 0] = 0

        if not (new_attention > raw_attention).any():
            break

        raw_attention = new_attention

    # new_attention = convolve(new_attention, weights=w)
    # new_attention = convolve(new_attention, size=kernel_size)
    new_attention = median_filter(new_attention, size=kernel_size)
    new_attention = median_filter(new_attention, size=kernel_size)

    return new_attention


def IntraSliceFilter(attention, diff_value, kernel_size=(1, 3, 3)):
    raw_attention = deepcopy(attention)

    while True:
        new_attention = maximum_filter(raw_attention, size=kernel_size)
        new_attention[new_attention > raw_attention] -= diff_value
        new_attention[new_attention < 0] = 0

        if not (new_attention > raw_attention).any():
            break

        raw_attention = new_attention

    new_attention = median_filter(new_attention, size=kernel_size)
    new_attention = median_filter(new_attention, size=kernel_size)

    return new_attention


def GetResult(roi, roi_image, base_rate=0.05):

    resolution = roi_image.GetSpacing()
    slice_rate = resolution[2] / resolution[0] * base_rate

    result = InterSliceFilter(roi, slice_rate)
    result = IntraSliceFilter(result, base_rate)  #result.shape=(23, 399, 399)

    return result


def RoiDilate(roi, roi_image):
    # result.shape=(23, 399, 399)
    result = deepcopy(roi)

    resolution = roi_image.GetSpacing()
    inter = round(5. / resolution[2])     # kernel_size=(3, 1, 1)
    intra = round(5. / resolution[0])     # kernel_size=(1, 3, 3)

    result = binary_dilation(result, structure=np.ones((inter * 2 + 1, 1, 1)))
    result = binary_dilation(result, structure=np.ones((1, intra * 2 + 1, intra * 2 + 1)))

    return result


def GetRoiEdge(attention):
    raw_attention = deepcopy(attention)
    raw_attention[raw_attention >= 1] = 1
    return maximum_filter(raw_attention, size=(1, 11, 11)) - raw_attention


def GetRegion(pro, tumor, oppo_pro, pro_image, th=0.5, is_show=False):
    # GetRegion(pro, pca, pro_edge, new_pro, pro_img)
    new_attention = np.zeros_like(pro)

    # 癌灶长出腺体的部分, 直接置1
    roi1_out = tumor - pro
    new_attention[roi1_out > 0] = 1

    # 癌灶和腺体相交的部分（相加为2）,计算点到腺体边界的距离
    total = tumor + pro
    total[total < 2] = 0
    total[total == 2] = 1
    
    inner_dis = np.multiply(total, GetResult(oppo_pro, pro_image))
    inner_dis[inner_dis > th] = 1
    inner_dis[inner_dis <= th] = 0

    new_attention = new_attention + inner_dis
    region = GetResult(new_attention, pro_image)

    if is_show:
        plt.imshow(region[10, :, :], cmap='gray')
        plt.contour(pro[10, :, :], colors='r')
        plt.contour(tumor[10, :, :], colors='b')
        plt.show()
        plt.close()
    # Imshow3DArray(
    #     np.concatenate([Normalize01(atten).transpose(1, 2, 0), Normalize01(atten_binary).transpose(1, 2, 0)], axis=1))

    return region


def GetRegionThreePart(pro, tumor, oppo_pro, oppo_pca, pro_image, th=0.5, is_show=False):
    new_attention = np.zeros_like(pro)
    dis_pro = GetResult(oppo_pro, pro_image)
    dis_pro[dis_pro == 1] = 0
    dis_pca = GetResult(oppo_pca, pro_image)
    dis_pca[dis_pca == 1] = 0

    # 癌灶长出腺体的部分, 直接置1
    roi_out = tumor - pro
    new_attention[roi_out > 0] = 1

    # 癌灶和腺体相交的部分（相加为2）,计算点到腺体边界的距离
    total = tumor + pro
    total[total < 2] = 0
    total[total == 2] = 1

    inner_dis = np.multiply(total, dis_pro)
    # inner_dis[inner_dis > th] = 1
    # inner_dis[inner_dis <= th] = 0

    # 腺体内癌灶外（腺体-癌灶=1）,点到腺体边界和癌灶边界距离最大的值
    outer = pro - tumor
    outer[outer < 0] = 0
    dis = np.maximum(dis_pro, dis_pca)
    outer_dis = np.multiply(outer, dis)

    region = new_attention + inner_dis + outer_dis
    region_1 = GetResult(region, pro_image)

    plt.figure(figsize=(6, 12))
    plt.subplot(321)
    plt.imshow(pro[11], cmap='gray')
    plt.contour(tumor[11], colors='r')
    plt.contour(outer[11], colors='y')
    plt.subplot(322)
    plt.imshow(new_attention[11], cmap='gray')
    plt.subplot(323)
    plt.imshow(inner_dis[11], cmap='jet')
    plt.subplot(324)
    plt.imshow(outer_dis[11], cmap='jet')
    plt.subplot(325)
    plt.imshow(region[11], cmap='jet')
    plt.subplot(326)
    plt.imshow(region_1[11], cmap='jet')
    plt.show()
    plt.close()

    return region


def SetInformation(box, roi_img):
    box = sitk.GetImageFromArray(box)
    box.SetDirection(roi_img.GetDirection())
    box.SetSpacing(roi_img.GetSpacing())
    box.SetOrigin(roi_img.GetOrigin())

    return box


def BinaryByTh(region, th):
    region_binary = deepcopy(region)
    region_binary[region_binary > th] = 1
    region_binary[region_binary <= th] = 0
    return region_binary


def DelData():
    source_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    # del_str = ['3D_AttentionMap_0520.nii',
    #            '3D_AttentionMap_binary_0520_0.0.nii',
    #            '3D_AttentionMap_binary_0520_0.1.nii',
    #            '3D_AttentionMap_binary_0520_0.2.nii',
    #            '3D_AttentionMap_binary_0520_0.3.nii',
    #            '3D_AttentionMap_binary_0520_0.4.nii',
    #            '3D_AttentionMap_binary_0520_0.5.nii',
    #            '3D_AttentionMap_binary_0520_0.6.nii',
    #            '3D_AttentionMap_binary_0520_0.7.nii',
    #            '3D_AttentionMap_binary_0520_0.8.nii',
    #            '3D_AttentionMap_binary_0520_0.9.nii'
    #            ]
    for case in os.listdir(source_folder):
        file_list = os.listdir(os.path.join(source_folder, case))
        del_str = [file for file in file_list if '0519' in file]
        print('****************del {}****************'.format(case))
        for str in del_str:
            source_path = os.path.join(source_folder, '{}/{}'.format(case, str))
            if os.path.exists(source_path):
                os.remove(source_path)


def test(case_folder):
    th_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # th_list = [0.2]
    index = 0
    for path, dirs, files in sorted(os.walk(case_folder, topdown=True)):
        if len(dirs) > 0: continue
        if not os.path.exists(os.path.join(path, 'prostate.nii.gz')):
            print(path, '!')
        case_path = path
        index += 1
    # for index, case in enumerate(os.listdir(case_folder)):
    #     case_path = os.path.join(case_folder, case)
        try:
            pro_path = os.path.join(case_path, 'prostate.nii.gz')
            pro_img = sitk.ReadImage(pro_path)
            pro = sitk.GetArrayFromImage(pro_img)
            pro = np.asarray(pro, dtype=np.float64)

            pca_path = os.path.join(case_path, 'mask_resample.nii.gz')
            pca_img = sitk.ReadImage(pca_path)
            pca = sitk.GetArrayFromImage(pca_img)
            pca = np.asarray(pca, dtype=np.float64)

            oppo_pro = np.zeros_like(pro)
            oppo_pro[pro == 0] = 1
            oppo_pro[pro == 1] = 0
            atten = GetRegion(pro, pca, oppo_pro, pro_img, th=0.5, is_show=False)

            if np.sum(atten) == 0:
                print('folder {}_{} has no label'.format(
                    path.split('\\')[-2], path.split('\\')[-1]))
            else:
                print('************ {} / {} | Storage {} ************'.format(
                    index, len(os.listdir(case_folder)), path.split('\\')[-1]))
                atten_img = SetInformation(atten, pro_img)
                sitk.WriteImage(atten_img, os.path.join(case_path, 'AttentionMap.nii'))

                for th in th_list:
                    atten_binary = BinaryByTh(atten, th)
                    atten_img_binary = SetInformation(atten_binary, pro_img)
                    sitk.WriteImage(atten_img_binary, os.path.join(case_path, 'AttentionMap_binary_{:.1f}.nii'.format(th)))
        except Exception as e:
            print(path.split('\\')[-1])
            print(e)


def test4ThreePart(case_list):
    case_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'

    th_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # case_list = os.listdir(case_folder)
    for index, case in enumerate(case_list):

        case_path = os.path.join(case_folder, case)

        pro_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
        pro_img = sitk.ReadImage(pro_path)
        pro = sitk.GetArrayFromImage(pro_img)

        pca_path = os.path.join(case_path, 'roi.nii')
        pca_img = sitk.ReadImage(pca_path)
        pca = sitk.GetArrayFromImage(pca_img)

        atten_path = os.path.join(case_path, '3D_AttentionMap_0512.nii')
        atten_img = sitk.ReadImage(atten_path)
        atten_before = sitk.GetArrayFromImage(atten_img)

        oppo_pro = np.zeros_like(pro)
        oppo_pro[pro == 0] = 1
        oppo_pro[pro == 1] = 0

        oppo_pca = np.zeros_like(np.asarray(pca, dtype=np.float64))
        oppo_pca[pca == 0] = 1
        oppo_pca[pca == 1] = 0

        atten = GetRegionThreePart(pro, pca, oppo_pro, oppo_pca, pro_img, th=0, is_show=False)

        # try:
        #     if np.sum(atten) == 0:
        #         print('{} has no label'.format(case))
        #     else:
        #         print('************ {} / {} | Storage {} ************'.format(index+1, len(case_list), case))
        #         atten_img = SetInformation(atten, pro_img)
        #         sitk.WriteImage(atten_img, os.path.join(case_path, '3D_AttentionMap_0512.nii'))
        #
        #         for th in th_list:
        #             atten_binary = BinaryByTh(atten, th)
        #             atten_img_binary = SetInformation(atten_binary, pro_img)
        #             sitk.WriteImage(atten_img_binary, os.path.join(case_path, '3D_AttentionMap_binary_{}_0512.nii'.format(str(th))))
        # except Exception as e:
        #     print(case)
        #     print(e)

        Imshow3DArray(np.concatenate([Normalize01(atten).transpose(1, 2, 0), Normalize01(atten_before).transpose(1, 2, 0)], axis=1),
                      roi=[np.concatenate([Normalize01(pca).transpose(1, 2, 0), Normalize01(pca).transpose(1, 2, 0)], axis=1),
                           np.concatenate([Normalize01(pro).transpose(1, 2, 0), Normalize01(pro).transpose(1, 2, 0)], axis=1)])


def SaveFig():
    case_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    for case in os.listdir(case_folder):
        case_path = os.path.join(case_folder, case)

        atten_path = os.path.join(case_path, '3D_AttentionMap.nii')
        atten_img = sitk.ReadImage(atten_path)
        atten = sitk.GetArrayFromImage(atten_img)

        atten_path = os.path.join(r'X:\StoreFormatData\PCa_ECE_YHX\ResampleDataCopy', '{}/3D_AttentionMap.nii'.format(case))
        atten_img = sitk.ReadImage(atten_path)
        atten_st = sitk.GetArrayFromImage(atten_img)

        plt.imshow(np.concatenate([Normalize01(atten).transpose(1, 2, 0)[:, :, 10],
                                   Normalize01(atten_st).transpose(1, 2, 0)[:, :, 10]], axis=1), cmap='gray')
        plt.savefig(os.path.join(r'X:\StoreFormatData\ProstateCancerECE\Image', '{}.jpg'.format(case)))


def GetNoLabel():
    folder = r'Y:\Imaging'
    # folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    case_list = os.listdir(folder)
    no_label_list = []
    for index, case in enumerate(case_list):

        case_path = os.path.join(folder, case)

        atten_path_bg = os.path.join(case_path, '3D_AttentionMap_binary_0.2_bg.nii')
        atten_img_bg = sitk.ReadImage(atten_path_bg)
        atten_bg = sitk.GetArrayFromImage(atten_img_bg)

        atten_path_pro = os.path.join(case_path, '3D_AttentionMap_binary_0.2_pro.nii')
        atten_img_pro = sitk.ReadImage(atten_path_pro)
        atten_pro = sitk.GetArrayFromImage(atten_img_pro)

        pro_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
        pro_img = sitk.ReadImage(pro_path)
        pro = sitk.GetArrayFromImage(pro_img)

        pca_path = os.path.join(case_path, 'roi.nii')
        pca_img = sitk.ReadImage(pca_path)
        pca = sitk.GetArrayFromImage(pca_img)

        new_array = np.concatenate(
            [Normalize01(atten_bg).transpose(1, 2, 0), Normalize01(atten_pro).transpose(1, 2, 0)], axis=1)

        new_pca = np.concatenate([Normalize01(pca).transpose(1, 2, 0), Normalize01(pca).transpose(1, 2, 0)], axis=1)

        new_pro = np.concatenate([Normalize01(pro).transpose(1, 2, 0), Normalize01(pro).transpose(1, 2, 0)], axis=1)

        Imshow3DArray(new_array, roi=[new_pro, new_pca])

        # if np.sum(atten) == 0:
        #     no_label_list.append(case)
        #     print(case)
    return no_label_list


def Binary(case_list, th):
    folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    for index, case in enumerate(case_list):

        case_path = os.path.join(folder, case)

        atten_path = os.path.join(case_path, '3D_AttentionMap.nii')
        atten_img = sitk.ReadImage(atten_path)
        atten = sitk.GetArrayFromImage(atten_img)

        atten_binary = deepcopy(atten)
        atten_binary[atten_binary > th] = 1
        atten_binary[atten_binary <= th] = 0

        if np.sum(atten_binary) == 0:
            print(case)
            continue
        else:
            atten_img_binary = SetInformation(atten_binary, atten_img)
            print('************ {} / {} | Storage {}, th = {} ************'.format(index + 1, len(case_list), case, th))
            sitk.WriteImage(atten_img_binary, os.path.join(case_path, '3D_AttentionMap_binary_{}.nii'.format(str(th))))


        # for th in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
        #     atten_binary = deepcopy(atten)
        #     atten_binary[atten_binary > float(th)] = 1
        #     atten_binary[atten_binary <= float(th)] = 0
        #
        #     if np.sum(atten_binary) == 0:
        #         continue
        #     else:
        #         atten_img_binary = SetInformation(atten_binary, atten_img)
        #         print('************ {} / {} | Storage {}, th = {} ************'.format(index + 1, len(case_list), case, th))
        #         sitk.WriteImage(atten_img_binary, os.path.join(case_path, '3D_AttentionMap_binary_0.7.nii'))
        #         break


def SplitRoi(folder, is_show=False):
    case_list = os.listdir(folder)
    for index, case in enumerate(case_list):
        case_path = os.path.join(folder, case)

        atten_path = os.path.join(case_path, '3D_AttentionMap_binary_1021_0.2.nii')
        if not os.path.exists(atten_path):
            continue
        atten_img = sitk.ReadImage(atten_path)
        atten = sitk.GetArrayFromImage(atten_img)

        pro_path = os.path.join(case_path, 'prostate_roi_5x5.nii.gz')
        pro_img = sitk.ReadImage(pro_path)
        pro = sitk.GetArrayFromImage(pro_img)

        pca_path = os.path.join(case_path, 'pca_roi_5x5.nii.gz')
        pca_img = sitk.ReadImage(pca_path)
        pca = sitk.GetArrayFromImage(pca_img)

        pca_in_atten = atten * pca
        pro_in_atten = (atten - pca_in_atten) * pro
        bg_in_atten = atten - pro_in_atten - pca_in_atten
        bg_in_atten[bg_in_atten < 0] = 0

        pca_in_atten_img = SetInformation(pca_in_atten, pca_img)
        pro_in_atten_img = SetInformation(pro_in_atten, pca_img)
        bg_in_atten_img = SetInformation(bg_in_atten, pca_img)

        print('************ {} / {} | Storage {} ************'.format(index + 1, len(case_list), case))
        sitk.WriteImage(pca_in_atten_img, os.path.join(case_path, '3D_AttentionMap_binary_1021_0.2_pca.nii'))
        sitk.WriteImage(pro_in_atten_img, os.path.join(case_path, '3D_AttentionMap_binary_1021_0.2_pro.nii'))
        sitk.WriteImage(bg_in_atten_img, os.path.join(case_path, '3D_AttentionMap_binary_1021_0.2_bg.nii'))


        if is_show:
            new_array_1 = np.concatenate([Normalize01(pro_in_atten).transpose(1, 2, 0), Normalize01(pca_in_atten).transpose(1, 2, 0)], axis=1)
            new_array_2 = np.concatenate([Normalize01(bg_in_atten).transpose(1, 2, 0), Normalize01(atten).transpose(1, 2, 0)], axis=1)
            new_array = np.concatenate([Normalize01(new_array_1), Normalize01(new_array_2)], axis=0)

            new_pca_1 = np.concatenate([Normalize01(pca).transpose(1, 2, 0), Normalize01(pca).transpose(1, 2, 0)], axis=1)
            new_pca_2 = np.concatenate([Normalize01(pca).transpose(1, 2, 0), Normalize01(pca).transpose(1, 2, 0)], axis=1)
            new_pca = np.concatenate([Normalize01(new_pca_1), Normalize01(new_pca_2)], axis=0)

            new_pro_1 = np.concatenate([Normalize01(pro).transpose(1, 2, 0), Normalize01(pro).transpose(1, 2, 0)], axis=1)
            new_pro_2 = np.concatenate([Normalize01(pro).transpose(1, 2, 0), Normalize01(pro).transpose(1, 2, 0)], axis=1)
            new_pro = np.concatenate([Normalize01(new_pro_1), Normalize01(new_pro_2)], axis=0)

            Imshow3DArray(new_array, roi=[new_pro, new_pca])


def DiffTh(case_list, is_show=False):
    folder = r'Y:\Imaging'
    for path, dirs, files in sorted(os.walk(folder, topdown=True)):
        if len(dirs) > 0: continue
        if not os.path.exists(os.path.join(path, 'prostate.nii.gz')):
            print(path, '!')
        case_path = path

        atten_path_0 = os.path.join(case_path, '3D_AttentionMap_0512.nii')
        atten_img_0 = sitk.ReadImage(atten_path_0)
        atten_0 = sitk.GetArrayFromImage(atten_img_0)

        atten_path_02 = os.path.join(case_path, '3D_AttentionMap_binary_0.2_0512.nii')
        atten_img_02 = sitk.ReadImage(atten_path_02)
        atten_02 = sitk.GetArrayFromImage(atten_img_02)

        atten_path_04 = os.path.join(case_path, '3D_AttentionMap_binary_0.4_0512.nii')
        atten_img_04 = sitk.ReadImage(atten_path_04)
        atten_04 = sitk.GetArrayFromImage(atten_img_04)

        atten_path_07 = os.path.join(case_path, '3D_AttentionMap_binary_0.7_0512.nii')
        atten_img_07 = sitk.ReadImage(atten_path_07)
        atten_07 = sitk.GetArrayFromImage(atten_img_07)

        pro_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
        pro_img = sitk.ReadImage(pro_path)
        pro = sitk.GetArrayFromImage(pro_img)

        pca_path = os.path.join(case_path, 'roi.nii')
        pca_img = sitk.ReadImage(pca_path)
        pca = sitk.GetArrayFromImage(pca_img)


        if is_show:
            new_array_1 = np.concatenate([Normalize01(atten_0).transpose(1, 2, 0), Normalize01(atten_02).transpose(1, 2, 0)], axis=1)
            new_array_2 = np.concatenate([Normalize01(atten_04).transpose(1, 2, 0), Normalize01(atten_07).transpose(1, 2, 0)], axis=1)
            new_array = np.concatenate([Normalize01(new_array_1), Normalize01(new_array_2)], axis=0)

            new_pca_1 = np.concatenate([Normalize01(pca).transpose(1, 2, 0), Normalize01(pca).transpose(1, 2, 0)], axis=1)
            new_pca_2 = np.concatenate([Normalize01(pca).transpose(1, 2, 0), Normalize01(pca).transpose(1, 2, 0)], axis=1)
            new_pca = np.concatenate([Normalize01(new_pca_1), Normalize01(new_pca_2)], axis=0)

            new_pro_1 = np.concatenate([Normalize01(pro).transpose(1, 2, 0), Normalize01(pro).transpose(1, 2, 0)], axis=1)
            new_pro_2 = np.concatenate([Normalize01(pro).transpose(1, 2, 0), Normalize01(pro).transpose(1, 2, 0)], axis=1)
            new_pro = np.concatenate([Normalize01(new_pro_1), Normalize01(new_pro_2)], axis=0)

            Imshow3DArray(new_array, roi=[new_pro, new_pca])


if __name__ == '__main__':
    from MeDIT.ArrayProcess import ExtractBlock
    # folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    # folder = r'X:\PrcoessedData\ProstateCancerECE_SUH'
    folder = r'Y:\Imaging'
    # no_label = ['WANG QING GUI', 'LIU AI HUA', 'JIANG SONG YUAN']
    # #
    # for case in sorted(os.listdir(folder)):
    #     case_path = os.path.join(folder, case)
    #     atten_path_dilate = os.path.join(case_path, '3D_AttentionMap_binary_0.2.nii')
    #     atten_img_binary = sitk.ReadImage(atten_path_dilate)
    #     atten_binary = sitk.GetArrayFromImage(atten_img_binary)
    #
    #     atten_path = os.path.join(case_path, '3D_AttentionMap_binary_0.2_0512.nii')
    #     atten_img = sitk.ReadImage(atten_path)
    #     atten = sitk.GetArrayFromImage(atten_img)
    #
    #     pro_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
    #     pro_img = sitk.ReadImage(pro_path)
    #     pro = sitk.GetArrayFromImage(pro_img)
    #
    #     pca_path = os.path.join(case_path, 'roi.nii')
    #     pca_img = sitk.ReadImage(pca_path)
    #     pca = sitk.GetArrayFromImage(pca_img)
    #
    #     Imshow3DArray(np.concatenate([Normalize01(atten_binary).transpose(1, 2, 0), Normalize01(atten).transpose(1, 2, 0)], axis=1),
    #                   roi=[np.concatenate([Normalize01(pro).transpose(1, 2, 0), Normalize01(pro).transpose(1, 2, 0)], axis=1),
    #                        np.concatenate([Normalize01(pca).transpose(1, 2, 0), Normalize01(pca).transpose(1, 2, 0)],
    #                                       axis=1)])
    #     t2_path = os.path.join(case_path, 't2.nii')
    #     t2_img = sitk.ReadImage(t2_path)
    #     t2 = sitk.GetArrayFromImage(t2_img)
    #
    #     dilate_atten = RoiDilate(atten_binary, atten_img_binary)
    #
    #     Imshow3DArray(np.concatenate([Normalize01(atten_binary).transpose(1, 2, 0), Normalize01(atten).transpose(1, 2, 0)], axis=1),
    #                   roi=[np.concatenate([Normalize01(pro).transpose(1, 2, 0), Normalize01(pro).transpose(1, 2, 0)], axis=1),
    #                        np.concatenate([Normalize01(pca).transpose(1, 2, 0), Normalize01(pca).transpose(1, 2, 0)],
    #                                       axis=1)])
    #     Imshow3DArray(Normalize01(t2).transpose(1, 2, 0),
    #                   roi=[Normalize01(pro).transpose(1, 2, 0), Normalize01(pca).transpose(1, 2, 0)])


    # case_list = GetNoLabel()
    # print(case_list)

    # DelData()

    # test(os.listdir(folder))
    # test(['CHEN JIE'])
    # test4ThreePart(os.listdir(folder))

    # SplitRoi(is_show=False)
    # DiffTh(case_list=['BAO ZHENG LI'], is_show=True)
    # list = GetNoLabel()
    # print(list)
    # no_label = ['WANG QING GUI', 'LYZ^liu yin zhong', 'LIU AI HUA', 'JIANG SONG YUAN']

    # SplitRoi(folder, is_show=False)
    # test(folder)


