import os

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


from MeDIT.SaveAndLoad import LoadImage
from MeDIT.Visualization import ShowColorByRoi, Imshow3DArray, FusionImage
from MeDIT.Normalize import Normalize01
from MeDIT.ArrayProcess import ExtractBlock, ExtractPatch


def Show():
    data_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    save_folder = r'X:\StoreFormatData\ProstateCancerECE\Image'
    for case in os.listdir(data_folder):
        case = 'BIAN JIN YOU'
        case_path = os.path.join(data_folder, case)

        atten_path = os.path.join(case_path, '3D_AttentionMap_0521.nii')
        t2_path = os.path.join(case_path, 't2.nii')
        adc_path = os.path.join(case_path, 'adc_Reg.nii')
        pro_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
        pca_path = os.path.join(case_path, 'roi.nii')

        _, t2, _ = LoadImage(t2_path)
        _, adc, _ = LoadImage(adc_path)
        _, pro, _ = LoadImage(pro_path)
        _, pca, _ = LoadImage(pca_path)
        _, atten, _ = LoadImage(atten_path)
        # print(np.max(atten))
        #
        atten_roi = deepcopy(atten)
        atten_roi[atten_roi > 0] = 1
        # #
        # for slice in [6, 7, 8]:
        #     t2_slice = t2[:, :, slice]
        #     pro_slice = pro[:, :, slice]
        #     pca_slice = pca[:, :, slice]
        #     atten_slice = atten[:, :, slice]
        #     atten_roi_slice = atten_roi[:, :, slice]
        #
        #     fusion_map = ShowColorByRoi(t2_slice, atten_slice, atten_roi_slice, color_map='jet', is_show=False)
        #
        #     plt.subplot(121)
        #     plt.imshow(t2_slice, cmap='gray')
        #     plt.contour(pro_slice, colors='r')
        #     plt.contour(pca_slice, colors='y')
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #     plt.subplot(122)
        #     plt.imshow(fusion_map, vmax=1., vmin=0.)
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #     # plt.savefig(os.path.join(save_folder, '{}_slice5.jpg'.format(case)))
        #     # plt.close()
        #     plt.show()
        # Imshow3DArray(np.concatenate([Normalize01(t2), Normalize01(atten)], axis=1),
        #               roi=[np.concatenate([Normalize01(pro), Normalize01(pro)], axis=1),
        #                    np.concatenate([Normalize01(pca), Normalize01(pca)], axis=1)])
        Imshow3DArray(np.concatenate([Normalize01(t2), Normalize01(atten)], axis=1),
                      roi=[Normalize01(pro), Normalize01(pca)])
        Imshow3DArray(np.concatenate([Normalize01(t2), Normalize01(adc)], axis=1))
        break
# Show()

def Compare():
    case_path = r'X:\StoreFormatData\ProstateCancerECE\ResampleData\BAO ZHENG LI'

    t2_path = os.path.join(case_path, 't2.nii')
    pro_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
    pca_path = os.path.join(case_path, 'roi.nii')
    _, t2, _ = LoadImage(t2_path)
    _, pro, _ = LoadImage(pro_path)
    _, pca, _ = LoadImage(pca_path)

    # original
    atten_path = os.path.join(case_path, '3D_AttentionMap_binary_0.2.nii')
    _, atten_ori, _ = LoadImage(atten_path)

    # 0520
    atten_path = os.path.join(case_path, '3D_AttentionMap_binary_0521_0.2.nii')
    _, atten_0520, _ = LoadImage(atten_path)

    print((atten_ori == atten_0520).all())
# Compare()

#
# def GetCenter(mask):
#     assert (np.ndim(mask) == 2)
#     roi_row = np.sum(mask, axis=1)
#     roi_column = np.sum(mask, axis=0)
#
#     row = np.nonzero(roi_row)[0]
#     column = np.nonzero(roi_column)[0]
#
#     center = [int(np.median(row)), int(np.median(column))]
#     return center


def ShowRegion():
    data_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    for case in os.listdir(data_folder):
        case_path = os.path.join(data_folder, case)

        t2_path = os.path.join(case_path, 't2_Reg.nii')
        binary_atten_path = os.path.join(case_path, '3D_AttentionMap_binary_0521_0.2.nii')
        bg_path = os.path.join(case_path, '3D_AttentionMap_binary_0.2_bg.nii')
        pro_path = os.path.join(case_path, '3D_AttentionMap_binary_0.2_pro.nii')
        pca_path = os.path.join(case_path, '3D_AttentionMap_binary_0.2_pca.nii')

        _, t2, _ = LoadImage(t2_path)
        _, atten, _ = LoadImage(binary_atten_path)
        _, bg, _ = LoadImage(bg_path)
        _, pro, _ = LoadImage(pro_path)
        _, pca, _ = LoadImage(pca_path)

        roi_slice = np.sum(pca, axis=(0, 1))
        slice = int(np.median(np.nonzero(roi_slice)))

        center = GetCenter(pca[..., slice])

        # for slice in [med-1, med, med+1]:
        #     t2_slice, _ = ExtractPatch(t2[..., slice], (150, 150), center)
        #     atten_slice, _ = ExtractPatch(atten[..., slice], (150, 150), center)
        #     pro_slice, _ = ExtractPatch(pro[..., slice], (150, 150), center)
        #     pca_slice, _ = ExtractPatch(pca[..., slice], (150, 150), center)
        #     bg_slice, _ = ExtractPatch(bg[..., slice], (150, 150), center)
        #
        #     # plt.figure(figsize=(12,5))
        #     # plt.subplot(141)
        #     # plt.imshow(atten_slice*t2_slice, cmap='gray')
        #     # plt.subplot(142)
        #     # plt.imshow(pro_slice*t2_slice, cmap='gray')
        #     # plt.subplot(143)
        #     # plt.imshow(pca_slice*t2_slice, cmap='gray')
        #     # plt.subplot(144)
        #     # plt.imshow(bg_slice*t2_slice, cmap='gray')
        #     # plt.show()
        #     # plt.close()
        #
        #     plt.imshow(t2_slice, cmap='gray')
        #     plt.contour(bg_slice, colors='r')
        #     plt.contour(pro_slice, colors='g')
        #     plt.contour(pca_slice, colors='b')
        #     plt.axis('off')
        #     plt.show()
        #     plt.close()

        t2_slice, _ = ExtractPatch(t2[..., slice], (150, 150), center)
        atten_slice, _ = ExtractPatch(atten[..., slice], (150, 150), center)
        pro_slice, _ = ExtractPatch(pro[..., slice], (150, 150), center)
        pca_slice, _ = ExtractPatch(pca[..., slice], (150, 150), center)
        bg_slice, _ = ExtractPatch(bg[..., slice], (150, 150), center)

        plt.title(case)
        plt.imshow(t2_slice, cmap='gray')
        plt.contour(bg_slice, colors='r')
        plt.contour(pro_slice, colors='g')
        plt.contour(pca_slice, colors='b')
        plt.axis('off')
        plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop\Image', '{}.jpg'.format(case)))
        plt.close()

# ShowRegion()


def ShowRegionOneCase():
    data_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    case = 'BIAN JIN YOU'
    case_path = os.path.join(data_folder, case)

    t2_path = os.path.join(case_path, 't2.nii')
    adc_path = os.path.join(case_path, 'adc_Reg.nii')
    PCa_path = os.path.join(case_path, 'roi.nii')
    PRO_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')

    binary_atten_path = os.path.join(case_path, '3D_AttentionMap_binary_0521_0.2.nii')
    # binary_atten_path = os.path.join(case_path, '3D_AttentionMap_0521.nii')
    bg_path = os.path.join(case_path, '3D_AttentionMap_binary_0.2_bg.nii')
    pro_path = os.path.join(case_path, '3D_AttentionMap_binary_0.2_pro.nii')
    pca_path = os.path.join(case_path, '3D_AttentionMap_binary_0.2_pca.nii')

    _, t2, _ = LoadImage(t2_path)
    _, adc, _ = LoadImage(adc_path)
    _, PCa, _ = LoadImage(PCa_path)
    _, PRO, _ = LoadImage(PRO_path)

    _, atten, _ = LoadImage(binary_atten_path)
    _, bg, _ = LoadImage(bg_path)
    _, pro, _ = LoadImage(pro_path)
    _, pca, _ = LoadImage(pca_path)

    roi_slice = np.sum(PCa, axis=(0, 1))
    med = int(np.median(np.nonzero(roi_slice)))
    print(med)

    center = GetCenter(PRO[..., med])

    for slice in [med-1, med, med+1]:

        adc_slice, _ = ExtractPatch(adc[..., slice], (150, 150), center)
        t2_slice, _ = ExtractPatch(t2[..., slice], (150, 150), center)
        PCa_slice, _ = ExtractPatch(PCa[..., slice], (150, 150), center)
        PRO_slice, _ = ExtractPatch(PRO[..., slice], (150, 150), center)

        atten_slice, _ = ExtractPatch(atten[..., slice], (150, 150), center)
        pro_slice, _ = ExtractPatch(pro[..., slice], (150, 150), center)
        pca_slice, _ = ExtractPatch(pca[..., slice], (150, 150), center)
        bg_slice, _ = ExtractPatch(bg[..., slice], (150, 150), center)

        # atten_roi = deepcopy(atten_slice)
        # atten_roi[atten_roi > 0] = 1
        # fusion_map = ShowColorByRoi(t2_slice, atten_slice, atten_roi, color_map='jet', is_show=False)
        # plt.imshow(fusion_map, vmax=1., vmin=0.)
        # plt.axis('off')
        # plt.show()

        # plt.subplot(121)
        plt.imshow(t2_slice, cmap='gray')
        # plt.contour(PCa_slice, colors='r')
        # plt.contour(PRO_slice, colors='y')
        plt.contour(atten_slice, colors='r')
        plt.axis('off')
        # # plt.subplot(122)
        # plt.imshow(adc_slice, cmap='gray')
        # plt.axis('off')
        plt.show()
        plt.close()
        # region = bg_slice + pca_slice*2 + pro_slice*3
        # region = Normalize01(region)
        # binary_region = deepcopy(region)
        # binary_region[binary_region > 0] = 1
        #
        # fusion_map = ShowColorByRoi(t2_slice, region, binary_region, color_map='jet', is_show=False, alpha=0.3)
        # plt.imshow(fusion_map, vmax=1., vmin=0.)
        # plt.axis('off')
        # plt.show()
# ShowRegionOneCase()


def Distribution():
    data_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    case = 'BIAN JIN YOU'
    case_path = os.path.join(data_folder, case)

    t2_path = os.path.join(case_path, 't2.nii')

    _, t2, _ = LoadImage(t2_path)

    pixels = np.sort(t2.flatten())
    min_drop = np.percentile(pixels, 2.5)
    max_drop = np.percentile(pixels, 97.5)
    pixels = pixels.tolist()
    pixels_95 = [value for value in pixels if min_drop < value < max_drop]
    # pixels = np.clip(pixels, a_min=min_drop, a_max=max_drop)


    plt.hist(pixels_95, bins=25, edgecolor='k')

    plt.yticks([])
    plt.show()
    plt.close()
# Distribution()


def GetCenter(roi, slice=None):
    roi = np.squeeze(roi)
    if np.ndim(roi) == 3 and not slice:
        roi = roi[0]
    elif np.ndim(roi) == 3 and slice:
        roi = roi[slice]
    non_zero = np.nonzero(roi)
    center_x = int(np.median(np.unique(non_zero[1])))
    center_y = int(np.median(np.unique(non_zero[0])))
    return (center_y, center_x)


def ShowAtten():
    '''
    Save the attention map of case of the max roi slice

    '''

    folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    shape = (160, 160)
    # ['BSL^bai song lai ^^6698-8', 'HTS^hua tai shan', 'JCB^jiang chang bao', 'GU SI KANG']
    for case in sorted(['DYG^ding ying gen', 'HTS^hua tai shan', 'JJS^jin ju sheng']):
        case_folder = os.path.join(folder, case)

        t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 't2.nii')))
        pca = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'roi.nii')))
        pro = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'ProstateROI_TrumpetNet.nii.gz')))
        atten = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, '3D_AttentionMap_0521.nii')))
        binary = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, '3D_AttentionMap_binary_0.2.nii')))
        binary_bg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, '3D_AttentionMap_binary_0.2_bg')))
        binary_pro = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, '3D_AttentionMap_binary_0.2_pro')))
        binary_pca = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, '3D_AttentionMap_binary_0.2_pca')))

        index = np.argmax(np.sum(pca, axis=(1, 2)))
        center = GetCenter(pca, slice=index)

        t2_crop, _ = ExtractPatch(t2[index], patch_size=shape, center_point=center)
        pro_crop, _ = ExtractPatch(pro[index], patch_size=shape, center_point=center)
        pca_crop, _ = ExtractPatch(pca[index], patch_size=shape, center_point=center)
        atten_crop, _ = ExtractPatch(atten[index], patch_size=shape, center_point=center)
        binary_crop, _ = ExtractPatch(binary[index], patch_size=shape, center_point=center)
        binary_bg, _ = ExtractPatch(binary_bg[index], patch_size=shape, center_point=center)
        binary_pro, _ = ExtractPatch(binary_pro[index], patch_size=shape, center_point=center)
        binary_pca, _ = ExtractPatch(binary_pca[index], patch_size=shape, center_point=center)

        # t2_binary = ShowColorByRoi(t2_crop, atten_crop, roi=atten_crop, alpha=0.3, is_show=False)
        plt.figure(figsize=(6, 6), dpi=300)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.imshow(t2_crop, cmap='gray')
        plt.contour(pro_crop, colors='r')
        plt.contour(pca_crop, colors='y')
        # plt.show()
        # plt.savefig(os.path.join(r'X:\StoreFormatData\ProstateCancerECE\AAAA', '{}_t2_binary.jpg'.format(case)), bbox_inches='tight', pad_inches=0.05)
        plt.close()

        all_region = binary_bg + 0.3*binary_pca + 0.6*binary_pro
        region = ShowColorByRoi(t2_crop, all_region, roi=binary_crop, alpha=0.5, is_show=False)
        plt.figure(figsize=(6, 6), dpi=300)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')
        plt.imshow(region)
        # plt.show()
        # plt.savefig(os.path.join(r'X:\StoreFormatData\ProstateCancerECE\AAAA', '{}_region.jpg'.format(case)), bbox_inches='tight', pad_inches=0.05)
        plt.close()
        # print(case)

        fusion = ShowColorByRoi(t2_crop, atten_crop, roi=binary_crop, color_map='jet', is_show=False)
        plt.figure(figsize=(6, 6), dpi=300)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.imshow(fusion)
        # plt.show()
        plt.savefig(os.path.join(r'X:\StoreFormatData\ProstateCancerECE\AAAA', '{}_atten.jpg'.format(case)),
                    bbox_inches='tight', pad_inches=0.05)

        plt.close()
ShowAtten()

