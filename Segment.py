import os
import numpy as np
import matplotlib.pyplot as plt
from CNNModel.SuccessfulModel.ProstateSegment import ProstateSegmentationTrumpetNet
from MeDIT.SaveAndLoad import LoadImage
from MeDIT.Visualization import FlattenImages
from MeDIT.Normalize import Normalize01


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

model_folder_path = r'/home/zhangyihong/Documents/SUHECE/ProstateSegmentTrumpetNet'
prostate_segmentor = ProstateSegmentationTrumpetNet()
prostate_segmentor.LoadConfigAndModel(model_folder_path)

raw_data_folder = r'/home/zhangyihong/Documents/SUHECE/Imaging'

for path, dirs, files in sorted(os.walk(raw_data_folder, topdown=True)):
    if len(dirs) > 0: continue
    if os.path.exists(os.path.join(path, 'prostate.nii.gz')): continue
    t2_path = os.path.join(path, 't2.nii')

    image, show_data, _ = LoadImage(t2_path, dtype=np.float32, is_show_info=False)

    predict_data, mask, mask_image = prostate_segmentor.Run(image, model_folder_path,
                                                            store_folder=os.path.join(path, 'prostate.nii.gz'))

    print(path)
    merge_image = FlattenImages([show_data[..., idx] for idx in range(show_data.shape[-1])])
    merge_roi = FlattenImages([mask[..., idx] for idx in range(mask.shape[-1])])

    plt.figure(figsize=(16, 16), dpi=500)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(Normalize01(merge_image), cmap='gray', vmin=0, vmax=1)
    plt.contour(merge_roi, linewidths=1)
    # plt.savefig(os.path.join(r'/home/zhangyihong/Documents/SUHECE/ProstateImage',
    #                          '{}_{}.jpg'.format(path.split('\\')[-2], path.split('\\')[-1])))
    plt.savefig(os.path.join(r'/home/zhangyihong/Documents/SUHECE/ProstateImage',
                             '{}_{}.jpg'.format(path.split('/')[-2], path.split('/')[-1])))
    plt.close()
    del image, show_data, predict_data, mask, mask_image, merge_image, merge_roi

