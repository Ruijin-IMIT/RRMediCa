


import copy
import csv
import os.path
import pickle
import shutil
from typing import Any, Callable, Optional, Tuple

import torch
import cv2
import random

import torch.utils.data as data
import torchvision.transforms as transforms

import os
import json
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool

THREADS = 30


def create_annotation_from_mask_RJPD(database, pd_map, qsm_labels, nm_labels):
    '''
    'annotations.json'
    database: containing the patient dirs. Each patient dir contains all the nifti files, including images and masks.
    labels: the mask labels that are cropped around. The interested regions are the lesion area.
    qsm_labels: find the lesion area center in the QSM, also used for lesion size
    nm_labels: find the lesion area center in the NM.
    :return:
    '''

    case_names = os.listdir(database)
    for case_name in case_names:
        print(case_name)
        case_dir = os.path.join(database, case_name)

        qsm_mask_path = os.path.join(case_dir, 'QSM_mask.nii.gz')
        qsm_mask_image = sitk.ReadImage(qsm_mask_path)
        spacing = qsm_mask_image.GetSpacing()
        array = sitk.GetArrayFromImage(qsm_mask_image)
        zs_all, ys_all, xs_all = [], [], []
        for l in qsm_labels:
            zs, ys, xs = np.where(array == l)
            zs_all.append(zs)
            ys_all.append(ys)
            xs_all.append(xs)
        zs_all = np.concatenate(zs_all)
        ys_all = np.concatenate(ys_all)
        xs_all = np.concatenate(xs_all)
        z_mid = np.mean(zs_all)
        y_mid = np.mean(ys_all)
        x_mid = np.mean(xs_all)
        z_distance = zs_all.max() - zs_all.min()
        y_distance = ys_all.max() - ys_all.min()
        x_distance = x_mid.max() - x_mid.min()
        distance = max(z_distance*spacing[2], y_distance*spacing[1], x_distance*spacing[0]) # unit: mm
        distance = round(distance, 1)
        phy_loc_xyz = qsm_mask_image.TransformContinuousIndexToPhysicalPoint((x_mid, y_mid, z_mid))
        phy_loc_xyz = [round(l, 1) for l in phy_loc_xyz]
        np_loc_zyx = [int(round(z_mid)), int(round(y_mid)), int(round(x_mid))]
        qsm_phy_loc_xyz_str = f"[{phy_loc_xyz[0]}, {phy_loc_xyz[1]}, {phy_loc_xyz[2]}]"
        qsm_np_loc_zyx_str = f"[{np_loc_zyx[0]}, {np_loc_zyx[1]}, {np_loc_zyx[2]}]"

        nm_mask_path = os.path.join(case_dir, 'NM_mask.nii.gz')
        nm_mask_image = sitk.ReadImage(nm_mask_path)
        array = sitk.GetArrayFromImage(nm_mask_image)
        zs_all, ys_all, xs_all = [], [], []
        for l in nm_labels:
            zs, ys, xs = np.where(array == l)
            zs_all.append(zs)
            ys_all.append(ys)
            xs_all.append(xs)
        zs_all = np.concatenate(zs_all)
        ys_all = np.concatenate(ys_all)
        xs_all = np.concatenate(xs_all)
        z_mid = np.mean(zs_all)
        y_mid = np.mean(ys_all)
        x_mid = np.mean(xs_all)
        phy_loc_xyz = nm_mask_image.TransformContinuousIndexToPhysicalPoint((x_mid, y_mid, z_mid))
        phy_loc_xyz = [round(l, 1) for l in phy_loc_xyz]
        np_loc_zyx = [int(round(z_mid)), int(round(y_mid)), int(round(x_mid))]
        nm_phy_loc_xyz_str = f"[{phy_loc_xyz[0]}, {phy_loc_xyz[1]}, {phy_loc_xyz[2]}]"
        nm_np_loc_zyx_str = f"[{np_loc_zyx[0]}, {np_loc_zyx[1]}, {np_loc_zyx[2]}]"

        lesion = {
            "id": "0",
            "tag": "RJPD",
            "label": pd_map[case_name],
            "nifti_files": "['QSM.nii.gz', 'NM.nii.gz']",
            "phy_loc_xyz": f"[{qsm_phy_loc_xyz_str}, {nm_phy_loc_xyz_str}]",
            "np_loc_zyx": f"[{qsm_np_loc_zyx_str}, {nm_np_loc_zyx_str}]",
            "distance": f"{distance} mm",
          }

        json_file_path = os.path.join(case_dir, f'annotations.json')
        export_data = {0: lesion}
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        # break

'''
    Gen annotations for train data
'''
database = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/train_data/train'
pd_class_csv = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/train_data/train_cases.csv'
def load_pd_class_csv(pd_class_csv):
    with open(pd_class_csv, 'r') as f:
        lines = f.readlines()
    d = {l.split(',')[0].strip(): l.split(',')[1].strip() for l in lines}
    return d

'''
    Gen annotations for val data
'''
database = '/media/wfk/Tai/WFK_Data/RJ-PD/data20250615/val'
pd_class_csv = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/val_data/val_cases.csv'

'''
    Gen annotations for test data
'''
database = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/test_data/test'
pd_class_csv = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/test_data/test_cases.csv'
pd_map = load_pd_class_csv(pd_class_csv)
create_annotation_from_mask_RJPD(database, pd_map, [9,10,11,12], [1,2])

