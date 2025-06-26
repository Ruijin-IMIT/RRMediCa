
import copy
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


def extract_lesions(directory, newbase=None):
    """
    Extract the ROI cropping for patient dir

    directory: the patient dir, containing .nii.gz files and annotations.json
    newbase: put the extracted ROIs in a new directory if set
    annotations.json: contain the detailed ROI definition, including related .nii.gz files and locations and size (distance).
    """
    # read the annotations
    if not os.path.isdir(directory):
        return
    case_name = os.path.basename(directory)
    annotations_path = os.path.join(directory, 'annotations.json')
    if not os.path.isfile(annotations_path):
        print(f"{annotations_path} does not exist")
        return
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # create lesions directory
    if not newbase:
        lesion_path = os.path.join(directory, 'lesions')
        os.makedirs(lesion_path, exist_ok=True)

    # print(f"Working on {directory}")
    for lesion_id in annotations:
        # parse the lesion/ROI records
        lesion = annotations[lesion_id]
        lesion_id = lesion['id']
        if len(lesion['distance'].strip()) == 0:
            print(f"Working on {directory}, Infor is empty {lesion}")
            # exit()
            # continue

        distance_mm = float(lesion['distance'].replace('mm', '').strip().split()[0])  # 提取数值部分

        if newbase:
            lesion_dir_name = f'{case_name}_lesion{lesion_id}_{distance_mm}mm'
            lesion_path = os.path.join(newbase, lesion_dir_name)
            os.makedirs(lesion_path, exist_ok=True)
            json_file_path = os.path.join(lesion_path, f'annotations.json')
            export_data = {lesion_id:lesion}
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        nifti_files = eval(lesion['nifti_files'])
        phy_loc_xyz_list = eval(lesion['phy_loc_xyz'])
        # np_loc_zyx = eval(lesion['np_loc_zyx'])

        # calculate the size
        xy_size = max(2 * distance_mm, 50)  # XY direction: 2 times the core size， and at least 50mm
        z_size = (1.5 * distance_mm)  # Z direction: 1.5 times core size，and at most 9 slices

        for nii_i, nifti_file in enumerate(nifti_files):
            phy_loc_xyz = phy_loc_xyz_list[nii_i]
            nifti_path = os.path.join(directory, nifti_file)
            image = sitk.ReadImage(nifti_path)

            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            direction = image.GetDirection()
            size = image.GetSize()

            voxel_loc = image.TransformPhysicalPointToIndex(tuple(phy_loc_xyz))
            voxel_loc = [int(round(x)) for x in voxel_loc]  # 确保为整数

            # claculate the cropping range
            xy_radius_voxel = int(round(xy_size / (2 * spacing[0])))  # 假设X和Y的spacing相同
            x_start = max(0, voxel_loc[0] - xy_radius_voxel)
            x_end = min(size[0], voxel_loc[0] + xy_radius_voxel)
            y_start = max(0, voxel_loc[1] - xy_radius_voxel)
            y_end = min(size[1], voxel_loc[1] + xy_radius_voxel)

            z_radius_voxel = int(round(z_size / (2 * spacing[2])))
            z_radius_voxel = min(z_radius_voxel, 4)
            z_start = max(0, voxel_loc[2] - z_radius_voxel)
            z_end = min(size[2], voxel_loc[2] + z_radius_voxel + 1)

            # use SimpleITK RegionOfInterest to extract ROI
            roi_filter = sitk.RegionOfInterestImageFilter()
            roi_filter.SetIndex([x_start, y_start, z_start])
            roi_filter.SetSize([x_end - x_start, y_end - y_start, z_end - z_start])
            roi_image = roi_filter.Execute(image)

            base_name = nifti_file[:-7]
            output_filename = f"lesion_{lesion_id}_{base_name}_{distance_mm}mm.nii.gz"
            if newbase:
                output_filename = nifti_file
            output_path = os.path.join(lesion_path, output_filename)

            sitk.WriteImage(roi_image, output_path)
            print(f"Saved lesion ROI to {output_path}")

def extract_database_lesions(database, newbase=None):
    case_names = os.listdir(database)
    for case_name in case_names:
        case_dir = os.path.join(database, case_name)
        extract_lesions(case_dir, newbase)

if __name__ == '__main__':
    database = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/train_data/train'
    newbase = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/train_data/train_lesion_crops'
    # extract_database_lesions(database, newbase)

    database = '/media/wfk/Tai/WFK_Data/RJ-PD/data20250615/val'
    newbase = '/media/wfk/Tai/WFK_Data/RJ-PD/data20250615/val_lesion_crops'
    # extract_database_lesions(database, newbase)

    database = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/test_data/test'
    newbase = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/test_data/test_lesion_crops'
    extract_database_lesions(database, newbase)





