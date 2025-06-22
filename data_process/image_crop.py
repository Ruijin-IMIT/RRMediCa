
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



def extract_liver_lesions(directory, newbase=None):
    """
    从目录中的多个肝脏磁共振影像中截取病灶区域

    参数:
        directory (str): 包含影像和annotations.json文件的目录路径
    """
    # 读取annotations.json文件
    case_name = os.path.basename(directory)
    annotations_path = os.path.join(directory, 'annotations.json')
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # 创建lesions子目录
    if not newbase:
        lesion_path = os.path.join(directory, 'lesions')
        os.makedirs(lesion_path, exist_ok=True)

    # print(f"Working on {directory}")
    for lesion_id in annotations:
        # 解析病灶信息
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
            # lesions_dir = os.path.join(newbase, 'lesions')
            os.makedirs(lesion_path, exist_ok=True)

            json_file_path = os.path.join(lesion_path, f'annotations.json')
            export_data = {lesion_id:lesion}
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)


        # nifti_files = lesion['nifti_files']
        # phy_loc_xyz = lesion['phy_loc_xyz']

        nifti_files = eval(lesion['nifti_files'])
        phy_loc_xyz_list = eval(lesion['phy_loc_xyz'])
        # np_loc_zyx = eval(lesion['np_loc_zyx'])

        # 计算截取尺寸
        xy_size = max(2 * distance_mm, 50)  # XY方向: 2倍病灶尺寸，至少50mm
        z_size = (1.5 * distance_mm)  # Z方向: 1.5倍病灶尺寸，最多9个slice

        for nii_i, nifti_file in enumerate(nifti_files):
            phy_loc_xyz = phy_loc_xyz_list[nii_i]
            # 读取NIfTI文件
            nifti_path = os.path.join(directory, nifti_file)
            image = sitk.ReadImage(nifti_path)

            # 获取图像的空间信息
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            direction = image.GetDirection()
            size = image.GetSize()


            # 使用SimpleITK内置方法转换物理坐标到体素索引
            voxel_loc = image.TransformPhysicalPointToIndex(tuple(phy_loc_xyz))
            voxel_loc = [int(round(x)) for x in voxel_loc]  # 确保为整数

            # 计算截取范围(XY方向)
            xy_radius_voxel = int(round(xy_size / (2 * spacing[0])))  # 假设X和Y的spacing相同
            x_start = max(0, voxel_loc[0] - xy_radius_voxel)
            x_end = min(size[0], voxel_loc[0] + xy_radius_voxel)
            y_start = max(0, voxel_loc[1] - xy_radius_voxel)
            y_end = min(size[1], voxel_loc[1] + xy_radius_voxel)

            # 计算截取范围(Z方向)
            z_radius_voxel = int(round(z_size / (2 * spacing[2])))
            z_radius_voxel = min(z_radius_voxel, 4)
            z_start = max(0, voxel_loc[2] - z_radius_voxel)
            z_end = min(size[2], voxel_loc[2] + z_radius_voxel + 1)

            # 使用SimpleITK的RegionOfInterest滤波器截取ROI
            roi_filter = sitk.RegionOfInterestImageFilter()
            roi_filter.SetIndex([x_start, y_start, z_start])
            roi_filter.SetSize([x_end - x_start, y_end - y_start, z_end - z_start])
            roi_image = roi_filter.Execute(image)

            # 生成输出文件名
            base_name = nifti_file[:-7]
            output_filename = f"lesion_{lesion_id}_{base_name}_{distance_mm}mm.nii.gz"
            if newbase:
                output_filename = nifti_file
            output_path = os.path.join(lesion_path, output_filename)

            # 保存截取的ROI
            sitk.WriteImage(roi_image, output_path)
            print(f"Saved lesion ROI to {output_path}")

def extract_database_lesions(database, newbase=None):
    case_names = os.listdir(database)
    for case_name in case_names:
        case_dir = os.path.join(database, case_name)
        extract_liver_lesions(case_dir, newbase)


if __name__ == '__main__':
    database = '/media/wfk/Tai/WFK_Data/RJ-Liver/RJ_Liver_MRI_02_part1part2_normalized'
    newbase = '/media/wfk/Tai/WFK_Data/RJ-Liver/RJ_Liver_MRI_02_part1part2_lesions'
    # extract_database_lesions(database, newbase)

    database = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/train_data/train'
    newbase = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/train_data/train_lesion_crops'
    # extract_database_lesions(database, newbase)

    database = '/media/wfk/Tai/WFK_Data/RJ-PD/data20250615/val'
    newbase = '/media/wfk/Tai/WFK_Data/RJ-PD/data20250615/val_lesion_crops'
    # extract_database_lesions(database, newbase)

    database = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/test_data/test'
    newbase = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/test_data/test_lesion_crops'
    extract_database_lesions(database, newbase)





