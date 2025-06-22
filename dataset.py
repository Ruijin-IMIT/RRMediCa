
from typing import Any, Callable, Optional, Tuple
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import json
import os.path
import pickle
import SimpleITK as sitk
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

THREADS = 30

def get_dataset_split(data_container, folds_num = 5, train_ratio = 0.8, data_split_file = None):
    '''
    The 5-fold data splitting plan, store the split plan in a file for consistent reentry across different experiments.
    :param data_container: contain all the loaded data
    :param folds_num: 5 by default
    :param train_ratio: 20%
    :param data_split_file:
    :return:
    '''
    if data_split_file is None:
        data_split_file = 'data_split_file_5folds.pkl'
    if os.path.exists(data_split_file):
        data_split = pickle.load(open(data_split_file, 'rb'))
        return data_split
    case_names = list(data_container.case_dict.keys())
    case_names = list(set(case_names))
    N = len(case_names)
    test_N = int(N * (1/folds_num))
    train_N = int(N * (1 - 1/folds_num) * train_ratio)
    data_split = {}
    selected_testing_cases = set()
    for i in range(folds_num):
        if i == folds_num - 1:
            test_cases = list(set(case_names).difference(selected_testing_cases))
        else:
            remaining = list(set(case_names).difference(selected_testing_cases))
            test_cases = random.sample(remaining, test_N)
        trainval_cases = list(set(case_names).difference(test_cases))
        train_cases = random.sample(trainval_cases, train_N)
        val_cases = list(set(trainval_cases).difference(train_cases))
        data_split[i] = {'train': train_cases, 'val': val_cases, 'test': test_cases}
        selected_testing_cases = selected_testing_cases.union(set(test_cases))
    pickle.dump(data_split, open(data_split_file, 'wb'))
    return data_split


def gen_base_train_transforms(lesion_size):
    '''
    Add randomness into the lesion sampling process; for small lesions, do center crop first
    :param lesion_size:
    :return: transforms at least containing resizing (128)
    '''
    transform_train = [transforms.ToTensor()]
    # To make sure small lesion is included in the final cropping
    if lesion_size < 50:  # all small and middle sized lesions may be center-cropped
        if random.random() * lesion_size <= 25:  # 100% center crop for lesion<25;
            lesion_size = max(lesion_size, 20)
            center_crop_size = random.randint(max(50, int(2.5 * lesion_size)), min(150, int(3.5 * lesion_size)))
            # make the crop_size more certain, try to avoid acc fluctuating
            transform_train += [transforms.CenterCrop(center_crop_size),
                                transforms.Resize(128, antialias=True), ]
        else:
            transform_train += [transforms.Resize(128, antialias=True), ]
    else:
        transform_train += [transforms.Resize(128, antialias=True), ]

    return transform_train

def gen_transform_train_complex(lesion_size=40):
    '''
    The most complex data augmentation. [hardest]
    :param lesion_size:
    :return:
    '''
    transform_train = gen_base_train_transforms(lesion_size)
    transform_train += [
        transforms.RandomResizedCrop(80, (0.8, 1.2), (3 / 4, 4 / 3), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing(0.5, (0.02, 0.33), (0.3, 3.3), 'random'),
        transforms.RandomAffine(90, (0.20, 0.20)),
        # transforms.RandomPerspective(0.5, 0.5),
        # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.)),
    ]
    return transform_train

def gen_transform_train_easy01(lesion_size=40):
    transform_train = gen_base_train_transforms(lesion_size)
    transform_train += [
        transforms.RandomResizedCrop(80, (0.8, 1.2), (3 / 4, 4 / 3), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomPerspective(0.5, 0.5),
    ]
    return transform_train

def gen_transform_train_easy02(lesion_size=40): # center_crop_size = 80, resize_size=128
    transform_train = gen_base_train_transforms(lesion_size)
    transform_train += [
        transforms.RandomResizedCrop(80, (0.8, 1.2), (3 / 4, 4 / 3), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing(0.5, (0.02, 0.33), (0.3, 3.3), 'random'),
    ]
    return transform_train

def gen_transform_train_easy03(lesion_size=40): # center_crop_size = 80, resize_size=128
    transform_train = gen_base_train_transforms(lesion_size)
    transform_train += [
        transforms.CenterCrop(80),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.)),
    ]
    if random.random() < 0.5:
        transform_train += [transforms.RandomAffine(10, (0.20, 0.20)),]
    return transform_train

def gen_transform_train(lesion_size): # updated on 11/11, resize first to 128, re-scale small images
    '''
    Randomly select one transform during training augmentation
    :param lesion_size:
    :return:
    '''
    transform_train = [transforms.ToTensor()]
    rand_seed = random.randint(0, 5)
    if rand_seed == 0:
        transform_train = gen_transform_train_easy01(lesion_size)
    elif rand_seed == 1:
        transform_train = gen_transform_train_easy02(lesion_size)
    elif rand_seed == 2:
        transform_train = gen_transform_train_easy03(lesion_size)
    else:
        transform_train = gen_transform_train_complex(lesion_size)
    return transform_train

def gen_transform_test(lesion_size, lesion_enlarge_f=1):
    '''
    For val/test data augmentation: less or no distortion
    :param lesion_size:
    :param lesion_enlarge_f:
    :return:
    '''
    transform_test = [transforms.ToTensor(), transforms.Resize(80, antialias=True)]
    if lesion_size <= 40:
        lesion_size = max(lesion_size, 20)
        # make the crop_size more certain, try to avoid acc fluctuating
        center_crop_size = int(lesion_size * 2 * lesion_enlarge_f)
        transform_test = [transforms.ToTensor(), transforms.CenterCrop(center_crop_size),
                           transforms.Resize(80, antialias=True)]
    return transform_test

def array_normalize_uint8(array):
    new_array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    return new_array

def load_data_case_func(params):
    case_folder, database = params
    case_name = os.path.basename(case_folder)
    annotations_path = os.path.join(case_folder, 'annotations.json')
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    if len(annotations) != 1:
        print(f"Error, more than one annotation record for {case_folder}")
    region_id = list(annotations.keys())[0]
    region = annotations[region_id]
    nifti_files = eval(region['nifti_files'])
    distance_mm = float(region['distance'].replace('mm', '').strip().split()[0])  # extract the numerical part
    phy_loc_xyz = eval(region['phy_loc_xyz'])
    region['nifti_files'] = nifti_files
    region['distance_mm'] = distance_mm
    region['phy_loc_xyz'] = phy_loc_xyz
    region['arrays'] = []
    region['mask_arrays'] = []
    region['np_sizes'] = []
    region['case_name'] = case_name

    # for multiple modality inputs, load all information
    for nifti_file in nifti_files:
        nifti_path = os.path.join(case_folder, nifti_file)
        if not os.path.exists(nifti_path):
            print(f'Error: nifti file not found: {nifti_path}') # TODO should add more handling
        image = sitk.ReadImage(nifti_path)
        # get the spacial information, and create a pseudo mask; put image, mask, infor in the region record
        spacing = image.GetSpacing()
        array = sitk.GetArrayFromImage(image)
        Z, Y, X = array.shape
        region['arrays'].append(array_normalize_uint8(array))
        np_pixels = region['distance_mm'] / spacing[0]
        shadow_half = min(Y // 2 - 2, np_pixels // 2)
        mask_array = np.zeros_like(array[0]).astype(bool) # only one slice
        mask_array[int(Y // 2 - shadow_half):int(Y // 2 + shadow_half), int(X // 2 - shadow_half):int(X // 2 + shadow_half)] = True
        region['mask_arrays'].append(mask_array)
        region['np_sizes'].append(np_pixels)

    return (case_name, region)

class Data_container():
    def __init__(self, database, class_names):
        '''
        Data loading and parsing facility, dedicated to Medical image analysis.
        :param database: the data root of all cases. Each case folder contains corresponding images and annotations.
        '''
        params_list = []
        case_names = os.listdir(database)
        for i, case_name in enumerate(case_names):
            case_folder = os.path.join(database, case_name)
            params_list.append((case_folder, database))

        pool = ThreadPool(THREADS)
        results = pool.map(load_data_case_func, params_list)  # starmap_async
        pool.close()
        print(f"Data_container: {len(results)} cases loaded")
        self.case_dict = {case_name:case_content for case_name, case_content in results}
        self.class_names = class_names
    def get_stats_class_distribution(self, class_names = None):
        if class_names is None:
            class_names = self.class_names
        len_case = len(self.case_dict)
        case_names = list(self.case_dict.keys())
        label_count_np = np.zeros([len_case, len(class_names)])
        for index in range(len_case):
            case_name = case_names[index]
            lesion = self.case_dict[case_name]
            values = []
            for class_name in class_names:
                value = int(lesion[class_name])
                values.append(value)
            label_count_np[index] = values

        target_classes = []
        target_class_counts = []
        for i in range(len(class_names)):
            cur_class_values = label_count_np[:, i]
            class_N = cur_class_values.max() + 1
            target_classes.append(int(class_N))
            unique, counts = np.unique(cur_class_values, return_counts=True)
            # print(f"i={i}, unique={unique}, counts={counts}")
            value_count_pairs = zip(unique, counts)
            value_count_pairs = sorted(value_count_pairs, key=lambda x: x[0])
            class_count = [int(c) for v, c in value_count_pairs]
            target_class_counts.append(class_count)
        # print(f" stats_class_distribution: target_classes={target_classes}, target_class_counts={target_class_counts}")
        return target_classes, target_class_counts

class Torch_Dataset(data.Dataset):
    def __init__(self, data_container, nifti_files, image_slice_N = 5, data_split_file=None, fold = 0, split_key = 'train', transform_key = 'train', amplifier = 1) -> None:
        self.split_key = split_key  # training set or test set
        self.transform_key = transform_key
        self.data_container = data_container
        self.case_dict = data_container.case_dict
        self.class_names = data_container.class_names
        # specify the nifti files/modalities used in this project,

        if data_split_file is None: # take all the cases in, set the fold to 0, train/val/test get all samples
            fold = 0
            all_cases = list(self.case_dict.keys())
            self.data_split = {0: {'train': all_cases, 'val': all_cases, 'test': all_cases}}
        else:
            self.data_split = get_dataset_split(data_container, folds_num = 5, train_ratio = 0.8, data_split_file = data_split_file)

        self.nifti_files = nifti_files
        self.image_slice_N = image_slice_N
        self.fold = fold

        self.tensor_size = 80 # the width and height of image slice
        # split_key can be one of : train, val, test, train_val
        cases = set()
        if split_key == 'train':
            cases = self.data_split[fold]['train']
        elif split_key == 'val':
            cases = self.data_split[fold]['val']
        elif split_key == 'test':
            cases = self.data_split[fold]['test']
        elif split_key == 'train_val':
            cases = list(self.data_split[fold]['train']) + list(self.data_split[fold]['val'])
        elif split_key == 'val_test':
            cases = list(self.data_split[fold]['val']) + list(self.data_split[fold]['test'])
        elif split_key == 'train_val_test':
            cases = list(self.data_split[fold]['train']) + list(self.data_split[fold]['val']) + list(self.data_split[fold]['test'])
        self.case_names = list(set(cases))
        # self.stats_class_distribution()
        self.case_names = self.case_names * amplifier # augment the number of training cases; for test/val: tilt to mean performance of many trails

    def __len__(self) -> int:
        return len(self.case_names)

    def mask_random_modalities(self, mask_prob = 0.25):
        '''
        Depend on self.nifti_files.
        To mask the modality with mask_prob. Only for multi-modal image inputs, increase robustness.
        :return:
        '''
        status_map = {}
        for mod in self.nifti_files:
            status_map[mod] = 1
            if random.random() < mask_prob:
                status_map[mod] = 0
        return status_map

    def get_array_slices(self, L, N, rand_slice=False):
        '''
        Return the randomly sampled array slices.
        Purpose: increase the randomness during data sampling.
        :param L: total len of slices
        :param N: the desired number of slices
        :param rand_slice: whether to randomize the slices
        :return:
        '''
        if L == 1:
            inds = [0] * N
        elif N <= L:
            if rand_slice:  # only for training
                starting_index = random.randint(0, L - N)
                inds = [i for i in range(starting_index, starting_index + N)]
            else:  # this is for testing/validation, sample around the region centroid
                starting_index = (L - N) // 2
                inds = [i for i in range(starting_index, starting_index + N)]
        else:
            ratio = (L - 1) / (N-1)
            inds = [round(i * ratio) for i in range(N)]
        return inds

    def __getitem__(self, index: int):
        case_name = self.case_names[index]
        lesion = self.case_dict[case_name]
        tensorlist = []
        status_map = self.mask_random_modalities()

        for nifti_file in self.nifti_files:
            if nifti_file not in lesion['nifti_files']:
                tensor = torch.randn([self.image_slice_N+1, self.tensor_size, self.tensor_size])
                tensorlist.append(tensor)
                continue
            nifti_ind = lesion['nifti_files'].index(nifti_file)
            array = lesion['arrays'][nifti_ind] # already of type uint8
            mask_array = lesion['mask_arrays'][nifti_ind]
            np_size = lesion['np_sizes'][nifti_ind]
            Z, Y, X = array.shape
            mid = Z // 2
            inds = self.get_array_slices(Z, self.image_slice_N, self.transform_key == 'train')
            comb_array = np.stack([array[i] for i in inds] + [(mask_array[:, :]*255).astype(np.uint8)], 2)
            comb_array = comb_array.astype(np.uint8) # must make sure: data type is uint8, or to_tensor() would not work
            if self.transform_key == 'train':
                transform = gen_transform_train(np_size)
            else:
                transform = gen_transform_test(np_size, 1)

            for t in transform:
                comb_array = t(comb_array)

            tensor = comb_array

            # Add randomness: randomly supress some input modalities
            # if self.train and status_map[nifti_file] == 0:
            #     tensor = torch.randn_like(tensor)

            # testing scenario: modality omission
            # if not self.train and modality not in self.mod_plan:
            #     tensor = torch.randn_like(tensor)

            if comb_array.shape[1] != self.tensor_size or comb_array.shape[2] != self.tensor_size:
                # print(f'Found irregular crop shape: {comb_array.shape} in {case_name}, {nifti_file}')
                tensor = torch.randn([self.image_slice_N + 1, self.tensor_size, self.tensor_size])
            tensorlist.append(tensor)

        tensor_input = torch.cat(tensorlist, 0)

        # Prepare the predicting targets, put them in a list
        targets = []
        for class_name in self.class_names:
            t = int(lesion[class_name])
            targets.append(t)

        return case_name, tensor_input, torch.tensor(targets)


if __name__ == '__main__':
    database = None
    class_names = None
    data_container = Data_container(database, class_names)






