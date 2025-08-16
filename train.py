'''
Author: Fakai Wang
Affiliation: Ruijn Hospital, Shanghai Jiao Tong University School of Medicine

'''

from common import *
import dataset

train_batch_size = 128 #256#128# 512
test_batch_size = 128 #256#128 #512

def main_train(data_container, data_split_file, ckpt_dir, exp_group, model_name, folds, nifti_files, in_slice_N, target_classes, target_class_counts):
    in_modality_N = len(nifti_files)
    train_amp = 10
    train_split = 'train_val'
    val_split = 'test'
    test_split = 'test'
    stats = {}
    losses = {}
    for fold in folds:
        logger.info(f'Training exp_group={exp_group}, model_name={model_name}, Fold = {fold}')
        stats[fold] = {}
        losses[fold] = {}
        start_epoch = 0
        Epoch_Num = 200
        data_slice_N = in_slice_N - 1
        trainset = dataset.Torch_Dataset(data_container, nifti_files, data_slice_N, data_split_file, fold, train_split, 'train', train_amp)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=16)
        valset = dataset.Torch_Dataset(data_container, nifti_files, data_slice_N, data_split_file, fold, val_split, 'test', 2)
        valloader = torch.utils.data.DataLoader(valset, batch_size=train_batch_size, shuffle=False, num_workers=16)
        testset = dataset.Torch_Dataset(data_container, nifti_files, data_slice_N, data_split_file, fold, test_split, 'test', 2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=train_batch_size, shuffle=False, num_workers=16)

        print(f'\n==>  ****  Exp {exp_group}, model_name=', model_name, ', fold=', fold, '  * ** ***  ')
        model_path = f'{ckpt_dir}/{exp_group}_{model_name}_F{fold}.pth'
        net, criterions, optimizer, scheduler = load_model(model_path, exp_group, model_name, False, in_modality_N, in_slice_N, target_classes, target_class_counts, fold, True)
        training_records = {'best_5':{}, 'best_epoch':None, 'best_acc':0, 'Last_Epoch':start_epoch+Epoch_Num-1}
        training_records.update({'ckpt_dir':ckpt_dir, 'exp_group':exp_group, 'model_name':model_name, 'nifti_files':nifti_files, 'in_slice_N':in_slice_N, 'target_classes':target_classes, 'fold':fold})
        stats[fold]['train'] = np.zeros([Epoch_Num, len(target_classes)])
        stats[fold]['val'] = np.zeros([Epoch_Num, len(target_classes)])
        stats[fold]['test'] = np.zeros([Epoch_Num, len(target_classes)])
        losses[fold]['train'] = np.zeros([Epoch_Num, len(target_classes)])
        losses[fold]['val'] = np.zeros([Epoch_Num, len(target_classes)])
        losses[fold]['test'] = np.zeros([Epoch_Num, len(target_classes)])
        # continue
        for epoch in range(start_epoch, start_epoch+Epoch_Num):
            result = {}
            train(epoch, net, criterions, optimizer, trainloader, target_classes, None, stats[fold]['train'], losses[fold]['train'])
            val(epoch, net, criterions, valloader, target_classes, result, stats[fold]['val'], losses[fold]['val'], ckpt_dir, exp_group,
                model_name, fold, True, training_records, 'val')
            # val(epoch, net, criterions, testloader, target_classes, result, stats[fold]['test'], losses[fold]['test'], ckpt_dir, exp_group,
            #     model_name, fold, False, training_records, 'test')
            scheduler.step()
        net, criterions, optimizer, scheduler = load_model(model_path, exp_group, model_name, True, in_modality_N, in_slice_N, target_classes, target_class_counts, fold,True)
        result = {}
        stats_test = np.zeros([1, len(target_classes)])
        losses_test = np.zeros([1, len(target_classes)])
        val(0, net, criterions, testloader, target_classes, result, stats_test, losses_test, ckpt_dir, exp_group, model_name, fold, False, training_records, 'test')
        save_model_train_val_evaluate_result(ckpt_dir, exp_group, model_name, fold, target_classes, stats, losses, result['case_names'], result['gts'], result['preds'])
    return stats, losses
def train_project_liver_mvi():
    database = '/media/wfk/Tai/WFK_Data/RJ-Liver/RJ_Liver_MRI_02_part1part2_lesions'
    data_split_file = '/media/wfk/Tai/Codes/ScriptsSJTU/rj-liver/rrmedica/data_process/liver_mvi_datasplit_5folds.pkl'
    data_container = dataset.Data_container(database)
    class_names = ['mvi', 'vetc', 'mtm']
    target_classes, target_class_counts = data_container.get_stats_class_distribution(class_names)
    # exit()
    device = 'cuda:0'
    NetFamily = 'VGG'
    ckpts_root = './ckpts'
    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)
    nifti_files_list = [['t2.nii.gz'],
                        ['t2.nii.gz', 'lap.nii.gz'],
                        ['t2.nii.gz', 'lap.nii.gz', 'pv.nii.gz'],
                        ['t2.nii.gz', 'lap.nii.gz', 'pv.nii.gz', 'dwi(b1).nii.gz'],
                        ['adc.nii.gz', 'dwi(b0).nii.gz', 'dwi(b1).nii.gz', 'ip.nii.gz', 'op.nii.gz',
                         'lap.nii.gz', 'pv.nii.gz', 'delay.nii.gz', 'preartery.nii.gz', 't2.nii.gz']
                        ]

    nifti_files_list = [['t2.nii.gz', 'lap.nii.gz', 'pv.nii.gz', 'dwi(b1).nii.gz']
                        ]

    for exp_group in ['Aux']:  # 'RelayAttn', 'Relay', , 'MF' 'RelayAttn', 'Relay', 'MF', 'Attn'
        for model_name in ['VGG5.7', 'VGG5.5', 'VGG5.3', 'VGG5.1']:
            for nifti_files in nifti_files_list:
                stats_all = {}
                losses_all = {}
                nifti_files_tag = '_'.join([n[:-7] for n in nifti_files])
                exp_folder_name = f"{exp_group}_{model_name}_{nifti_files_tag}"
                ckpt_dir = os.path.join(ckpts_root, exp_folder_name)
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                data_container.nifti_files = nifti_files

                in_modality = len(nifti_files)
                logger.info(f' ***  New Experiment:In modality = {in_modality} ** nifti_files={nifti_files}')
                logger.info(f' Network config: exp_group = {exp_group}, model_name = {model_name} ** ')
                folds = [0,1,2,3,4]
                in_modality_N = len(nifti_files)
                in_slice_N = 6 # the total slice number in the model, image slices and mask slice := image_slice_N + 1
                stats, losses = main_train(data_container, data_split_file, ckpt_dir, exp_group, model_name, folds, nifti_files, in_slice_N, target_classes, target_class_counts)
                print(stats)
                stats_all[f"{exp_group}_{model_name}_{nifti_files_tag}"] = stats
                losses_all[f"{exp_group}_{model_name}_{nifti_files_tag}"] = losses
                transform_stats_to_cvs(ckpt_dir, stats_all, losses_all)


def train_project_RJPD():
    database = '/home/fakai/Documents/Data/RRMediCa/train_lesion_crops'
    data_split_file = '/home/fakai/Documents/Data/RRMediCa/RJPD_datasplit_5folds.pkl'
    class_names = ['label']
    data_container = dataset.Data_container(database, class_names)
    target_classes, target_class_counts = data_container.get_stats_class_distribution()
    print(target_classes, target_class_counts)
    ckpts_root = '/home/fakai/Documents/Data/RRMediCa/ckpts_RJPD01'
    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)
    nifti_files_list = [['NM.nii.gz'],
                        ['QSM.nii.gz'],
                        ['QSM.nii.gz', 'NM.nii.gz'],
                        ]
    model_names = ['VGG1.1', 'VGG1.3', 'VGG1.5', 'resnet1.1', 'resnet1.2', 'resnet1.3', 'shufflenet1.1',
                   'shufflenet1.2', 'shufflenet1.3', 'densenet1.1', 'densenet1.2', 'densenet1.3']

    for exp_group in ['Aux']:  # 'RelayAttn', 'Relay', , 'MF' 'RelayAttn', 'Relay', 'MF', 'Attn'
        for nifti_files in nifti_files_list:
            for model_name in model_names:
                stats_all = {}
                losses_all = {}
                nifti_files_tag = '_'.join([n[:-7] for n in nifti_files])
                exp_folder_name = f"{exp_group}_{model_name}_{nifti_files_tag}"
                ckpt_dir = os.path.join(ckpts_root, exp_folder_name)
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                data_container.nifti_files = nifti_files

                in_modality = len(nifti_files)
                logger.info(f' ***  New Experiment:In modality = {in_modality} ** nifti_files={nifti_files}')
                logger.info(f' Network config: exp_group = {exp_group}, model_name = {model_name} ** ')
                folds = [0, 1, 2, 3, 4]
                in_modality_N = len(nifti_files)
                in_slice_N = 6 # the total slice number in the model, image slices and mask slice := image_slice_N + 1
                stats, losses = main_train(data_container, data_split_file, ckpt_dir, exp_group, model_name, folds, nifti_files, in_slice_N, target_classes, target_class_counts)
                print(stats)
                stats_all[f"{exp_group}_{model_name}_{nifti_files_tag}"] = stats
                losses_all[f"{exp_group}_{model_name}_{nifti_files_tag}"] = losses
                transform_stats_to_cvs(ckpt_dir, stats_all, losses_all)


if __name__ == '__main__':
    train_project_RJPD()


