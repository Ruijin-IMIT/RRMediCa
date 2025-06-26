
from . import dataset
from .common import *

train_batch_size = 128 #256#128# 512
test_batch_size = 128 #256#128 #512


def main_test(data_container, data_split_file, model_path):
    model_file_name = os.path.basename(model_path)
    ckpt_dir = os.path.dirname(model_path)
    model_dir_name = os.path.basename(ckpt_dir)
    items = model_dir_name.split('_')
    exp_group = items[0]
    model_name = items[1]
    nifti_files = [ i + '.nii.gz' for i in items[2:]]

    if not os.path.exists(model_path):
        print(f"Error: {model_path} does not exist!")
        return None
    checkpoint = torch.load(model_path)
    training_records = checkpoint['training_records']
    exp_group = training_records['exp_group']
    model_name = training_records['model_name']
    nifti_files = training_records['nifti_files']
    in_slice_N = training_records['in_slice_N']
    target_classes = training_records['target_classes']
    fold = training_records['fold']
    target_class_counts = None
    in_modality_N = len(nifti_files)
    test_split = 'test'
    stats = {}
    losses = {}
    stats[fold] = {}
    losses[fold] = {}
    data_slice_N = in_slice_N - 1
    testset = dataset.Torch_Dataset(data_container, nifti_files, data_slice_N, data_split_file, fold, test_split, 'test', 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=train_batch_size, shuffle=False, num_workers=16)

    print(f'\n==>  *** **  Exp {exp_group}, model_name=', model_name, ', fold=', fold, '  * ** ***  ')
    # load the default model in the checkpoint folder
    net, criterions, optimizer, scheduler = load_model(model_path, exp_group, model_name, True, in_modality_N, in_slice_N, target_classes, target_class_counts, fold,True)
    result = {}
    stats_test = np.zeros([1, len(target_classes)])
    losses_test = np.zeros([1, len(target_classes)])
    # fold = training_records['fold']

    val(0, net, criterions, testloader, target_classes, result, stats_test, losses_test, ckpt_dir, exp_group, model_name, fold, False, training_records, 'test')
    # print(stats_test)
    return result


def ensemble(ckpts_root, results, gts):
    key_preds_list = [(r_key, results[r_key]['preds']) for r_key in results ]
    for key, l in key_preds_list:
        print(f"{key}: {(results[key]['preds'] == gts).sum() / len(gts):.03f}")
    preds_list = [l for k,l in key_preds_list]
    preds_all = np.asarray(preds_list)
    M, C, T = preds_all.shape # the number of Model, Case, Target
    print(f"preds_all.shape: {preds_all.shape}")
    preds_ensenble = np.zeros_like(preds_all[0])
    probs_ensenble = np.zeros_like(preds_all[0])
    probs_ensenble = probs_ensenble.astype(np.float32)
    for c in range(C):
        for t in range(T):
            values = preds_all[:, c, t]
            unique_values, counts = np.unique(values, return_counts=True)
            value_count_pair = [(value, count) for value, count in zip(unique_values, counts)]
            value_count_pair = sorted(value_count_pair, key=lambda x: x[1], reverse=True)
            value = value_count_pair[0][0]
            count = value_count_pair[0][1]
            preds_ensenble[c, t] = value
            probs_ensenble[c, t] = round(count / len(values), 3)

    acc = (preds_ensenble == gts).sum() / len(gts)
    print(f"Ensemble acc=", round(acc, 3))

    flags = preds_ensenble == gts
    print(flags.shape, probs_ensenble.shape)
    for prob_thres in [0.001, 0.01, 0.02, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        confident_flags = flags[probs_ensenble > prob_thres]
        if len(confident_flags) < len(flags) * 0.2:
            continue
        acc = confident_flags.sum() / len(confident_flags)
        print(f"Ensemble (at least {prob_thres*100} % confidence, total={len(confident_flags)}) acc=", round(acc, 3))
    return preds_ensenble

def select_and_ensemble(ckpts_root, results, gts):
    model_dir_names = set([r_key.split('/')[0] for r_key in results])
    model_keys = []
    for model_dir_name in model_dir_names:
        if not os.path.isdir(os.path.join(ckpts_root, model_dir_name)):
            continue
        exp_group = model_dir_name.split('_')[0]
        model_name = model_dir_name.split('_')[1]
        train_result_file_path = os.path.join(ckpts_root, model_dir_name, f'{exp_group}_{model_name}_All_results.pkl')
        with open(train_result_file_path, 'rb') as f:
            train_results_all = pickle.load(f)
        # print(train_result_file_path)
        for fold in range(5):
            if f'{exp_group}_{model_name}_F{fold}' not in train_results_all:
                continue
            model_result = train_results_all[f'{exp_group}_{model_name}_F{fold}']
            test_preds = model_result['preds']
            test_gts = model_result['gts']
            if (test_preds == test_gts).sum() / len(test_gts) < 0.55:
                continue
            if model_result['stats'][fold]['val'][-5:].mean() < 0.62:
                continue
            if model_result['stats'][fold]['val'][-5:].std() > 0.005:
                continue
            # model_key = f'{model_dir_name}/{exp_group}_{model_name}_F{fold}.pth' # select the 'best' model
            model_key = f'{model_dir_name}/{exp_group}_{model_name}_F{fold}_last.pth' # select the last model
            model_keys.append(model_key)

    print("Selected model_keys:", model_keys)
    key_preds_list = [(r_key, results[r_key]['preds']) for r_key in results if r_key in model_keys]
    for key, l in key_preds_list:
        print(f"{key}: {(results[key]['preds'] == gts).sum() / len(gts):.03f}")
    preds_list = [l for k,l in key_preds_list]
    preds_all = np.asarray(preds_list)
    M, C, T = preds_all.shape # the number of Model, Case, Target
    print(f"preds_all.shape: {preds_all.shape}")
    preds_ensenble = np.zeros_like(preds_all[0])
    probs_ensenble = np.zeros_like(preds_all[0])
    probs_ensenble = probs_ensenble.astype(np.float32)
    for c in range(C):
        for t in range(T):
            values = preds_all[:, c, t]
            unique_values, counts = np.unique(values, return_counts=True)
            value_count_pair = [(value, count) for value, count in zip(unique_values, counts)]
            value_count_pair = sorted(value_count_pair, key=lambda x: x[1], reverse=True)
            value = value_count_pair[0][0]
            count = value_count_pair[0][1]
            preds_ensenble[c, t] = value
            probs_ensenble[c, t] = round(count / len(values), 3)

    acc = (preds_ensenble == gts).sum() / len(gts)
    print(f"Ensemble acc=", round(acc, 3))

    flags = preds_ensenble == gts
    print(flags.shape, probs_ensenble.shape)
    for prob_thres in [0.001, 0.01, 0.02, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        confident_flags = flags[probs_ensenble > prob_thres]
        if len(confident_flags) < len(flags) * 0.2:
            continue
        acc = confident_flags.sum() / len(confident_flags)
        print(f"Ensemble (at least {prob_thres*100} % confidence, total={len(confident_flags)}) acc=", round(acc, 3))


def test_project_RJPD(database, ckpts_root, model_keys=None, force_predict=True):
    class_names = ['label']
    data_container = dataset.Data_container(database, class_names)
    target_classes, target_class_counts = data_container.get_stats_class_distribution()
    print(target_classes, target_class_counts)

    if model_keys is None:
        model_keys = ['Aux_resnet1.3_QSM/Aux_resnet1.3_F3_last.pth', 'Aux_VGG1.3_NM/Aux_VGG1.3_F2_last.pth',
                  'Aux_VGG1.5_QSM/Aux_VGG1.5_F4_last.pth', 'Aux_resnet1.2_NM/Aux_resnet1.2_F2_last.pth',
                  'Aux_shufflenet1.1_NM/Aux_shufflenet1.1_F2_last.pth', 'Aux_resnet1.1_NM/Aux_resnet1.1_F2_last.pth',
                  'Aux_densenet1.1_NM/Aux_densenet1.1_F2_last.pth',
                  'Aux_shufflenet1.3_NM/Aux_shufflenet1.3_F1_last.pth',
                  'Aux_VGG1.5_NM/Aux_VGG1.5_F2_last.pth', 'Aux_densenet1.3_QSM/Aux_densenet1.3_F4_last.pth',
                  'Aux_densenet1.2_QSM/Aux_densenet1.2_F1_last.pth', 'Aux_densenet1.2_QSM/Aux_densenet1.2_F2_last.pth',
                  'Aux_resnet1.1_QSM/Aux_resnet1.1_F3_last.pth', 'Aux_VGG1.1_NM/Aux_VGG1.1_F1_last.pth']

    def find_models(model_root):
        models = []
        model_dir_names = os.listdir(model_root)
        for model_dir_name in model_dir_names:
            model_dir = os.path.join(model_root, model_dir_name)
            if not os.path.isdir(model_dir):
                continue
            items = os.listdir(model_dir)
            # best_models = [m for m in items if m[-4:] == '.pth' and m[-7:-5] == '_F']
            best_models = [m for m in items if m[-9:] == '_last.pth']
            models.extend([os.path.join(model_dir, m) for m in best_models])
        return models
        
    models = find_models(ckpts_root)
    models = [os.path.join(ckpts_root, key) for key in model_keys]
    print(f'Selected models:{models}')

    results = {}
    def predict():
        for model_path in models:
            result = main_test(data_container, None, model_path)
            r_key = model_path.split('/')[-2] + '/' + model_path.split('/')[-1]
            results[r_key] = result
            print(result['preds'].shape)
            # print(result['preds'].tolist())
            acc = (result['gts'] == result['preds']).sum() / len(result['gts'])
            print(f"model_path:={model_path}, result=", round(acc, 3))

    ensemble_result_file = os.path.join(ckpts_root, 'ensemble_results.pkl')
    ensemble_result_file = os.path.join(ckpts_root, 'last_ensemble_results.pkl')
    if not force_predict and os.path.exists(ensemble_result_file):
        with open(ensemble_result_file, 'rb') as f:
            results = pickle.load(f)
    else:
        predict()
        with open(ensemble_result_file, 'wb') as f:
            pickle.dump(results, f)

    for r_key in results:
        gts = results[r_key]['gts']
        case_names = results[r_key]['case_names']
        break
    # About dims: preds_ensenble and gts are of shape C * T, C = case number, T = target number
    preds_ensenble = ensemble(ckpts_root, results, gts)
    return case_names, preds_ensenble, gts


if __name__ == '__main__':
    database = '/media/wfk/Tai/WFK_Data/RJ-PD/PDCADxFoundationRelease/train_data/train_lesion_crops'
    ckpts_root = '/home/wfk/BaiduYun/selected_ckpts'
    test_project_RJPD(database, ckpts_root)


