'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import os
import numpy as np
import pickle

from models import *
from utils import progress_bar

import logging
def setup_logger(name, log_file='app.log', level=logging.INFO):
    """return a logger"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger('RRMedica')
logger.info('Start')

lr = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_batch_size = 128 #256#128# 512
test_batch_size = 128 #256#128 #512

def load_model(model_path, exp_group, model_name, resume, in_modality_N, in_slice_N, target_classes, target_class_counts, fold, return_feat_flag = True):
    '''
    Load pytorch models, considering different architecture configurations, input configurations, predicting target configurations
    exp_group: the high-level model strategy & setting for modeling
    model_name: the model name, used to select network architecture and parameter complexity
    in_modality_N, in_slice_N: the input dimensions (first and last). in_modality_N specify the number of input channels, in_slice_N specify the image slice number.
    target_classes, target_class_counts: target_classes is the class numbers for all predicting targets (type:list).
                                        target_class_counts is the list of class counts for all predicting targets (type:list of list).
    fold: the fold index for current model
    return_feat_flag: whether to return features in the forward function
    '''
    print(f"model_path={model_path}, exp_group={exp_group}, model_name={model_name}, resume={resume}, in_modality_N={in_modality_N}, ",
          f"in_slice_N={in_slice_N}, target_classes={target_classes}, target_class_counts={target_class_counts}, fold={fold}, return_feat_flag={return_feat_flag}")
    if exp_group == 'Aux':
        if 'VGG' in model_name:
            net = VGG_aux(model_name, in_modality_N, in_slice_N, target_classes, return_feat_flag)
        elif 'resnet' in model_name:
            net = ResNet_aux(model_name, in_modality_N, in_slice_N, target_classes, return_feat_flag)
        elif 'densenet' in model_name:
            net = DenseNet_aux(model_name, in_modality_N, in_slice_N, target_classes, return_feat_flag)
        elif 'shufflenet' in model_name:
            net = ShuffleNetV2_aux(model_name, in_modality_N, in_slice_N, target_classes, return_feat_flag)

    net = net.to(device)
    if resume:
        if os.path.exists(model_path):
            print('==> Resuming from checkpoint.. ', model_path)
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
    if target_class_counts is not None:
        target_class_weights = [torch.FloatTensor([sum(l)/e for e in l]).to(device) for l in target_class_counts]
        criterions = [nn.CrossEntropyLoss(target_class_weights[i]) for i in range(len(target_classes))]
    else:
        criterions = [nn.CrossEntropyLoss() for i in range(len(target_classes))]
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return net, criterions, optimizer, scheduler


# Training
def train(epoch, net, criterions, optimizer, trainloader, target_classes, result, stats_np, losses_np):
    '''
    Train one epoch, and store and training losses & accuracies for each predicting target in losses_np & stats_np respectively
    :param epoch: 
    :param net: 
    :param criterions: 
    :param optimizer: 
    :param trainloader: 
    :param target_classes: 
    :param stats_np: 
    :param losses_np: 
    :return: 
    '''
    print('\nEpoch: %d' % epoch)
    net.train()
    case_names = []
    train_loss = 0
    loss_weights = [1] * len(target_classes)
    corrects = [0] * len(target_classes)
    losses = [0] * len(target_classes)
    total = 0
    for batch_idx, (batch_case_names, inputs, targets_labels) in enumerate(trainloader):
        inputs, targets_labels = inputs.to(device), targets_labels.to(device)
        B, TargetNum = targets_labels.size()
        optimizer.zero_grad()
        features_all, outputs = net(inputs)
        loss_sum = 0
        print_str = ' Acc: '
        total += B
        case_names.extend(batch_case_names)
        for i in range(TargetNum):
            cur_loss = criterions[i](outputs[i], targets_labels[:, i])
            loss_sum += cur_loss * loss_weights[i]
            losses[i] += cur_loss.item()
            predicted = outputs[i].argmax(1)
            corrects[i] += predicted.eq(targets_labels[:, i]).sum().item()
            print_str += ' %.1f%% ' % (100. * corrects[i] / total)
        loss_sum.backward()
        train_loss += loss_sum.item()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader), print_str),
    # batch ends, now store the losses for each signal, accuracy for each signal
    if result is not None:
        result['case_names'] = case_names
    accs = [round(corrects[i] / total, 3) for i in range(len(target_classes))]
    losses = [round(losses[i] / total, 4) for i in range(len(target_classes))]
    stats_np[epoch] = accs
    losses_np[epoch] = losses

def val(epoch, net, criterions, testloader, target_classes, result, stats_np, losses_np, ckpt_dir, exp_group, model_name, fold, try_save, training_records=None, val_tag = ''):
    net.eval()
    test_loss = 0
    corrects = [0] * len(target_classes)
    losses = [0] * len(target_classes)
    corrects_weighted = [0] * len(target_classes)
    total = 0
    total_weighted = 0 # [0] * 12
    weights = [0.2] + [1] * 8 + [0.2, 0.2, 5] 
    weights = [1] * len(target_classes)
    case_names = []
    labels_record = []
    preds_record = []
    with torch.no_grad():
        for batch_idx, (batch_case_names, inputs, targets_labels) in enumerate(testloader):
            inputs, targets_labels = inputs.to(device), targets_labels.to(device)
            B, TargetNum = targets_labels.size()
            # optimizer.zero_grad()
            features_all, outputs = net(inputs)
            loss_sum = 0
            print_str = ' Acc: '
            total += B
            case_names.extend(batch_case_names)
            labels_record.append(targets_labels.cpu().numpy())
            preds_record.append(targets_labels.cpu().numpy())
            for i in range(TargetNum):
                cur_loss = criterions[i](outputs[i], targets_labels[:, i])
                loss_sum += cur_loss
                losses[i] += cur_loss.item()
                predicted = outputs[i].argmax(1)
                corrects[i] += predicted.eq(targets_labels[:, i]).sum().item()
                corrects_weighted[i] += predicted.eq(targets_labels[:, i]).sum().item() * target_classes[i] * weights[i]
                preds_record[-1][:, i] = predicted.cpu().numpy()
                total_weighted += B * target_classes[i] * weights[i]
                print_str += ' %.1f%% ' % (100. * corrects[i] / total)
            test_loss += loss_sum.item()
            # optimizer.step()
            progress_bar(batch_idx, len(testloader), print_str)

    labels_record = np.concatenate(labels_record)
    preds_record = np.concatenate(preds_record)

    print_str = f'Epoch: {epoch} {val_tag} Acc: '
    accs = []
    for i in range(TargetNum):
        print_str += ' %.1f%% ' % (100. * corrects[i] / total)
        accs.append(round(corrects[i] / total, 3))
    logger.info(f"{print_str}")

    accs = [round(corrects[i] / total, 3) for i in range(len(target_classes))]
    losses = [round(losses[i] / total, 4) for i in range(len(target_classes))]
    if result is not None: # record the final testing of cur fold
        result['case_names'] = case_names
        result['gts'] = labels_record
        result['preds'] = preds_record

    if stats_np is not None and losses_np is not None:
        stats_np[epoch] = accs
        losses_np[epoch] = losses

    # Save checkpoint.
    acc = sum([corrects_weighted[i] for i in range(len(corrects_weighted))]) / total_weighted
    acc = acc * (1 - max(0, 120 - epoch)/120 * 0.1) # punish small epoch ( epoch<120 ) savings, may not be stable
    acc = round(acc, 4)

    if try_save and training_records is not None:
        save_new = False
        if len(training_records['best_5']) < 5:
            save_new = True
        else:
            k_v_pairs = sorted([(k,training_records['best_5'][k]) for k in training_records['best_5']], key=lambda x: x[1])
            k_v_smallest = k_v_pairs[0]
            if k_v_smallest[1] < acc:
                save_new = True
                del training_records['best_5'][k_v_smallest[0]]
                os.remove(f'{ckpt_dir}/{exp_group}_{model_name}_F{fold}_E{k_v_smallest[0]}.pth')
        if save_new:
            training_records['best_5'][epoch] = acc
            k_v_pairs = sorted([(k, training_records['best_5'][k]) for k in training_records['best_5']],
                               key=lambda x: x[1])
            print('Saving.. in training_records...', k_v_pairs)
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'training_records':training_records
            }
            torch.save(state, f'{ckpt_dir}/{exp_group}_{model_name}_F{fold}_E{epoch}.pth')

    if try_save and acc > training_records['best_acc']:
        training_records['best_epoch'] = epoch
        training_records['best_acc'] = acc
        print('Saving.. as the best')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'training_records':training_records
        }
        torch.save(state, f'{ckpt_dir}/{exp_group}_{model_name}_F{fold}.pth')

    if try_save and training_records['Last_Epoch'] == epoch:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'training_records':training_records
        }
        torch.save(state, f'{ckpt_dir}/{exp_group}_{model_name}_F{fold}_last.pth')

    return losses, accs

def predictions_deduplicate(names, targets_labels_all, predicted_labels_all):
    unique_names = sorted(list(set(names)))
    name_gts = {n: [] for n in unique_names}
    name_scores = {n: [] for n in unique_names}
    for n, targets_labels, predicted_labels in zip(names, targets_labels_all, predicted_labels_all):
        name_gts[n].append(targets_labels)
        name_scores[n].append(predicted_labels)

    L = len(unique_names)
    label_N = targets_labels_all.shape[1]
    M = targets_labels_all.max() + 1 # max label is M-1, not M; counter vector should be able to include M-1
    pred_count = np.zeros([L, label_N, M])
    gts = np.zeros([L, label_N])
    for i in range(L):
        name = unique_names[i]
        gts[i] = np.asarray(name_gts[name][0])
        scores = np.asarray(name_scores[name])
        for j in range(label_N):
            for m in scores[:, j]:
                pred_count[i][j][m] += 1
    preds = pred_count.argmax(axis=2)
    return unique_names, gts, preds


def save_model_train_val_evaluate_result(ckpt_dir, exp_group, model_name, fold, target_classes, stats, losses, test_case_names, test_gts, test_preds):
    model_preds = {}
    result_path = os.path.join(f'{ckpt_dir}/{exp_group}_{model_name}_All_results.pkl')
    if os.path.exists(result_path):
        with open(result_path, 'rb') as f:
            model_preds = pickle.load(f)
    result = {}
    result['exp_group'] = exp_group
    result['model_name'] = model_name
    result['fold'] = fold
    result['stats'] = stats
    result['losses'] = losses
    result['case_names'] = test_case_names
    result['gts'] = test_gts
    result['preds'] = test_preds
    model_preds[f'{exp_group}_{model_name}_F{fold}'] = result
    # check if all 5 folds are complete, compute cross-validation performance & save
    if sum([f'{exp_group}_{model_name}_F{f}' in model_preds for f in range(5)]) == 5:
        names_all = []
        gts_all = np.zeros([0, len(target_classes)])
        preds_all = np.zeros([0, len(target_classes)])
        for f in range(5):
            names_all += model_preds[f'{exp_group}_{model_name}_F{f}']['case_names']
            gts_all = np.concatenate([gts_all, model_preds[f'{exp_group}_{model_name}_F{f}']['gts']])
            preds_all = np.concatenate([preds_all, model_preds[f'{exp_group}_{model_name}_F{f}']['preds']])
        result = {}
        result['exp_group'] = exp_group
        result['model_name'] = model_name
        result['fold'] = 'all'
        result['case_names'] = names_all
        result['gts'] = gts_all
        result['preds'] = preds_all
        model_preds[f'{exp_group}_{model_name}_FAll'] = result
    pickle.dump(model_preds, open(result_path, 'wb'))

def transform_stats_to_cvs(save_dir, stats_all, losses_all):
    for exp_key in stats_all:
        csv_content = "Experiment results\n-----  Below are the accuracies:  ------\n"
        stats = stats_all[exp_key]
        for fold_key in stats:
            csv_content = csv_content + f"  Fold: {fold_key}\n"
            stat_nps = []
            for split_key in stats[fold_key]:
                csv_content = csv_content + f"    Split: {split_key}\n"
                stat_np = stats[fold_key][split_key]
                stat_nps.append(stat_np)
            stat_nps = np.concatenate(stat_nps, axis = 1)
            stat_np = np.round(stat_nps, 3).tolist()
            for row in stat_np:
                row = [f'{s}' for s in row]
                row_str = ','.join(row)
                csv_content = csv_content + f"{row_str}\n"
        save_file = os.path.join(save_dir, "inference_acc.csv")
        with open(save_file, 'w') as f:
            f.write(csv_content)

        csv_content = f"Experiment results\n---  Below are the losses: ------\n"
        losses = losses_all[exp_key]
        for fold_key in losses:
            csv_content = csv_content + f"  Fold: {fold_key}\n"
            stat_nps = []
            for split_key in losses[fold_key]:
                csv_content = csv_content + f"    Split: {split_key}\n"
                stat_np = losses[fold_key][split_key]
                stat_nps.append(stat_np)
            stat_nps = np.concatenate(stat_nps, axis=1)
            stat_np = np.round(stat_nps, 4).tolist()
            for row in stat_np:
                row = [f'{s}' for s in row]
                row_str = ','.join(row)
                csv_content = csv_content + f"{row_str}\n"
        save_file = os.path.join(save_dir, "inference_loss.csv")
        with open(save_file, 'w') as f:
            f.write(csv_content)

if __name__ == '__main__':
    print('common.py')


