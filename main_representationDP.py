"""
Created on Feb 5, 2026.
main_representationDP.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms, models
import numpy as np
from torch import nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
from sklearn import metrics
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import expit  # sigmoid


from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_representationDP import Training
from Prediction_representationDP import Prediction
from data.data_provider import vindr_data_loader_2D, chexpert_data_loader_2D, mimic_data_loader_2D, UKA_data_loader_2D, cxr14_data_loader_2D, padchest_data_loader_2D

import warnings
warnings.filterwarnings('ignore')
from huggingface_hub import login




def main_train_central_2D(global_config_path="/PATH/dp_dinov3/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', dataset_name='vindr', dino=True, image_size=224, batch_size=30, lr=1e-5):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/PATH/dp_dinov3/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        train_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    # Changeable network parameters
    if dino:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",  # dinov3
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))
    else:
        model = AutoModel.from_pretrained(
            "facebook/convnext-small-224", # ImageNet-1k
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))

    loss_function = BCEWithLogitsLoss

    model_info = params['Network']
    model_info['lr'] = lr
    model_info['batch_size'] = batch_size
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                 weight_decay=float(params['Network']['weight_decay']),
                                 amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names)

    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=params['Network']['num_epochs'])



def main_train_DP_2D(global_config_path="/PATH/dp_dinov3/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', dataset_name='vindr', dino=True, image_size=224, batch_size=30, lr=1e-5):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/PATH/dp_dinov3/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        train_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment, image_size=image_size)
        valid_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    # Changeable network parameters
    if dino:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",  # dinov3
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))
    else:
        model = AutoModel.from_pretrained(
            "facebook/convnext-small-224", # ImageNet-1k
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))
        # model = load_pretrained_timm_model(num_classes=len(weight), model_name='resnet50d', pretrained=pretrained)



    loss_function = BCEWithLogitsLoss

    model_info = params['Network']
    model_info['lr'] = lr
    model_info['batch_size'] = batch_size
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']))

    model.train()
    errors = ModuleValidator.validate(model, strict=False)
    assert len(errors) == 0
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=params['Network']['num_epochs'],
        target_epsilon=params['DP']['epsilon'],
        target_delta=float(params['DP']['delta']),
        max_grad_norm=params['DP']['max_grad_norm'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names)

    if resume == True:
        trainer.load_checkpoint_DP(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names, privacy_engine=privacy_engine)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, privacy_engine=privacy_engine)
    trainer.train_DP_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=params['Network']['num_epochs'])




def main_test_nonDP(demographics, global_config_path="/PATH/dp_dinov3/config/config.yaml",
                                                 experiment_name='central_exp_for_test', experiment_epoch_num=100,
                        dataset_name='vindr', dino=True, image_size=224, new_seed=False):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels


    # Changeable network parameters
    if dino:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",  # dinov3
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))
    else:
        model = AutoModel.from_pretrained(
            "facebook/convnext-small-224", # ImageNet-1k
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=11)


    os.makedirs(os.path.join(params['target_dir'], params['stat_log_path'], demographics), exist_ok=True)
    if new_seed:
        index_list = []
        for counter in range(1000):
            index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))
        np.save(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/bootstrapping_seeds_' + str(dataset_name) + ".npy", index_list, allow_pickle=True)


    else:
        index_list = np.load(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/bootstrapping_seeds_' + str(dataset_name) + ".npy", allow_pickle=True)

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model, epoch_num=experiment_epoch_num)
    pred_array, target_array = predictor.predict_only(test_loader)

    #########################################
    pred_array = pred_array.cpu().numpy()
    target_array = target_array.int().cpu().numpy()
    np.save(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/pred_array_'  + str(dataset_name) + '.npy', pred_array, allow_pickle=True)
    np.save(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/target_array_'  + str(dataset_name) + '.npy', target_array, allow_pickle=True)

    df = pd.DataFrame(pred_array.mean(1), columns=['probability_mean'])
    for idx in range(pred_array.shape[-1]):
        df.insert(idx + 1, 'prob_' + label_names[idx], pred_array[:, idx])
        df.insert(idx + 1, 'gt_' + label_names[idx], target_array[:, idx])
    df.to_csv(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/predictions_teston_' + str(
        dataset_name) + '.csv', sep=',', index=False)


    msg = (
        f'\n\n\n\t Epoch number: {experiment_epoch_num} below:'
    )
    with open(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)

    AUC_list = predictor.bootstrapper(pred_array, target_array, index_list, dataset_name, demographics=demographics)



def main_test_DP(index_list, demographics, global_config_path="/PATH/dp_dinov3/config/config.yaml",
                                                 experiment_name='central_exp_for_test', experiment_epoch_num=100,
                        dataset_name='vindr', dino=True, image_size=224):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels


    # Changeable network parameters
    if dino:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",  # dinov3
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))
    else:
        model = AutoModel.from_pretrained(
            "facebook/convnext-small-224", # ImageNet-1k
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=11)
    optimizer = torch.optim.NAdam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']))


    model.train()
    errors = ModuleValidator.validate(model, strict=False)
    assert len(errors) == 0
    privacy_engine = PrivacyEngine()

    model, _, _ = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer, # not important during testing; you should only put a placeholder here
        data_loader=test_loader, # not important during testing; you should only put a placeholder here
        epochs=params['Network']['num_epochs'], # not important during testing; you should only put a placeholder here
        target_epsilon=params['DP']['epsilon'], # not important during testing; you should only put a placeholder here
        target_delta=float(params['DP']['delta']), # not important during testing; you should only put a placeholder here
        max_grad_norm=params['DP']['max_grad_norm']) # not important during testing; you should only put a placeholder here

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model, epoch_num=experiment_epoch_num)
    pred_array, target_array = predictor.predict_only(test_loader)

    #########################################
    pred_array = pred_array.cpu().numpy()
    target_array = target_array.int().cpu().numpy()

    os.makedirs(os.path.join(params['target_dir'], params['stat_log_path'], demographics), exist_ok=True)

    np.save(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/pred_array_'  + str(dataset_name) + '.npy', pred_array, allow_pickle=True)
    np.save(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/target_array_'  + str(dataset_name) + '.npy', target_array, allow_pickle=True)

    df = pd.DataFrame(pred_array.mean(1), columns=['probability_mean'])
    for idx in range(pred_array.shape[-1]):
        df.insert(idx + 1, 'prob_' + label_names[idx], pred_array[:, idx])
        df.insert(idx + 1, 'gt_' + label_names[idx], target_array[:, idx])
    df.to_csv(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/predictions_teston_' + str(
        dataset_name) + '.csv', sep=',', index=False)


    msg = (
        f'\n\n\n\t Epoch number: {experiment_epoch_num} below:'
    )
    with open(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)

    AUC_list = predictor.bootstrapper(pred_array, target_array, index_list, dataset_name, demographics=demographics)




def main_pvalue_out_of_bootstrap(index_list, global_config_path="/PATH/dp_dinov3/config/config.yaml",
                                                 experiment_name1='central_exp_for_test', experiment_name2='central_exp_for_test',
                                                 dataset_name='vindr', image_size=224, demographics='Full_set'):
    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']


    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False, image_size=image_size)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Initialize prediction 1
    predictor1 = Prediction(cfg_path1, label_names)

    pred_array1 = np.load(os.path.join(params1['target_dir'], params1['stat_log_path'], demographics) + '/pred_array.npy', allow_pickle=True)
    target_array1 = np.load(os.path.join(params1['target_dir'], params1['stat_log_path'], demographics) + '/target_array.npy', allow_pickle=True)

    AUC_list1 = predictor1.bootstrapper(pred_array1, target_array1, index_list, dataset_name)

    #########################################

    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2, label_names)

    pred_array2 = np.load(os.path.join(params2['target_dir'], params2['stat_log_path'], demographics) + '/pred_array.npy', allow_pickle=True)
    target_array2 = np.load(os.path.join(params2['target_dir'], params2['stat_log_path'], demographics) + '/target_array.npy', allow_pickle=True)

    AUC_list2 = predictor2.bootstrapper(pred_array2, target_array2, index_list, dataset_name)


    print('individual labels p-values:\n')
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)

        if ratio1 <= 0.05:
            print(f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2')
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)

            if ratio2 <= 0.05:
                print(f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1')
            else:
                print(f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label')

    print('\nAvg AUC of labels p-values:\n')
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)

    if ratio1 <= 0.05:
        print(f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average')
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)

        if ratio2 <= 0.05:
            print(f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average')
        else:
            print(f'\tp-value: {ratio1}; models NOT significantly different on average for all labels')


    msg = f'\n\nindividual labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)

        if ratio1 <= 0.05:
            msg = f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2'
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)

            if ratio2 <= 0.05:
                msg = f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1'
            else:
                msg = f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
            f.write(msg)


    msg = f'\n\nAvg AUC of labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)

    if ratio1 <= 0.05:
        msg = f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average'
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)

        if ratio2 <= 0.05:
            msg = f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average'
        else:
            msg = f'\tp-value: {ratio1}; models NOT significantly different on average for all labels'

    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)




def main_embedding_extractor_nonDP(global_config_path="/PATH/dp_dinov3/config/config.yaml",
                                                 experiment_name='central_exp_for_test', experiment_epoch_num=100,
                        dataset_name='padchest', new_seed=False, dino=False, image_size=224, init_mode=False, save_tag="imgnetTrained"):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=False, image_size=image_size)
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels


    # Changeable network parameters
    if dino:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",  # dinov3
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))
    else:
        model = AutoModel.from_pretrained(
            "facebook/convnext-small-224", # ImageNet-1k
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))




    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=11)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=11)


    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)

    if not init_mode:
        predictor.setup_model(model=model, epoch_num=experiment_epoch_num)

    if init_mode:
        predictor.setup_model(model=model, epoch_num=experiment_epoch_num, init_mode=True)
    else:
        predictor.setup_model(model=model, epoch_num=experiment_epoch_num, init_mode=False)


    train_embeddings_cache, train_labels_cache = predictor.predict_embeddings(train_loader)

    embeddings_cache, labels_cache = predictor.predict_embeddings(test_loader)

    #########################################
    train_Z = train_embeddings_cache.numpy()  # (N, 768)
    train_Y = train_labels_cache.numpy().astype(np.int32)  # (N, 5)

    Z = embeddings_cache.numpy()  # (N, 768)
    Y = labels_cache.numpy().astype(np.int32)  # (N, 5)

    out_dir = os.path.join(params['target_dir'], params['stat_log_path'], "rep_geometry")
    os.makedirs(out_dir, exist_ok=True)

    train_Z_path = os.path.join(out_dir, f"train_Z_{save_tag}_{dataset_name}.npy")
    train_Y_path = os.path.join(out_dir, f"train_Y_{save_tag}_{dataset_name}.npy")
    np.save(train_Z_path, train_Z, allow_pickle=False)
    np.save(train_Y_path, train_Y, allow_pickle=False)

    Z_path = os.path.join(out_dir, f"Z_{save_tag}_{dataset_name}.npy")
    Y_path = os.path.join(out_dir, f"Y_{save_tag}_{dataset_name}.npy")
    np.save(Z_path, Z, allow_pickle=False)
    np.save(Y_path, Y, allow_pickle=False)





def main_embedding_extractor_DP(global_config_path="/PATH/dp_dinov3/config/config.yaml",
                                                 experiment_name='central_exp_for_test', experiment_epoch_num=100,
                        dataset_name='padchest', dino=False, image_size=224, init_mode=False, save_tag="imgnetTrained"):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    elif dataset_name == 'padchest':
        train_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='train', augment=False, image_size=image_size)
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels


    # Changeable network parameters
    if dino:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",  # dinov3
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))
    else:
        model = AutoModel.from_pretrained(
            "facebook/convnext-small-224", # ImageNet-1k
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))




    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=11)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=11)


    optimizer = torch.optim.NAdam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']))


    model.train()
    errors = ModuleValidator.validate(model, strict=False)
    assert len(errors) == 0
    privacy_engine = PrivacyEngine()

    model, _, _ = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer, # not important during testing; you should only put a placeholder here
        data_loader=test_loader, # not important during testing; you should only put a placeholder here
        epochs=params['Network']['num_epochs'], # not important during testing; you should only put a placeholder here
        target_epsilon=params['DP']['epsilon'], # not important during testing; you should only put a placeholder here
        target_delta=float(params['DP']['delta']), # not important during testing; you should only put a placeholder here
        max_grad_norm=params['DP']['max_grad_norm']) # not important during testing; you should only put a placeholder here

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)

    if init_mode:
        predictor.setup_model_DP(model=model, privacy_engine=privacy_engine, epoch_num=experiment_epoch_num, init_mode=True)
    else:
        predictor.setup_model_DP(model=model, privacy_engine=privacy_engine, epoch_num=experiment_epoch_num, init_mode=False)


    train_embeddings_cache, train_labels_cache = predictor.predict_embeddings(train_loader)

    embeddings_cache, labels_cache = predictor.predict_embeddings(test_loader)

    #########################################
    train_Z = train_embeddings_cache.numpy()              # (N, 768)
    train_Y = train_labels_cache.numpy().astype(np.int32) # (N, 5)

    Z = embeddings_cache.numpy()              # (N, 768)
    Y = labels_cache.numpy().astype(np.int32) # (N, 5)

    out_dir = os.path.join(params['target_dir'], params['stat_log_path'], "rep_geometry")
    os.makedirs(out_dir, exist_ok=True)

    train_Z_path = os.path.join(out_dir, f"train_Z_{save_tag}_{dataset_name}.npy")
    train_Y_path = os.path.join(out_dir, f"train_Y_{save_tag}_{dataset_name}.npy")
    np.save(train_Z_path, train_Z, allow_pickle=False)
    np.save(train_Y_path, train_Y, allow_pickle=False)


    Z_path = os.path.join(out_dir, f"Z_{save_tag}_{dataset_name}.npy")
    Y_path = os.path.join(out_dir, f"Y_{save_tag}_{dataset_name}.npy")
    np.save(Z_path, Z, allow_pickle=False)
    np.save(Y_path, Y, allow_pickle=False)








def main_geometry_steps_1_2_3_from_saved_embeddings(
        global_config_path="/PATH/dp_dinov3/config/config.yaml",
        experiment_name='central_exp_for_test',
        dataset_name='padchest',
        trained_tag="imgnetTrained",
        init_tag="imgnetInit",
        seeds_filename="bootstrapping_seeds_padchest.npy",
        verbose=True,
):
    """
    Loads:
      Z  from  stat_logs/rep_geometry/Z_<trained_tag>_<dataset>.npy
      Z0 from  stat_logs/rep_geometry/Z_<init_tag>_<dataset>.npy
      bootstrap indices from stat_logs/rep_geometry/<seeds_filename>

    Computes:
      Step 2) Drift Δ (bootstrapped mean with 95% CI)
      Step 3) Effective dimension d_eff of Z (bootstrapped mean with 95% CI)

    Writes:
      stat_logs/rep_geometry/geometry_<trained_tag>_vs_<init_tag>_<dataset>.txt
      stat_logs/rep_geometry/geometry_<trained_tag>_vs_<init_tag>_<dataset>.csv
    """

    params = open_experiment(experiment_name, global_config_path)

    rep_dir = os.path.join(params['target_dir'], params['stat_log_path'], "rep_geometry")
    os.makedirs(rep_dir, exist_ok=True)

    Z_path = os.path.join(rep_dir, f"Z_{trained_tag}_{dataset_name}.npy")
    Z0_path = os.path.join(rep_dir, f"Z_{init_tag}_{dataset_name}.npy")
    seeds_path = os.path.join(rep_dir, seeds_filename)

    if not os.path.exists(Z_path):
        raise FileNotFoundError(f"Missing trained embeddings: {Z_path}")
    if not os.path.exists(Z0_path):
        raise FileNotFoundError(f"Missing init embeddings: {Z0_path}")
    if not os.path.exists(seeds_path):
        raise FileNotFoundError(f"Missing bootstrap seeds: {seeds_path}")

    Z = np.load(Z_path)  # (N, D)
    Z0 = np.load(Z0_path)  # (N, D)

    if Z.shape != Z0.shape:
        raise ValueError(f"Shape mismatch: Z {Z.shape} vs Z0 {Z0.shape}. "
                         f"Ensure both were extracted on the same test loader with shuffle=False.")

    N, D = Z.shape

    # seeds: either (B, N) array OR list-of-arrays saved with allow_pickle
    boot_idx = np.load(seeds_path, allow_pickle=True)
    boot_idx = np.array(boot_idx)

    if boot_idx.ndim != 2:
        raise ValueError(f"Bootstrap seeds must be 2D (B, N). Got shape: {boot_idx.shape}")

    if boot_idx.shape[1] != N:
        raise ValueError(f"Bootstrap index length {boot_idx.shape[1]} != N={N}. "
                         f"Use seeds generated for this exact PadChest test set ordering.")

    B = boot_idx.shape[0]

    # Step 2: Drift Δ
    # per-image squared L2 distance
    d_i = np.sum((Z - Z0) ** 2, axis=1)  # (N,)

    drift_bs = np.empty((B,), dtype=np.float64)
    for b in range(B):
        drift_bs[b] = d_i[boot_idx[b]].mean()

    drift_mean = float(drift_bs.mean())
    drift_std = float(drift_bs.std())
    drift_lo, drift_hi = np.percentile(drift_bs, [2.5, 97.5])

    # Step 3: Effective dimension d_eff
    def effective_dimension(Zmat: np.ndarray) -> float:
        Zc = Zmat - Zmat.mean(axis=0, keepdims=True)
        Sigma = (Zc.T @ Zc) / float(Zc.shape[0])  # (D, D)
        tr = float(np.trace(Sigma))
        fro2 = float(np.sum(Sigma * Sigma))  # ||Sigma||_F^2 = tr(Sigma^2)
        return float((tr * tr) / (fro2 + 1e-12))

    deff_bs = np.empty((B,), dtype=np.float64)
    for b in range(B):
        deff_bs[b] = effective_dimension(Z[boot_idx[b]])

    deff_mean = float(deff_bs.mean())
    deff_std = float(deff_bs.std())
    deff_lo, deff_hi = np.percentile(deff_bs, [2.5, 97.5])

    out_txt = os.path.join(rep_dir, f"geometry_{trained_tag}_vs_{init_tag}_{dataset_name}.txt")
    out_csv = os.path.join(rep_dir, f"geometry_{trained_tag}_vs_{init_tag}_{dataset_name}.csv")

    msg = (
        f"experiment: {experiment_name}\n"
        f"dataset: {dataset_name}\n"
        f"trained_tag: {trained_tag}\n"
        f"init_tag: {init_tag}\n"
        f"seeds_file: {seeds_filename}\n"
        f"N: {N} | D: {D} | B: {B}\n\n"
        f"Drift Δ: {drift_mean:.6f} ± {drift_std:.6f} [95% CI: {drift_lo:.6f}, {drift_hi:.6f}]\n"
        f"d_eff: {deff_mean:.6f} ± {deff_std:.6f} [95% CI: {deff_lo:.6f}, {deff_hi:.6f}]\n"
    )
    with open(out_txt, "w") as f:
        f.write(msg)

    df = pd.DataFrame([
        {"metric": "drift_mean", "value": drift_mean},
        {"metric": "drift_std", "value": drift_std},
        {"metric": "drift_ci_lo", "value": float(drift_lo)},
        {"metric": "drift_ci_hi", "value": float(drift_hi)},
        {"metric": "deff_mean", "value": deff_mean},
        {"metric": "deff_std", "value": deff_std},
        {"metric": "deff_ci_lo", "value": float(deff_lo)},
        {"metric": "deff_ci_hi", "value": float(deff_hi)},
        {"metric": "N", "value": N},
        {"metric": "D", "value": D},
        {"metric": "B", "value": B},
    ])
    df.to_csv(out_csv, index=False)

    if verbose:
        print(msg)

    return {
        "drift_mean": drift_mean,
        "drift_std": drift_std,
        "drift_ci": (float(drift_lo), float(drift_hi)),
        "deff_mean": deff_mean,
        "deff_std": deff_std,
        "deff_ci": (float(deff_lo), float(deff_hi)),
        "out_txt": out_txt,
        "out_csv": out_csv,
    }






def main_linear_probe_from_saved_embeddings(
        global_config_path="/PATH/dp_dinov3/config/config.yaml",
        dataset_name='padchest',
        trained_tag='mimicTrained',
        experiment_name='central_exp_for_test',
        C=1.0):

    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    label_names = test_dataset.chosen_labels

    index_list = np.load(
        os.path.join(params['target_dir'], params['stat_log_path'], 'rep_geometry/bootstrapping_seeds_padchest.npy'), allow_pickle=True)

    rep_dir = os.path.join(params['target_dir'], params['stat_log_path'], "rep_geometry")
    os.makedirs(rep_dir, exist_ok=True)

    train_Z_path = os.path.join(rep_dir, "train_Z_" + trained_tag + "_padchest.npy")
    train_Y_path = os.path.join(rep_dir, "train_Y_" + trained_tag + "_padchest.npy")

    test_Z_path = os.path.join(rep_dir, "Z_" + trained_tag + "_padchest.npy")
    test_Y_path = os.path.join(rep_dir, "Y_" + trained_tag + "_padchest.npy")

    test_Z = np.load(test_Z_path)  # (N, D)
    test_Y = np.load(test_Y_path)  # (N, D)

    train_Z = np.load(train_Z_path)  # (N, D)
    train_Y = np.load(train_Y_path)  # (N, D)


    # multilabel linear probe = 5 independent binary logistic regressions
    base = LogisticRegression(
        penalty="l2",
        C=C,
        solver="liblinear",   # stable for binary
        max_iter=2000
    )
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(train_Z, train_Y)

    # scores -> sigmoid -> probabilities (Nte, 5)
    scores = clf.decision_function(test_Z)              # (Nte, 5)
    pred_array = expit(scores).astype(np.float32)       # (Nte, 5)
    target_array = test_Y.astype(np.int32)              # (Nte, 5)


    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)

    #########################################

    np.save(os.path.join(rep_dir, f"probe_pred_array_{trained_tag}_{dataset_name}.npy"), pred_array, allow_pickle=False)
    np.save(os.path.join(rep_dir, f"probe_target_array_{trained_tag}_{dataset_name}.npy"), target_array, allow_pickle=False)


    df = pd.DataFrame(pred_array.mean(1), columns=['probability_mean'])
    for idx in range(pred_array.shape[-1]):
        df.insert(idx + 1, 'prob_' + label_names[idx], pred_array[:, idx])
        df.insert(idx + 1, 'gt_' + label_names[idx], target_array[:, idx])
    df.to_csv(os.path.join(rep_dir, f"probe_predictions_{trained_tag}_teston_{dataset_name}.csv"), sep=',', index=False)


    AUC_list = predictor.bootstrapper(pred_array, target_array, index_list, dataset_name)




