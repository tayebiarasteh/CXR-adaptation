"""
Created on Feb 5, 2026.
main_adaptation.py

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


from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_adaptation import Training
from Prediction_adaptation import Prediction
from data.data_provider import vindr_data_loader_2D, chexpert_data_loader_2D, mimic_data_loader_2D, UKA_data_loader_2D, cxr14_data_loader_2D, padchest_data_loader_2D
from Train_Valid_tta_lnconvnext import TTA_Adaptation

import warnings
warnings.filterwarnings('ignore')
from huggingface_hub import login




def main_train_central_2D(global_config_path="/PATH/CXR-adaptation/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', dataset_name='vindr', dino=True, image_size=224, batch_size=30, lr=1e-5):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/PATH/CXR-adaptation/config/config.yaml"

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




def main_test_baseline(demographics, global_config_path="/PATH/CXR-adaptation/config/config.yaml",
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




def _make_dataset(cfg_path, dataset_name, mode, image_size, augment, demographics=None):
    if dataset_name == "vindr":
        return vindr_data_loader_2D(cfg_path=cfg_path, mode=mode, augment=augment, image_size=image_size)
    if dataset_name == "chexpert":
        return chexpert_data_loader_2D(cfg_path=cfg_path, mode=mode, augment=augment, image_size=image_size)
    if dataset_name == "mimic":
        return mimic_data_loader_2D(cfg_path=cfg_path, mode=mode, augment=augment, image_size=image_size)
    if dataset_name == "UKA":
        return UKA_data_loader_2D(cfg_path=cfg_path, mode=mode, augment=augment, image_size=image_size)
    if dataset_name == "cxr14":
        return cxr14_data_loader_2D(cfg_path=cfg_path, mode=mode, augment=augment, image_size=image_size)
    if dataset_name == "padchest":
        return padchest_data_loader_2D(cfg_path=cfg_path, mode=mode, augment=augment, image_size=image_size)
    raise ValueError(f"Unknown dataset: {dataset_name}")




def _init_convnext(dino: bool, num_labels: int):
    if dino:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",
            use_safetensors=True
        )
    else:
        model = AutoModel.from_pretrained(
            "facebook/convnext-small-224",
            use_safetensors=True
        )
    model.head = torch.nn.Linear(in_features=768, out_features=num_labels)

    try:
        model.load_state_dict(
            torch.load('/home/soroosh/Documents/Repositories/CXR-adaptation/pretraining_convnext_small_mimic.tar')[
                'model_state_dict'])
    except:
        model.load_state_dict(
            torch.load('/home/soroosh/Documents/Repositories/CXR-adaptation/pretraining_convnext_small_mimic.pth'))

    return model


def _mimic_source_priors(cfg_path, image_size=224):
    """
    Computes per-label prevalence on MIMIC train split for 5-label setup.
    """
    ds = mimic_data_loader_2D(cfg_path=cfg_path, mode="train", augment=False, image_size=image_size)
    w = ds.pos_weight().float().cpu().numpy()  # (C,)
    priors = 1.0 / (1.0 + w)
    return priors.tolist(), ds.chosen_labels



def main_train_tta_lnconvnext_direct(
    global_config_path="/PATH/CXR-adaptation/config/config.yaml",
    experiment_name="tta_direct",
    dataset_name="padchest",
    dino=True,
    image_size=224,
    batch_size=30,

    # optimizer
    lr=1e-5,
    weight_decay=0.0,

    # TTA knobs (new method)
    eps_prior=0.002,
    dual_lr=0.05,
    w_consistency=1.0,
    w_anchor=2.0,
    steps_per_batch=1,

    # what to adapt
    adapt_ln=True,
    adapt_head=False,
):
    """
    Adapts on external TRAIN split (labels ignored in adaptation),
    saves adapted weights. Evaluation stays separate
    """
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    login(token=params["hf_login"])

    # data: adapt on train split
    adapt_ds = _make_dataset(cfg_path, dataset_name, mode="train", image_size=image_size, augment=False)
    label_names = adapt_ds.chosen_labels

    adapt_loader = torch.utils.data.DataLoader(
        dataset=adapt_ds,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        num_workers=10
    )

    # base model
    model = _init_convnext(dino=dino, num_labels=len(label_names))

    # source priors from MIMIC train
    source_priors, mimic_label_names = _mimic_source_priors(cfg_path, image_size=image_size)

    # sanity: label order must match, otherwise priors map wrongly
    if list(map(str.lower, mimic_label_names)) != list(map(str.lower, label_names)):
        print("WARNING: Label order mismatch between MIMIC and target dataset.")
        print("MIMIC labels:", mimic_label_names)
        print("Target labels:", label_names)
        print("You MUST reorder source_priors to match target label order.")

    # adapt
    tta = TTA_Adaptation(cfg_path, label_names=label_names)

    adapted_name = f"tta_{dataset_name}.pth"

    model = tta.adapt(
        model=model,
        adapt_loader=adapt_loader,
        save_name=adapted_name,
        lr=lr,
        weight_decay=weight_decay,
        steps_per_batch=steps_per_batch,
        adapt_ln=adapt_ln,
        adapt_head=adapt_head,
        source_priors=source_priors,
        eps_prior=eps_prior,
        dual_lr=dual_lr,
        w_consistency=w_consistency,
        w_anchor=w_anchor,
    )




def main_test_tta_lnconvnext_direct(
    global_config_path="/PATH/CXR-adaptation/config/config.yaml", experiment_name="central_exp_for_test",
    dataset_name="padchest", dino=True, adapt=True, image_size=224, batch_size=30):
    """
    External protocol
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    print('Loading the model and weights ...')
    login(token=params["hf_login"])

    eval_ds = _make_dataset(cfg_path, dataset_name, mode="test", image_size=image_size, augment=False)
    label_names = eval_ds.chosen_labels

    test_loader = torch.utils.data.DataLoader(
        dataset=eval_ds, batch_size=batch_size,
        pin_memory=True, drop_last=False, shuffle=False, num_workers=10
    )

    model = _init_convnext(dino=dino, num_labels=len(label_names))

    if adapt:
        adapted_name = f"tta_{dataset_name}.pth"
    else:
        adapted_name = "epoch5_trained_model.pth"

    print('\nStarting the prediction ...')
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model_adapted(model=model, save_name=adapted_name)
    os.makedirs(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name), exist_ok=True)

    # pred_array, target_array = predictor.predict_only(test_loader)
    #
    # #########################################
    # pred_array = pred_array.cpu().numpy()
    # target_array = target_array.int().cpu().numpy()
    #
    # np.save(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/pred_array_'  + str(dataset_name) + '.npy', pred_array, allow_pickle=True)
    # np.save(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/target_array_'  + str(dataset_name) + '.npy', target_array, allow_pickle=True)
    #
    # df = pd.DataFrame(pred_array.mean(1), columns=['probability_mean'])
    # for idx in range(pred_array.shape[-1]):
    #     df.insert(idx + 1, 'prob_' + label_names[idx], pred_array[:, idx])
    #     df.insert(idx + 1, 'gt_' + label_names[idx], target_array[:, idx])
    #
    # df.to_csv(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/predictions_teston_' + str(
    #     dataset_name) + '.csv', sep=',', index=False)


    valid_F1, valid_AUC, valid_accuracy, valid_specificity, valid_sensitivity, valid_precision = predictor.evaluate_2D(test_loader)

    print(
        f'\t avg AUROC: {valid_AUC.mean() * 100:.1f}% | avg accuracy: {valid_accuracy.mean() * 100:.1f}%'
        f' | avg specificity: {valid_specificity.mean() * 100:.1f}%'
        f' | avg recall (sensitivity): {valid_sensitivity.mean() * 100:.1f}% | avg F1: {valid_F1.mean() * 100:.1f}%\n')

    print('Individual AUC:')
    for idx, pathology in enumerate(label_names):
        try:
            print(f'\t{pathology}: {valid_AUC[idx] * 100:.1f}%')
        except:
            print(f'\t{pathology}: {valid_AUC * 100:.1f}%')

    print('\nIndividual accuracy:')
    for idx, pathology in enumerate(label_names):
        print(f'\t{pathology}: {valid_accuracy[idx] * 100:.1f}%')

    print('\nIndividual sensitivity:')
    for idx, pathology in enumerate(label_names):
        print(f'\t{pathology}: {valid_sensitivity[idx] * 100:.1f}%')

    print('\nIndividual specificity:')
    for idx, pathology in enumerate(label_names):
        print(f'\t{pathology}: {valid_specificity[idx] * 100:.1f}%')

    # saving the training and validation stats
    msg = f'\n\n----------------------------------------------------------------------------------------\n' \
          f'avg AUC: {valid_AUC.mean() * 100:.1f}% | avg accuracy: {valid_accuracy.mean() * 100:.1f}% ' \
          f' | avg sensitivity: {valid_sensitivity.mean() * 100:.1f}%' \
          f' | avg specificity: {valid_specificity.mean() * 100:.1f}% | avg precision: {valid_precision.mean() * 100:.1f}% | avg F1: {valid_F1.mean() * 100:.2f}%\n\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)

    msg = f'Individual AUC:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        try:
            msg = f'{pathology}: {valid_AUC[idx] * 100:.1f}% | '
        except:
            msg = f'{pathology}: {valid_AUC * 100:.1f}% | '

        with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(
                dataset_name), 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual accuracy:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {valid_accuracy[idx] * 100:.1f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(
                dataset_name), 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual sensitivity:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {valid_sensitivity[idx] * 100:.1f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(
                dataset_name), 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual specificity:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {valid_specificity[idx] * 100:.1f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(
                dataset_name), 'a') as f:
            f.write(msg)

    with open(os.path.join(params['target_dir'], params['stat_log_path'], dataset_name) + '/External_adapted_test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)










if __name__ == '__main__':
    global_config_path = "/home/soroosh/Documents/Repositories/CXR-adaptation/config/config.yaml"
    # delete_experiment(experiment_name="vindr_adapted_mimicpret_224_le5e6", global_config_path=global_config_path)


    # main_train_central_2D(global_config_path=global_config_path, valid=True, resume=False, augment=True,
    #                       experiment_name='domain_central_padchest_50percent_convsmall_dinov3_224', dataset_name='padchest',
    #                       dino=True, image_size=224, batch_size=64, lr=1e-5)


    dataset = 'cxr14'
    experiment_name = dataset + "_adapted_mimicpret_224_le1e5"

    main_train_tta_lnconvnext_direct(
        global_config_path=global_config_path,
        experiment_name= experiment_name,
        dataset_name=dataset,
        batch_size=16,
        eps_prior=0.01,
        lr=1e-5,
        w_anchor=4.0,
    )

    main_test_tta_lnconvnext_direct(
    global_config_path = global_config_path,
    experiment_name = experiment_name,
    dataset_name = dataset,
    adapt=True,
    batch_size = 32,
    )

    # main_test_tta_lnconvnext_direct(
    # global_config_path = global_config_path,
    # experiment_name = "central_mimic_convsmall_dinov3_lr1e5_224",
    # dataset_name = dataset,
    # adapt=False,
    # batch_size = 30,
    # )


    # vindr: lr=5e-6 | w_anchor=3.0 | eps_prior=0.01
    # cxr14: lr=1e-5 | w_anchor=4.0
    # padchest: lr=2e-5 | w_anchor=3.0
    # chexpert: lr=1e-5 | w_anchor=5.0
    # UKA: lr=1e-5 | w_anchor=3.0