import sys
import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
import utils
from contextlib import contextmanager
import wandb
from datetime import datetime
import socket
# from ipynb.fs.full.data_prep import get_berry_wise_df

from models import (
    PredictionModel, LoraOptions,
    TimeHead, GenotypeHead, YieldHead, FungicideHead, RotHead,
    print_trainable_parameters
)
from data.dataset import (
    CranberryPatchDataset, CranberryBerryWiseDataset,
    DefaultSplitDescriptor, CoordSplitDescriptor, GenotypeSplitDescriptor, TrackSplitDescriptor,
)

def train(model, dataset, config, save_dir=None, seed=None, use_wandb=True, test_id=None):
    now = datetime.now()
    # results in something like "facebook_dinov2-base_time_genotype_2_epochs_pw.pth"
    name = (
        f"{config['backbone_name'].replace('/', '_')}_"
        f"{'_'.join([head.shorthand for head in model.prediction_heads])}"
        f"{'_lora' if model.lora_config is not None else ''}_"
        f"{config['epochs']}_epochs_"
        f"{'pw' if type(dataset) is CranberryPatchDataset else 'bw'}_"
        f"{now.hour}h{now.minute}m.pth"
    )

    if test_id is not None:
        name = f'{test_id}.pth'

    is_berry_wise = type(dataset) is CranberryBerryWiseDataset

    if save_dir is not None:
        config['model_location'] = save_dir

    if save_dir is not None:
        # TODO: Warn if this happens, but just modify the filename and continue
        assert not os.path.isfile(os.path.join(save_dir, name)), "A model with these specs has already been trained, it would be overwritten."
        assert os.path.isdir(save_dir), "The specified save directory does not exist. You must create it manually."

    # --- Prepare data --- | TODO: add this part to config, don't hardcode
    data_splits = config['dataset_split_descriptor'].split(dataset)
    dataset_split_descriptor = config['dataset_split_descriptor']
    config['dataset_split_descriptor'] = str(config['dataset_split_descriptor'])

    train_idxs = data_splits['train_idxs']
    val_idxs = data_splits['test_idxs']
    train_dataset = data_splits['train_dataset']
    val_dataset = data_splits['test_dataset']
    
    if seed is None:
        gen1, gen2 = None, None
    else:
        gen1 = torch.Generator().manual_seed(seed)
        gen2 = torch.Generator().manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, generator=gen1, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, generator=gen2, num_workers=4)

    processor = AutoImageProcessor.from_pretrained(
        config['backbone_name'],
        do_rescale=False,
        do_resize=False,
        use_fast=False
    )

    # --- Train model ---
    assert len(model.prediction_heads) != 0, "This model configuration only outputs latent tensors, and can't be trained"
    device = next(model.parameters()).device
    print(f'[{datetime.now().strftime("%m-%d-%Y %X")}] | Training started on {device}')

    if use_wandb:
        wandb.init(
            project=f'breeders_aeye_predictors',
            mode='online',
            name=name,
            config=config
        )
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    criterions = dict()
    for head in model.prediction_heads:
        criterions[head.name] = head.loss()

    # -- Training loop --
    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        
        # - Training -
        model.train()
        epoch_train_loss = 0.0

        for patch, labels in tqdm(train_dataloader):
            if processor is not None:
                inputs = processor(patch, return_tensors='pt').to(device)
            else:
                inputs = {'pixel_values': patch.to(device)}

            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = 0.0

            if 'time' in outputs:
                loss += criterions['time'](outputs['time'].squeeze(1), labels['encoded_date']) * model.time_weight

            if 'genotype' in outputs:
                geno_pred = outputs['genotype'].squeeze(1)
                loss += criterions['genotype'](geno_pred, labels['encoded_genotype']) / geno_pred.shape[0] * model.genotype_weight

            if 'fungicide' in outputs:
                fungicide_pred = outputs['fungicide']
                loss += criterions['fungicide'](fungicide_pred, labels['encoded_fungicide']) * model.fungicide_weight

            if 'rot' in outputs:
                rot_pred = outputs['rot']
                loss += criterions['rot'](rot_pred, labels['encoded_rot']) * model.rot_weight

            if 'yield' in outputs:
                raise NotImplementedError

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # - Validation -
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for patch, labels in tqdm(val_dataloader):
                if processor is not None:
                    inputs = processor(patch, return_tensors='pt').to(device)
                else:
                    inputs = {'pixel_values': patch.to(device)}
                
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = model(inputs)
                loss = 0.0

                if 'time' in outputs:
                    loss += criterions['time'](outputs['time'].squeeze(1), labels['encoded_date']) * model.time_weight

                if 'genotype' in outputs:
                    geno_pred = outputs['genotype'].squeeze(1)
                    loss += criterions['genotype'](geno_pred, labels['encoded_genotype']) / geno_pred.shape[0] * model.genotype_weight
                
                if 'fungicide' in outputs:
                    fungicide_pred = outputs['fungicide']
                    loss += criterions['fungicide'](fungicide_pred, labels['encoded_fungicide']) * model.fungicide_weight

                if 'rot' in outputs:
                    rot_pred = outputs['rot']
                    loss += criterions['rot'](rot_pred, labels['encoded_rot']) * model.rot_weight

                if 'yield' in outputs:
                    raise NotImplementedError

                epoch_val_loss += loss.item()
                
            avg_val_loss = epoch_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)

        print(f"\tTrain loss: {avg_train_loss:.4f}")
        print(f"\tVal loss: {avg_val_loss:.4f}")

        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            })

    print(f'[{datetime.now().strftime("%m-%d-%Y %X")}] | Training complete')

    evaluate_model(model, processor, val_dataloader, use_wandb=use_wandb)

    if save_dir is not None:
        model.save(os.path.join(save_dir, name), dataset_split_descriptor=dataset_split_descriptor)
    else:
        print(f"NOT saving model {name}")

    # return validation dataset so that it can be used for evaluation or other inference tasks
    return val_dataset

@torch.no_grad()
def evaluate_model(model, processor, dataloader, use_wandb=False):
    '''
    evaluate error in terms of average days off from ground truth
    '''
    TOTAL_NUM_DAYS = dataloader.dataset.dataset.num_dates
    
    model.eval()
    device = next(model.parameters()).device
    print(f'Evaluating on {device}')

    days_error = torch.Tensor().to(device)
    correct_genotype_guesses = 0
    genotype_guesses = 0

    correct_fungicide_guesses = 0
    fungicide_guesses = 0

    correct_rot_guesses = 0
    rot_guesses = 0

    for patch, labels in tqdm(dataloader):
        if processor is not None:
            inputs = processor(patch, return_tensors='pt').to(device)
        else:
            inputs = {'pixel_values': patch.to(device)}
        labels = {k: v.to(device) for k, v in labels.items()}
        
        outputs = model(inputs)

        if 'time' in outputs:
            time_pred = outputs['time'].squeeze(1)
            days_error = torch.cat((days_error, torch.abs(time_pred-labels['encoded_date'])), 0)
        
        if 'genotype' in outputs:
            genotype_pred = outputs['genotype']
            genotype_guesses += genotype_pred.shape[0]
            correct_genotype_guesses += torch.sum(torch.argmax(genotype_pred, dim=1) == torch.argmax(labels['encoded_genotype'], dim=1)).item()

        if 'yield' in outputs:
            raise NotImplementedError

        if 'fungicide' in outputs:
            fungicide_pred = outputs['fungicide']
            fungicide_guesses += fungicide_pred.shape[0]
            correct_fungicide_guesses += torch.sum((fungicide_pred > 0).float() == labels['encoded_fungicide']).item()

        if 'rot' in outputs:
            rot_pred = outputs['rot']
            rot_guesses += rot_pred.shape[0]
            correct_rot_guesses += torch.sum((rot_pred > 0).float() == labels['encoded_rot']).item()

    days_error *= TOTAL_NUM_DAYS

    mean_err = days_error.mean().item()
    std_dev = days_error.std().item()
    if 'time' in outputs:
        print('Time prediction:')
        print(f'\tMean error: {round(mean_err, 3)} days')
        print(f'\tStd. deviation: {round(std_dev, 3)} days')

    if genotype_guesses > 0:
        print('Genotype prediction:')
        genotype_guess_rate = round(correct_genotype_guesses/genotype_guesses*100, 3)
        print(f'\tCorrect guess rate: {genotype_guess_rate}%')
        if genotype_guess_rate < 0.2:
            print('\tThis is a very low genotype guess rate. This is expected if using a genotype split, but unexpected otherwise')

    if fungicide_guesses > 0:
        print('Fungicide prediction:')
        print(f'\t{round(correct_fungicide_guesses/fungicide_guesses*100, 3)}%')

    if rot_guesses > 0:
        print('Rot prediction:')
        print(f'\t{round(correct_rot_guesses/rot_guesses*100, 3)}%')

    if use_wandb:
        perf_log = dict()
        if 'time' in outputs:
            perf_log['time_perf'] = str((round(mean_err, 3), round(std_dev, 3)))
        if 'genotype' in outputs:
            perf_log['genotype corr%'] = str(round(correct_genotype_guesses/genotype_guesses, 3)*100)
        if 'fungicide' in outputs:
            perf_log['fungicide corr%'] = str(round(correct_fungicide_guesses/fungicide_guesses, 3)*100)
        if 'rot' in outputs:
            perf_log['rot corr%'] = str(round(correct_rot_guesses/rot_guesses, 3)*100)
        wandb.log(perf_log)

        wandb.finish()

if __name__ == "__main__":
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead(size=3), TimeHead(), RotHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')
    
    tlc_patch_df = torch.load('prepped_data/tlc_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(tlc_patch_df, dataset_locations)

    # berry_wise_df = get_berry_wise_df()
    # dataset = CranberryBerryWiseDataset(berry_wise_df, dataset_locations)

    dataset_split_descriptor = GenotypeSplitDescriptor(
        train_genotypes=['cnj05-64-9', 'cnj05-73-39', 'cnj05-80-2', 'cnj06-22-10', 'cnj06-3-1'],
        test_genotypes=['cnj12-30-24', 'cnj14-31-142', 'haines']
    )

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 1,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir=None, use_wandb=False)
    # # train(model, dataset, config, save_dir='./checkpoints_bw', use_wandb=True)

    # --- evaluate an old model ---
    # model = PredictionModel(
    #     backbone_name='facebook/dinov2-with-registers-base',
    #     prediction_heads=[GenotypeHead()],
    #     lora_options=None
    # ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # processor = AutoImageProcessor.from_pretrained(
    #     'facebook/dinov2-with-registers-base',
    #     do_rescale=False,
    #     do_resize=False,
    #     use_fast=False
    # )

    # # model.load('./checkpoints_bw/facebook_dinov2-with-registers-base_geno_8_epochs_bw_5h50m.pth')
    # model.load('./old/old_checkpoints/geno_drop_tests/facebook_dinov2_base_genotype_8_epochs_23h39m.pth')

    # dataset_locations = utils.load_toml('./dataset_locations.toml')
    # berry_wise_df = get_berry_wise_df()
    # dataset = CranberryBerryWiseDataset(berry_wise_df, dataset_locations)
    # # bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    # # dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    # splits = DefaultSplitDescriptor().split(dataset)
    # # splits = model.dataset_split_descriptor.split(dataset)
    # val_dataloader = DataLoader(splits['test_dataset'], batch_size=32, shuffle=False, num_workers=4)

    # evaluate_model(model, processor, val_dataloader)