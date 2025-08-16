import torch
import gc

from models import (
    PredictionModel, LoraOptions,
    TimeHead, GenotypeHead, YieldHead, FungicideHead,
    print_trainable_parameters
)
from data.dataset import (
    CranberryPatchDataset, CranberryBerryWiseDataset,
    DefaultSplitDescriptor, CoordSplitDescriptor, GenotypeSplitDescriptor, TrackSplitDescriptor
)
from train import train

import utils
from ipynb.fs.full.data_prep import get_berry_wise_df
import socket

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def t1():
    # berry wise time prediction with google vit backbone
    model = PredictionModel(
        backbone_name='google/vit-base-patch16-224',
        prediction_heads=[TimeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')
    dataset = CranberryBerryWiseDataset(get_berry_wise_df(), dataset_locations)
    dataset_split_descriptor = DefaultSplitDescriptor(train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_bw', use_wandb=True)

def t2():
    # berry wise time prediction with dinov2 backbone
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')
    dataset = CranberryBerryWiseDataset(get_berry_wise_df(), dataset_locations)
    dataset_split_descriptor = DefaultSplitDescriptor(train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_bw', use_wandb=True)

def t3():
    # berry wise genotype prediction with dinov2 backbone
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead(size=3)],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')
    dataset = CranberryBerryWiseDataset(get_berry_wise_df(), dataset_locations)
    dataset_split_descriptor = DefaultSplitDescriptor(train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_bw', use_wandb=True)

def t4():
    # patch-wise genotype and time predictor with dinov2 backbone.
    # dataset split by genotype
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead(), TimeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)


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
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True)

def t5():
    # patch-wise genotype predictor with dinov2 backbone.
    # dataset split by genotype
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = GenotypeSplitDescriptor(
        train_genotypes=['haines', 'cnj05-73-39', 'cnj12-30-24', 'cnj06-22-10', 'cnj06-3-1'],
        test_genotypes=['cnj05-80-2', 'cnj14-31-142', 'cnj05-64-9']
    )

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True)

def t6():
    # patch-wise control/treatment predictor with dinov2 backbone.
    # dataset split by genotype
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = GenotypeSplitDescriptor(
        train_genotypes=['haines', 'cnj05-73-39', 'cnj12-30-24', 'cnj06-22-10', 'cnj06-3-1'],
        test_genotypes=['cnj05-80-2', 'cnj14-31-142', 'cnj05-64-9']
    )

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True)

def t7():
    # berry wise control/treatment prediction with dinov2 backbone
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')
    dataset = CranberryBerryWiseDataset(get_berry_wise_df(), dataset_locations)
    dataset_split_descriptor = DefaultSplitDescriptor(train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_bw', use_wandb=True)


def t8():
    # row5
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True)

def t9():
    # row6
    model = PredictionModel(
        backbone_name='google/vit-base-patch16-224',
        prediction_heads=[TimeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True)

def t10():
    # row10
    model = PredictionModel(
        backbone_name='google/vit-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead(), FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True)

def b2():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead(), FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True)

def b2_2():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead(), FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    # dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)
    dataset_split_descriptor = DefaultSplitDescriptor(train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='b2_2')

def b2_3():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    # dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)
    dataset_split_descriptor = DefaultSplitDescriptor(train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='b2_3')

def b2_4():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    # dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)
    dataset_split_descriptor = DefaultSplitDescriptor(train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='b2_4')

def b2_5():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead(weight=4), FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    # dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)
    dataset_split_descriptor = DefaultSplitDescriptor(train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='b2_5')

def a3():
    model = PredictionModel(
        backbone_name='microsoft/swin-base-patch4-window7-224',
        prediction_heads=[TimeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='a3')

def a4():
    model = PredictionModel(
        backbone_name='google/siglip2-base-patch16-224',
        prediction_heads=[TimeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='a4')

def c1():
    model = PredictionModel(
        backbone_name='google/vit-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead(weight=4.0), FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c1-2')

def c2():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead(weight=3.0), FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c2')

def c3():
    model = PredictionModel(
        backbone_name='microsoft/swin-base-patch4-window7-224',
        prediction_heads=[TimeHead(), GenotypeHead(weight=3.0), FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c3')

def c4():
    model = PredictionModel(
        backbone_name='google/siglip2-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead(weight=3.0), FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c4')

def c2_2():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead(weight=10.0), FungicideHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c2-2')


def d1():
    model = PredictionModel(
        backbone_name='google/vit-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d1')

def d2():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d2')

def d3():
    model = PredictionModel(
        backbone_name='microsoft/swin-base-patch4-window7-224',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d3')

def d4():
    model = PredictionModel(
        backbone_name='google/siglip2-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d4')

def d2_2():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead()],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d2-2')

def d2_3():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead()],
        lora_options=None,
        do_normalization=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d2-3')


def d3_2():
    model = PredictionModel(
        backbone_name='microsoft/swin-base-patch4-window7-224',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None,
        do_normalization=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d3-2')

def d1_1():
    model = PredictionModel(
        backbone_name='google/vit-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d1-1')

def d2_4():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None,
        do_normalization=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d2-4')

def d4_2():
    model = PredictionModel(
        backbone_name='google/siglip2-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d4-2')

def a1_n():
    model = PredictionModel(
        backbone_name='google/vit-base-patch16-224',
        prediction_heads=[TimeHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='a1_n')

def a2_n():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='a2_n')

def a3_n():
    model = PredictionModel(
        backbone_name='microsoft/swin-base-patch4-window7-224',
        prediction_heads=[TimeHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='a3_n')

def a4_n():
    model = PredictionModel(
        backbone_name='google/siglip2-base-patch16-224',
        prediction_heads=[TimeHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='a4_n')

def c2_3():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead(weight=2.0), FungicideHead(weight=0.8)],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c2-3')


def c2_4N():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead(weight=3.0), FungicideHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c2-4-N')

def c2_5():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead(weight=5.0), FungicideHead(weight=0.8)],
        lora_options=None
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c2-5')

def c1_N():
    model = PredictionModel(
        backbone_name='google/vit-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead(weight=3.0), FungicideHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c1-N')

def c3_N():
    model = PredictionModel(
        backbone_name='microsoft/swin-base-patch4-window7-224',
        prediction_heads=[TimeHead(), GenotypeHead(weight=3.0), FungicideHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c3-N')

def c4_N():
    model = PredictionModel(
        backbone_name='google/siglip2-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead(weight=3.0), FungicideHead()],
        lora_options=None,
        do_normalization=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='c4-N')

def d2_5():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None,
        do_normalization=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='d2-5')

def k1():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead()],
        lora_options=None,
        do_normalization=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = GenotypeSplitDescriptor(
        train_genotypes=['cnj12-30-24', 'cnj14-31-142', 'cnj05-73-39', 'cnj06-3-1', 'haines'],
        test_genotypes=['cnj05-64-9', 'cnj05-80-2', 'cnj06-22-10']
    )

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='k1')

def k2():
    model = PredictionModel(
        backbone_name='google/siglip2-base-patch16-224',
        prediction_heads=[GenotypeHead()],
        lora_options=None,
        do_normalization=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = GenotypeSplitDescriptor(
        train_genotypes=['cnj12-30-24', 'cnj14-31-142', 'cnj05-73-39', 'cnj06-3-1', 'haines'],
        test_genotypes=['cnj05-64-9', 'cnj05-80-2', 'cnj06-22-10']
    )

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='k2')

def l1():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None,
        do_normalization=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = GenotypeSplitDescriptor(
        train_genotypes=['cnj12-30-24', 'cnj14-31-142', 'cnj05-73-39', 'cnj06-3-1', 'haines'],
        test_genotypes=['cnj05-64-9', 'cnj05-80-2', 'cnj06-22-10']
    )

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='l1')

def l2():
    model = PredictionModel(
        backbone_name='google/siglip2-base-patch16-224',
        prediction_heads=[TimeHead(), GenotypeHead()],
        lora_options=None,
        do_normalization=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = GenotypeSplitDescriptor(
        train_genotypes=['cnj12-30-24', 'cnj14-31-142', 'cnj05-73-39', 'cnj06-3-1', 'haines'],
        test_genotypes=['cnj05-64-9', 'cnj05-80-2', 'cnj06-22-10']
    )

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='l2')

def m1():
    model = PredictionModel(
        backbone_name='facebook/dinov2-with-registers-base',
        prediction_heads=[GenotypeHead()],
        lora_options=None,
        do_normalization=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_locations = utils.load_toml('./dataset_locations.toml')    
    bog_2_patch_df = torch.load('prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    dataset_split_descriptor = CoordSplitDescriptor(approx_train_size=0.8)

    config = {
        "hostname": socket.gethostname(),
        "backbone_name": model.backbone_name,
        "prediction_heads": str(model.prediction_heads),
        "lora_config": str(model.lora_config),
        "dataset_split_descriptor": dataset_split_descriptor,
        "learning_rate": 0.005,
        "epochs": 8,
        "batch_size": 32
    }

    train(model, dataset, config, save_dir='./checkpoints_pw', use_wandb=True, test_id='m1')

'google/vit-base-patch16-224'
'facebook/dinov2-with-registers-base'
'microsoft/swin-base-patch4-window7-224'
'google/siglip2-base-patch16-224'


if __name__ == "__main__":
    # c1_N()
    # clear_mem()
    # c3_N()
    # clear_mem()
    # c4_N()
    # clear_mem()
    # d2_5()
    # clear_mem()
    # k1()
    # clear_mem()
    # k2()
    # clear_mem()
    # l1()
    # clear_mem()
    # l2()
    m1()