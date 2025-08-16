import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoImageProcessor#, TrainingArguments, Trainer
from transformers import SiglipModel
from peft import LoraConfig, get_peft_model#, TaskType, PeftModel
import peft
from typing import Optional
from enum import Enum
from PIL import Image
from dataclasses import dataclass, field
from types import SimpleNamespace

@dataclass
class LoraOptions:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

@dataclass
class TimeHead():
    name: str = "time"
    loss: type[nn.modules.loss._Loss] = nn.MSELoss
    weight: float = 1 # loss weight
    shorthand: str = field(default="time", repr=False) # shorthands are used in the filename
    size: int = field(default=1, repr=False)

@dataclass
class GenotypeHead():
    name: str = "genotype"
    loss: type[nn.modules.loss._Loss] = nn.BCEWithLogitsLoss
    weight: float = 1.0
    shorthand: str = field(default="geno", repr=False)
    size: int = field(default=8, repr=False)

@dataclass
class YieldHead():
    name: str = "yield"
    loss: type[nn.modules.loss._Loss] = nn.MSELoss
    weight: float = 1.0
    shorthand: str = field(default="yld", repr=False)
    size: int = field(default=1, repr=False)

@dataclass
class FungicideHead():
    name: str = "fungicide"
    loss: type[nn.modules.loss._Loss] = nn.BCEWithLogitsLoss
    weight: float = 1.0 
    shorthand: str = field(default="fung", repr=False)
    size: int = field(default=1, repr=False)

@dataclass
class RotHead():
    name: str = "rot"
    loss: type[nn.modules.loss._Loss] = nn.BCEWithLogitsLoss
    weight: int = 1
    shorthand: str = field(default="rot", repr=False)
    size: int = field(default=1, repr=False)

class SiglipWrapper(SiglipModel):
    def forward(self, **x):
        return SimpleNamespace(last_hidden_state=super().get_image_features(**x).unsqueeze(1))

class PredictionModel(nn.Module):
    def __init__(
            self,
            backbone_name: str,
            prediction_heads: list[TimeHead | GenotypeHead | FungicideHead | RotHead],
            lora_options: Optional[LoraOptions] = None,
            do_normalization: bool = False,
            mlp_hidden_dim=256
        ):
        super().__init__()

        self.prediction_heads = prediction_heads
        self.backbone_name = backbone_name
        self.do_normalization = do_normalization

        # add the feature extractor backbone
        if 'siglip' in backbone_name.lower():
            backbone = SiglipWrapper.from_pretrained(backbone_name)
            backbone.config.hidden_size = 768 # TODO: make this more robust
        else:
            backbone = AutoModel.from_pretrained(backbone_name)
        feature_dim = backbone.config.hidden_size

        if lora_options is None:
            self.lora_config = None
            self.backbone = backbone
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            lora_config = LoraConfig(
                r=lora_options.r,
                lora_alpha=lora_options.lora_alpha,
                lora_dropout=lora_options.lora_dropout,
                target_modules=['query', 'value'], # TODO: verify this is right for all backbones we're using
                bias="none"
            )
            self.lora_config = lora_config
            self.backbone = get_peft_model(backbone, lora_config)

        # add the feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim//4)
        )

        # add each specified prediction head
        for head in prediction_heads:
            mlp = nn.Sequential(
                nn.Linear(feature_dim//4, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, head.size)
            )
            setattr(self, head.name, mlp)
            setattr(self, f'{head.name}_weight', head.weight)

        # print_trainable_parameters(self)

    @classmethod
    def from_state_dict(cls, path, prediction_heads=None, dataset_split_predictor=None):
        '''
        Instantiate a model from a checkpoint.

        Aguments:
            path (str): path to the checkpoint
            prediction_heads (optional[list[str]]): the prediction heads you want to instantiate the model with.
                if none, will use the same prediction heads that the checkpoint trained.

            dataset_split_predictior (optional): You can pass in the dataset split descriptor from your current dataset
                to verify that the checkpoint was trained with the same split. This helps prevent data leakage
        '''
        raise NotImplementedError

    def forward(self, x):
        # shouldn't be needed to do with torch.no_grad() because it should be frozen anyway
        features = self.backbone(**x)

        cls_token = features.last_hidden_state[:, 0, :]  # (batch_size, feature_dim)
        if self.do_normalization:
            cls_token /= cls_token.norm(p=2, dim=1, keepdim=True)

        cls_token_encoded = self.feat_encoder(cls_token)

        outputs = dict()

        # add outputs for the specified prediction heads to output
        for head in self.prediction_heads:
            outputs[head.name] = getattr(self, head.name)(cls_token_encoded)

        # if dict is empty (no outputs specified), just return the encoder output
        if not bool(outputs): 
            outputs['latent'] = cls_token_encoded

        return outputs

    def save(self, path, dataset_split_descriptor=None):
        state_to_save = {
            'backbone_name': self.backbone_name,
            'feat_encoder': self.feat_encoder.state_dict(),
            'do_normalization': self.do_normalization,
        }
        
        for head in self.prediction_heads:
            state_to_save[head.name] = getattr(self, head.name).state_dict()

        if self.lora_config is not None:
            state_to_save['adapter'] = peft.get_peft_model_state_dict(model.backbone)
            state_to_save['lora_config'] = self.lora_config

        if dataset_split_descriptor is not None:
            state_to_save['dataset_split_descriptor'] = dataset_split_descriptor

        torch.save(state_to_save, path)
        print(f"Model saved to {path}")

    def load(self, path):
        '''
        load the fine-tuned parts of a PredictionModel,
        i.e. applicable prediction heads and LoRA layers
        '''
        print(f'Loading from {path}')
        checkpoint = torch.load(path, weights_only=False)

        # make sure there is a key in the checkpoint for each module
        # we are trying to initialize
        self_modules = set([name for name, _ in self.named_children()])
        self_modules.remove('backbone') # we never train this
        diff = self_modules - set(checkpoint.keys())
        assert len(diff) == 0, f'Trying to initialize a model, but the following modules are not in the checkpoint: {list(diff)}'
        
        assert checkpoint['backbone_name'].lower() == self.backbone_name.lower(), \
            f'Trying to load a checkpoint that was trained with a different backbone than the current model.\n' + \
            f'\tModel uses {self.backbone_name.lower()}, checkpoint uses {checkpoint['backbone_name'].lower()}'

        # make sure the lora config matches between the checkpoint and this model
        if checkpoint.get('lora_config') != self.lora_config:
            raise Exception(f'Mismatch between LoraConfig in the checkpoint and this model')

        # load the weights for the prediction heads
        for module in self_modules:
            getattr(self, module).load_state_dict(checkpoint[module])

        # load LoRA weights, if applicable
        if self.lora_config is not None:
            peft.set_peft_model_state_dict(model.backbone, checkpoint['adapter'])

        if 'dataset_split_descriptor' in checkpoint:
            self.dataset_split_descriptor = checkpoint['dataset_split_descriptor']

        if 'do_normalization' in checkpoint:
            assert self.do_normalization == checkpoint['do_normalization']

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

if __name__ == "__main__":
    model = PredictionModel("google/siglip-base-patch16-224", [TimeHead()])

    processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224", input_data_format="channels_first")
    dummy_input = processor(images=torch.zeros(4, 3,224,224), return_tensors='pt')
    output = model(dummy_input)

    print(output.keys())
