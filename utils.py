import tomllib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from tqdm import tqdm

def load_toml(toml_file_path):
    with open(toml_file_path, 'rb') as f:
        obj = tomllib.load(f)

    return obj


var_to_letter = {
    # uppercase
    'Haines': 'A',
    'CNJ05-64-9': 'B',
    'CNJ05-73-39': 'C',
    'CNJ05-80-2': 'D',
    'CNJ06-22-10': 'E',
    'CNJ06-3-1': 'F',
    'CNJ12-30-24': 'G',
    'CNJ14-31-142': 'H',
    # lowercase
    'haines': 'A',
    'cnj05-64-9': 'B',
    'cnj05-73-39': 'C',
    'cnj05-80-2': 'D',
    'cnj06-22-10': 'E',
    'cnj06-3-1': 'F',
    'cnj12-30-24': 'G',
    'cnj14-31-142': 'H',
    # with underscore, no leading zeroes
    'Haines': 'A',
    'CNJ_5_64_9': 'B',
    'CNJ_5_73_39': 'C',
    'CNJ_5_80_2': 'D',
    'CNJ_6_22_10': 'E',
    'CNJ_6_3_1': 'F',
    'CNJ_12_30_24': 'G',
    'CNJ_14_31_142': 'H',
    # berrywise only
    'CNJ_12_20_24': 'I'
}

letter_to_var = {
    'A': 'haines' ,
    'B': 'cnj05-64-9' ,
    'C': 'cnj05-73-39' ,
    'D': 'cnj05-80-2' ,
    'E': 'cnj06-22-10' ,
    'F': 'cnj06-3-1' ,
    'G': 'cnj12-30-24' ,
    'H': 'cnj14-31-142' ,
}

def values_to_colors(values: list, cmap_name='viridis'):
    '''
    given a list of values, return a perceptually uniform
    set of colors for each unique value
    '''
    unique_values = sorted(set(values))
    n = len(unique_values)
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / (n - 1)) for i in range(n)] if n > 1 else [cmap(0.5)]
    value_to_color = {val: mcolors.to_hex(color) for val, color in zip(unique_values, colors)}
    color_list = [value_to_color[val] for val in values]
    return color_list, value_to_color


def prettify_backbone_name(backbone_name: str) -> str:
    bbn = backbone_name.lower()

    if 'vit' in bbn:
        return 'Vanilla ViT'
    elif 'dinov2' in bbn:
        return 'DINOv2'
    elif 'siglip' in bbn:
        return 'SigLIP2'
    elif 'swin' in bbn:
        return 'Swin'
    else:
        raise NotImplementedError
    
def scale_plot_lims(ax, xscale=1.0, yscale=1.0):
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    new_xlim = [x * xscale for x in current_xlim]
    new_ylim = [y * yscale for y in current_ylim]

    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)

@torch.no_grad()
def run_inference(model, processor, dataloader):
    model.eval()
    device = next(model.parameters()).device
    all_outputs = []

    print(f"Running inference on {device}")
    for patch, _ in tqdm(dataloader):
        if processor is not None:
            inputs = processor(patch, return_tensors='pt').to(device)
        else:
            inputs = {'pixel_values': patch.to(device)}
            
        outputs = model(inputs)['latent']
        all_outputs.append(outputs.cpu())

    return torch.cat(all_outputs, dim=0)