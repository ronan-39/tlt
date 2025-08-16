# import os
# import cv2
# import pandas as pd
# from skimage import io, transform

from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset
from torchvision.io import decode_image
import utils
import zarr
from PIL import Image
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import itertools

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CranberryPatchDataset(Dataset):
    def __init__(self, patch_df, config, transform=None):
        self.patch_df = patch_df
        self.transform = transform
        self.config = config
        
        assert patch_df.dataset_info is not None, '`dataset_info` field in the dataframe must be set'

        self.patch_df.loc[self.patch_df['fungicide'].isnull(), ['fungicide']] = 'control'
        
        dataset_name = patch_df.dataset_info['dataset_name']
        self.OG_IMG_RESCALE = patch_df.dataset_info['OG_IMG_RESCALE']
        self.PATCH_SIZE = patch_df.dataset_info['PATCH_SIZE']
        self.GRID_DIMS = patch_df.dataset_info['GRID_DIMS']

        self.zarr_dataset = zarr.open_array(patch_df.dataset_info['zarr_file_path'], mode='r')

        self.first_date = patch_df['date'].min()
        self.last_date = patch_df['date'].max()
        self.num_dates = len(patch_df['date'].unique())

        self.all_genotypes = patch_df['plot'].unique()
        self.num_genotypes = len(self.all_genotypes)

        self.patch_df['plot_enc'] = self.patch_df['plot'].astype('category').cat.codes

    def __len__(self):
        return len(self.patch_df.index)
    
    def __getitem__(self, idx):
        # returns images in [C, H, W] shape (for pytorch)
        patch = torch.from_numpy(self.zarr_dataset[idx]).permute(2, 0, 1).float() # it was 2,1,0, but thats erroneous maybe

        if self.transform:
            patch = self.transform(patch)

        label = self.patch_df.iloc[idx][['date', 'plot_enc', 'fungicide']]
        encoded_date = torch.tensor((label['date'] - self.first_date) / (self.last_date - self.first_date)).float()
        encoded_genotype = F.one_hot(torch.tensor(label['plot_enc']).long(), num_classes=self.num_genotypes).float()
        encoded_fungicide = torch.tensor([1.0 if label['fungicide']=='treatment' else 0.0])

        torch_labels = {
            'encoded_date': encoded_date,
            'encoded_genotype': encoded_genotype,
            'encoded_fungicide': encoded_fungicide,
        }

        # return patch, encoded_date, encoded_genotype, encoded_fungicide
        return patch, torch_labels

class CranberryBerryWiseDataset(Dataset):
    def __init__(self, berry_wise_df, dataset_locations, transform=None):
        self.df = berry_wise_df
        self.dataset_locations = dataset_locations
        self.transform = transform

        self.first_date = self.df['date'].min()
        self.last_date = self.df['date'].max()
        self.num_dates = len(self.df['date'].unique())

        self.all_genotypes = self.df['plot'].unique()
        self.num_genotypes = len(self.all_genotypes)

        self.df['plot_enc'] = self.df['plot'].astype('category').cat.codes

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        # image = decode_image(self.df.iloc[idx]['filename']) / 255
        image = self.transform(Image.open(self.df.iloc[idx]['filename']).convert("RGB"))

        #'filename', 'plot', 'date', 'fungicide', 'file_type', 'is_rotten'
        label = self.df.iloc[idx][['date', 'plot_enc', 'fungicide', 'is_rotten']]
        encoded_date = torch.tensor((label['date'] - self.first_date) / (self.last_date - self.first_date)).float()
        encoded_genotype = F.one_hot(torch.tensor(label['plot_enc']).long(), num_classes=self.num_genotypes).float()
        encoded_fungicide = torch.tensor([1.0 if label['fungicide']=='treatment' else 0.0])
        encoded_rot = torch.tensor([1.0 if label['is_rotten']==True else 0.0])

        # old
        # return image, encoded_date, encoded_genotype, encoded_fungicide
        
        # new
        torch_labels = {
            'encoded_date': encoded_date,
            'encoded_genotype': encoded_genotype,
            'encoded_fungicide': encoded_fungicide,
            'encoded_rot': encoded_rot
        }

        return image, torch_labels


@dataclass
class DefaultSplitDescriptor():
    '''randomly split dataset into test/train sets'''
    train_size: float = 0.8
    random_state: int = 1

    def split(self, dataset):
        print("Splitting dataset randomly")
        assert self.random_state == 1, "random state should usually not change. are you sure?"

        indices = list(range(len(dataset)))
        train_idxs, test_idxs = train_test_split(
            indices,
            train_size=int(len(dataset)*self.train_size),
            test_size = len(dataset) - int(len(dataset)*self.train_size),
            random_state = self.random_state
        )
        train_dataset = Subset(dataset, train_idxs)
        test_dataset = Subset(dataset, test_idxs)

        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_idxs': train_idxs,
            'test_idxs': test_idxs
        }

@dataclass
class CoordSplitDescriptor():
    '''
    split the the patchwise dataset, taking all patches on the left side of the image to be in the testing
    set, and all images on the right side to be in the training set. the split will approximate the desired
    split, but the left/right deliniator will round to the nearest column
    '''
    approx_train_size: float = 0.8

    def split(self, dataset):
        print("Splitting dataset by patch coordinates")
        max_x = round(self.approx_train_size * dataset.GRID_DIMS[1]) - 1
        
        train_row_idxs = dataset.patch_df['patch_coords'].apply(lambda x: x[1] <= max_x)
        train_idxs = np.flatnonzero(train_row_idxs)
        test_idxs = np.flatnonzero(~train_row_idxs)

        real_train_size = train_idxs.size / (train_idxs.size + test_idxs.size)
        print(f'Attempting to split the dataset with a {round(self.approx_train_size, 2)}/{round(1-self.approx_train_size, 2)} split.', end=' ')
        print(f'Actual split: {round(real_train_size, 2)}/{round(1-real_train_size, 2)}.')

        train_dataset = Subset(dataset, train_idxs)
        test_dataset = Subset(dataset, test_idxs)

        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_idxs': train_idxs,
            'test_idxs': test_idxs
        }

@dataclass
class GenotypeSplitDescriptor():
    '''dedicate `train_genotypes` fully to training, `test_genotypes` fully to testing'''
    train_genotypes: list[str]
    test_genotypes: list[str]

    def split(self, dataset):
        print("Splitting dataset by genotype")
        assert len(self.train_genotypes) != 0
        assert len(self.test_genotypes) != 0
        assert not set(self.train_genotypes).intersection(set(self.test_genotypes)), 'data leakage'

        # TODO: fix case insensitivity here
        assert all([g in dataset.all_genotypes for g in self.train_genotypes]), 'invalid train genotypes. genotypes are case-sensitive'
        assert all([g in dataset.all_genotypes for g in self.test_genotypes]), 'invalid test genotypes. genotypes are case-sensitive'

        if len(self.train_genotypes) + len(self.test_genotypes) < dataset.num_genotypes:
            print('[WARNING] you arent using all the genotypes in the dataset for this test train split. is this intentional?')

        train_row_idxs = dataset.patch_df['plot'].apply(lambda x: x in self.train_genotypes)
        test_row_idxs = dataset.patch_df['plot'].apply(lambda x: x in self.test_genotypes)

        train_idxs = np.flatnonzero(train_row_idxs)
        test_idxs = np.flatnonzero(test_row_idxs)

        train_dataset = Subset(dataset, train_idxs)
        test_dataset = Subset(dataset, test_idxs)

        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_idxs': train_idxs,
            'test_idxs': test_idxs
        }

@dataclass
class TrackSplitDescriptor():
    '''split berrywise datasets according to the split id'''
    approx_train_size: float = 0.7
    random_state: int = 1

    def split(self, dataset):
        print("Splitting dataset by track")
        assert type(dataset) == CranberryBerryWiseDataset, "track split only works with berry-wise dataset"
        assert self.random_state == 1, "random state should usually not change. are you sure?"

        rng = np.random.default_rng(seed=self.random_state) 
        unique_tracks = rng.permutation(dataset.df['track_unique'].unique())  

        images_per_track = [len(dataset.df[dataset.df['track_unique'] == uid].index) for uid in unique_tracks]
        total_images = sum(images_per_track)

        acc = list(itertools.accumulate(images_per_track))
        idx = np.argmin(np.abs(np.array(acc) - total_images * self.approx_train_size))

        train_tracks = unique_tracks[:idx]
        test_tracks = unique_tracks[idx:]

        train_idxs = np.flatnonzero(dataset.df['track_unique'].isin(train_tracks))
        test_idxs = np.flatnonzero(dataset.df['track_unique'].isin(test_tracks))

        assert len(train_idxs) + len(test_idxs) + total_images
        actual_train_split = len(train_idxs)/total_images
        print(f'Attempting to split the dataset with a {round(self.approx_train_size, 2)}/{round(1-self.approx_train_size, 2)} split.', end=' ')
        print(f'Actual split: {round(actual_train_split, 2)}/{round(1-actual_train_split, 2)}.')

        train_dataset = Subset(dataset, train_idxs)
        test_dataset = Subset(dataset, test_idxs)

        # import pandas as pd
        # pd.set_option("display.max_rows", None)
        # print(dataset.df.iloc[test_idxs])

        # print(f'{test_tracks=}')
        # print(f'{train_tracks=}')

        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_idxs': train_idxs,
            'test_idxs': test_idxs
        }

def benchmark_patch():
    import random
    from time import perf_counter

    bog_2_patch_df = torch.load('./prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    dataset_locations = utils.load_toml('./dataset_locations.toml')
    bog_2_patch_dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    n = 4
    random.seed(1)
    idxs = random.sample(range(len(bog_2_patch_dataset)), n*n)

    fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(10,10))
    axs = axs.flatten()

    print("Loading dataset sample. This may take a while")
    start = perf_counter()
    for ax, idx in zip(axs, idxs):
        # patch, _, _, _ = bog_2_patch_dataset[idx]
        patch, _ = bog_2_patch_dataset[idx]
        ax.imshow(patch.permute(1,2,0))

        plot_name = bog_2_patch_dataset.patch_df.iloc[idx]['plot']
        patch_coord = bog_2_patch_dataset.patch_df.iloc[idx]['patch_coords']
        date_readable = bog_2_patch_dataset.patch_df.iloc[idx]['date']
        fungicide_readable = bog_2_patch_dataset.patch_df.iloc[idx]['fungicide']

        ax.set_title(f'{plot_name} {patch_coord}\n{date_readable} {fungicide_readable}')
        ax.set_xticks([])
        ax.set_yticks([])
    end = perf_counter()

    print(f"Loaded {n*n} patches in {end - start:.3f} seconds. {n*n / (end-start):.3f} patches/second")

    plt.tight_layout(rect=(0., 0., 1., 0.95))
    plt.suptitle("dataset sample, patch-wise")
    plt.subplots_adjust(hspace=0.2)
    plt.show()

def benchmark_berry():
    import random
    from time import perf_counter
    from ipynb.fs.full.data_prep import get_berry_wise_df

    berry_wise_df = get_berry_wise_df()
    dataset_locations = utils.load_toml('./dataset_locations.toml')
    berry_wise_dataset = CranberryBerryWiseDataset(berry_wise_df, dataset_locations)

    n = 4
    random.seed(1)
    idxs = random.sample(range(len(berry_wise_dataset)), n*n)

    fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(10,10))
    axs = axs.flatten()

    print("Loading dataset sample. This may take a while")
    start = perf_counter()
    for ax, idx in zip(axs, idxs):
        # patch, _, _, _ = berry_wise_dataset[idx]
        patch, _ = berry_wise_dataset[idx]
        ax.imshow(patch.permute(1,2,0))

        plot_name = berry_wise_dataset.df.iloc[idx]['plot']
        date_readable = berry_wise_dataset.df.iloc[idx]['date']
        fungicide_readable = berry_wise_dataset.df.iloc[idx]['fungicide']

        ax.set_title(f'{plot_name}\n{date_readable} {fungicide_readable}')
        ax.set_xticks([])
        ax.set_yticks([])
    end = perf_counter()

    print(f"Loaded {n*n} patches in {end - start:.3f} seconds. {n*n / (end-start):.3f} patches/second")

    plt.tight_layout(rect=(0., 0., 1., 0.95))
    plt.suptitle("dataset sample, berry-wise")
    plt.subplots_adjust(hspace=0.2)
    plt.show()

if __name__ == "__main__":
    # benchmark_berry()
    benchmark_patch()
    # from ipynb.fs.full.data_prep import get_berry_wise_df

    # berry_wise_df = get_berry_wise_df()
    # dataset_locations = r_utils.load_toml('./dataset_locations.toml')
    # berry_wise_dataset = CranberryBerryWiseDataset(berry_wise_df, dataset_locations)

    # # bog_2_patch_df = torch.load('./prepped_data/bog_2_patches_p224_size_1344x2016.pt', weights_only=False) # 'p224' indicates that the patches are 224x224 pixels
    # # dataset_locations = r_utils.load_toml('./dataset_locations.toml')
    # # bog_2_patch_dataset = CranberryPatchDataset(bog_2_patch_df, dataset_locations)

    # # print(next(iter(bog_2_patch_dataset))[1:])
    # # print(next(iter(berry_wise_dataset))[1:])
    # # print(berry_wise_dataset[0][1:])
    # # print(berry_wise_dataset[1][1:])
    # # print(berry_wise_dataset[2][1:])

    # split_desc = TrackSplitDescriptor(approx_train_size=0.7)
    # data_splits = split_desc.split(berry_wise_dataset)