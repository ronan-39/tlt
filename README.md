Code and dataset coming soon

Implementation of the 'time-lapse trajectories' method presented in the associated work. Dataset referenced in associated work is publicly available below.

## Installation
1. Create the virtual environment
```bash
conda env create --n tlt python
conda activate tlt
```

2. Install PyTorch. See [pytorch](https://pytorch.org/get-started/locally/)

3. Clone this repository and install other requirements
```bash
git clone https://github.com/ronan-39/tlt
cd tlt
pip install -r requirements.txt
```

## Dataset and Preprocessing
The dataset is available in both RAW(56.4GB) and JPEG(11.3GB) formats. All results presented in the paper were obtained with the provided JPEG images. Dataset layout and other information is detailed in a readme inside the download. At a glance: ~830 images at a resolution of 8600 by 5792 pixels, taken with a Canon EOS 5DS R camera. Images retain camera metadata including exposure, aperature, ISO, etc.

Download link: [TimeLapseCranberry](https://rutgers.box.com/s/gs7op6ck3du5b12b5vflavyot1o2b89u)

### Preprocessing
The dataset is prepared by splitting the images into patches and removing patches that predominantly display the PVC frame present in each image. After downloading and extracting the dataset, follow this setup process:

1. Create `dataset_locations.toml` in the root directory of the repo. Populate it with the required fields to point towards the root directory of the dataset, i.e.
```toml
time_lapse_cranberry_root = ".../TLC_jpeg"
```

2. Run the `data/patchwise_prep.ipynb` notebook. This will take some time, but only needs to be run once. By default, this will split the original images into a 9x6 grid of 224x224px patches, remove patches with with PVC, and save them to a `.zarr` archive to `./prepped_data`.

## Code Overview
### Models
To instantiate a model, use the `PredictionModel` class from `models.py`.
In the constructor, specify:
- The backbone, by its model name on HuggingFace (tested with dinov2(/w+/wo registers), google vit)
- Which prediction heads you want to use (`TimeHead()`, `GenotypeHead()`, `FungicideHead()`, or an empty list). 
- Optionally, provide a `LoraOptions()` object to train with LoRA 

If no prediction heads are specified, the model will just output from the encoder. You should only do this if you're loading a checkpoint.

To save models, use the `save(path: str, dataset_split_descriptor=None)` method. It's optional but reccomended to specify `dataset_split_descriptor`, which will save a description of how the dataset was split alongside the checkpoint, making it easier to avoid data leakage when you run inference later.

To load models, instantiate the model *with the correct backbone and LoRA config for a given checkpoint*, and then use the `load(path: str)` method. In your instantiation, you can exclude prediction heads that were trained in the checkpoint, but you can't include prediction heads that were not trained.

### Training
See the main function `train.py`, which demonstrates all of this.