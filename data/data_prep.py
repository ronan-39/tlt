import os
from pathlib import Path
import dateutil.parser as dparser
import pandas as pd
import numpy as np
import datetime
import tomllib
import utils

config = utils.load_toml('./dataset_locations.toml')

tlc_root = config['tlc_root']
varieties = [
    '/CNJ05-64-9',
    '/CNJ05-73-39',
    '/CNJ05-80-2',
    '/CNJ06-22-10',
    '/CNJ06-3-1',
    '/CNJ14-31-142',
    '/CNJ12-30-24',
    '/Haines'
]

def get_tlc_df():
    tlc_df = pd.DataFrame(columns=['filename', 'rel_path', 'plot', 'date', 'fungicide', 'file_type'])

    for variety in varieties:
        paths = list(Path(tlc_root+variety).iterdir())
        for path in paths:
            if not path.is_file():
                    continue
            filename = path.parts[-1].lower()
            rel_path = str(Path(*path.parts[-2:]))

            if 'trt' in filename:
                fungicide = 'treatment'
            elif 'ctrl' in filename or 'control' in filename:
                fungicide = 'control'
            else:
                fungicide = None

            if 'cr2' in filename:
                file_type = 'cr2'
            if 'jpg' in filename:
                file_type = 'jpg'

            date_str = filename
            for substring in ['.', 'jpg', 'ctrl', 'control', 'trt1', 'trt', 'cr2', 'jpg']:
                date_str = date_str.replace(substring, '')
            date_str = ' '.join(date_str.replace('-', '$').replace('_', '$').split('$')[-3:])

            date = dparser.parse(date_str, fuzzy=True).date()

            plot = variety.lower()[1:]

            tlc_df.loc[len(tlc_df)] = [filename, rel_path, plot, date, fungicide, file_type]

    return tlc_df

def get_berry_wise_df():
    berry_wise_root = config['berry_wise_root']
    rot_file = Path(berry_wise_root) / "rot.txt"
    rot_dict = {}
    with open(rot_file, 'r') as f:
        for line in f:
            if line.strip():
                key, tracks = line.split(":")
                key = key.strip()
                track_set = {t.strip() for t in tracks.split(",")}
                rot_dict[key] = track_set

    berry_wise_df = pd.DataFrame(columns=['filename', 'plot', 'fungicide', 'track', 'track_unique', 'date', 'file_type', 'is_rotten'])
    for plot_dir in Path(berry_wise_root).iterdir():
        if plot_dir.is_file():
            continue
        raw_key = plot_dir.name
        for berry_dir in plot_dir.iterdir():
            if not berry_dir.is_dir():
                continue
            berry_files = list(berry_dir.iterdir())
            plots = [plot_dir.name.replace('_TRT', '').replace('_CTRL', '') for _ in berry_files]
            dates = [int(p.stem) for p in berry_files]
            fungicides = [('treatment' if 'TRT' in plot_dir.name else 'control') for _ in berry_files]
            file_types = ['png' for _ in berry_files]
            is_rotten_list = [(raw_key in rot_dict and berry_dir.name in rot_dict[raw_key]) for _ in berry_files]
            
            tracks = [int(berry_dir.name) for _ in berry_files]
            tracks_unique = [f'{t}{var_to_letter[p]}_{'ctrl' if f=='control' else 'trt'}' for t,p,f in zip(tracks, plots, fungicides) ]

            new_rows = pd.DataFrame({
                'filename': berry_files,
                'plot': plots,
                'track': tracks,
                'track_unique': tracks_unique,
                'date': dates,
                'fungicide': fungicides,
                'file_type': file_types,
                'is_rotten': is_rotten_list
            })

            berry_wise_df = pd.concat([berry_wise_df, new_rows], ignore_index=True)

    # manually filter out CNJ_6_3_1_TRT because the annotations are inaccurate
    berry_wise_df = berry_wise_df[~((berry_wise_df['plot'] == 'CNJ_6_3_1') & (berry_wise_df['fungicide'] == 'treatment'))]
    return berry_wise_df

if __name__ == '__main__':
    tlc_df = get_tlc_df()
    unique_dates = tlc_df['date'].unique()
    images_per_date = [len(tlc_df[tlc_df['date'] == d].index) for d in unique_dates]
    print(len(tlc_df))