import random
import numpy as np
import torch
import os
import torch.utils.data as data_utils
import dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
torch.set_default_tensor_type(torch.DoubleTensor)
from collections import defaultdict
from pathlib import Path

import glob
import multiprocessing
from functools import partial

class NILMDataloader(DataLoader):
    def __init__(self, args, data_folder, appliance_names, mean, std, window_size=480, stride=30, mask_prob=0.25, bert=False):
        self.args = args
        self.mask_prob = mask_prob
        self.batch_size = args.batch_size
        self.data_folder = data_folder
        self.appliance_names = appliance_names
        self.window_size = window_size
        self.stride = stride
        self.mean = mean
        self.std = std
        
        
        
        if bert:
            self.train_dataset = BERTDataset(args, data_folder=os.path.join(self.data_folder, 'train'),
                                             appliance_names=self.appliance_names,
                                             window_size=self.window_size,
                                             stride=self.stride, mask_prob=self.mask_prob)
            
            self.val_dataset = NILMDataset(args, data_folder=os.path.join(self.data_folder, 'validation'),
                                           appliance_names=self.appliance_names,
                                           window_size=self.window_size,
                                           stride=self.stride,
                                           mean=self.mean,
                                           std=self.std)
            
            if os.path.isdir(os.path.join(data_folder, 'test')):
                self.test_dataset = NILMDataset(args, data_folder=os.path.join(self.data_folder, 'test'),
                                                appliance_names=self.appliance_names,
                                                window_size=self.window_size,
                                                stride=self.stride,
                                                mean=self.mean,
                                                std=self.std)
            elif os.path.isdir(os.path.join(data_folder, 'ukdale')):
                self.test_dataset = UKDALEDataset(args, data_folder=os.path.join(self.data_folder, 'ukdale'),
                                                  appliance_names=self.appliance_names,
                                                  window_size=self.window_size,
                                                  stride=self.stride,
                                                  mean=self.mean,
                                                  std=self.std)
        else:
            self.train_dataset = NILMDataset(args, data_folder=os.path.join(self.data_folder, 'train'),
                                             appliance_names=self.appliance_names,
                                             window_size=self.window_size,
                                             stride=self.stride)
            self.val_dataset = NILMDataset(args, data_folder=os.path.join(self.data_folder, 'validation'),
                                           appliance_names=self.appliance_names,
                                           window_size=self.window_size,
                                           stride=self.stride,
                                           mean=self.train_dataset.aggregate_mean,
                                           std=self.train_dataset.aggregate_std)
            if os.path.isdir(os.path.join(data_folder, 'test')):
                self.test_dataset = NILMDataset(args, data_folder=os.path.join(self.data_folder, 'test'),
                                                appliance_names=self.appliance_names,
                                                window_size=self.window_size,
                                                stride=self.stride,
                                                mean=self.train_dataset.aggregate_mean,
                                                std=self.train_dataset.aggregate_std)
            elif os.path.isdir(os.path.join(data_folder, 'ukdale')):
                self.test_dataset = UKDALEDataset(args, data_folder=os.path.join(self.data_folder, 'ukdale'),
                                                  appliance_names=self.appliance_names,
                                                  window_size=self.window_size,
                                                  stride=self.stride,
                                                  mean=self.train_dataset.aggregate_mean,
                                                  std=self.train_dataset.aggregate_std)

    @classmethod
    def code(cls):
        return 'dataloader'

    def get_dataloaders(self):
        train_loader = self._get_loader(self.train_dataset)
        val_loader = self._get_loader(self.val_dataset)
        test_loader = self._get_loader(self.test_dataset, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def _get_loader(self, dataset, shuffle=True):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True)
    
  

class UKDALEDataset(Dataset):
    def __init__(self, args, data_folder, appliance_names, window_size=480, stride=30, mean=None, std=None):
        self.data_folder = data_folder
        self.file_path = Path(data_folder)
        self.house_indicies = [1,2,3,4,5]
        self.appliance_names = appliance_names
        self.window_size = window_size
        self.cutoff = [args.cutoff[i] for i in ['aggregate'] + self.appliance_names]
        self.threshold = [args.threshold[i] for i in self.appliance_names]
        self.min_on = [args.min_on[i] for i in self.appliance_names]
        self.min_off = [args.min_off[i] for i in self.appliance_names]
        self.stride = stride
        self.sampling = '6s'
        self.mean = mean
        self.std = std

        print('Test train-mean:', mean)
        print('Test train-std:', std)
        print('Thresholds:', self.threshold)

        # Load and preprocess data
        self.aggregated_values, self.appliance_values, self.timestamps = self.load_data()
        print('first Timestamps test:', self.timestamps[0])
        print('last Timestamps test:', self.timestamps[-1])
        print('Aggregated values before normalization:', self.aggregated_values)
        
        self.status = self.compute_status(self.appliance_values)

        if self.mean is not None and self.std is not None:
            self.aggregated_values = (self.aggregated_values - self.mean) / self.std

        print('Appliance values:', self.appliance_values)
        print('Appliance status:', self.status)
        print('Aggregated values after normalization:', self.aggregated_values)
        print('Appliance names:', self.appliance_names)
        print('Sum of ons:', np.sum(self.status, axis=0))
        print('Total length:', self.status.shape[0])

    def compute_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        if not self.threshold:
            self.threshold = [10 for i in range(columns)]
        if not self.min_on:
            self.min_on = [1 for i in range(columns)]
        if not self.min_off:
            self.min_off = [1 for i in range(columns)]

        for i in range(columns):
            initial_status = data[:, i] >= self.threshold[i]
            status_diff = np.diff(initial_status)
            events_idx = status_diff.nonzero()

            events_idx = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

            events_idx = events_idx.reshape((-1, 2))
            on_events = events_idx[:, 0].copy()
            off_events = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                on_events = on_events[off_duration > self.min_off[i]]
                off_events = off_events[np.roll(off_duration, -1) > self.min_off[i]]

                on_duration = off_events - on_events
                on_events = on_events[on_duration >= self.min_on[i]]
                off_events = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on: off] = 1
            status[:, i] = temp_status

        return status
    
    def load_data(self):
        allowed_appliances = {'dishwasher', 'fridge', 'microwave', 'washing_machine', 'kettle'}
        for appliance in self.appliance_names:
            assert appliance in allowed_appliances, f"Invalid appliance: {appliance}. Must be one of {allowed_appliances}."

        for house_id in self.house_indicies:
            assert house_id in [1,2,3,4,5]

        if not self.cutoff:
            self.cutoff = [6000] * (len(self.appliance_names) + 1)

        directory = self.file_path

        all_timestamps = []
        all_data = []

        for house_id in self.house_indicies:
            house_folder = directory.joinpath(f'house_{house_id}')
            house_label = pd.read_csv(house_folder.joinpath('labels.dat'), sep=' ', header=None)

            house_data = pd.read_csv(house_folder.joinpath('channel_1.dat'), sep=' ', header=None)
            house_data.iloc[:, 0] = pd.to_datetime(house_data.iloc[:, 0], unit='s')
            house_data.columns = ['time', 'aggregate']
            house_data = house_data.set_index('time')
            house_data = house_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)

            appliance_list = house_label.iloc[:, 1].values
            app_index_dict = defaultdict(list)

            for appliance in self.appliance_names:
                data_found = False
                for i in range(len(appliance_list)):
                    if appliance_list[i] == appliance:
                        app_index_dict[appliance].append(i + 1)
                        data_found = True

                if not data_found:
                    app_index_dict[appliance].append(-1)

            if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                self.house_indicies.remove(house_id)
                continue

            for appliance in self.appliance_names:
                if app_index_dict[appliance][0] == -1:
                    house_data[appliance] = np.zeros(len(house_data))
                else:
                    temp_data = pd.read_csv(house_folder.joinpath(f'channel_{app_index_dict[appliance][0]}.dat'), sep=' ', header=None)
                    temp_data.iloc[:, 0] = pd.to_datetime(temp_data.iloc[:, 0], unit='s')
                    temp_data.columns = ['time', appliance]
                    temp_data = temp_data.set_index('time')
                    temp_data = temp_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)
                    house_data = pd.merge(house_data, temp_data, how='inner', on='time')

            all_timestamps.extend(house_data.index)
            all_data.append(house_data)

        entire_data = pd.concat(all_data).dropna().copy()
        entire_data = entire_data[entire_data['aggregate'] > 0]
        entire_data[entire_data < 5] = 0
        entire_data = entire_data.clip(lower=0, upper=self.cutoff)

        # Verifica la consistenza dei timestamp e dei dati
        print(f"Dimensione dei timestamp finali: {len(entire_data.index)}")
        print(f"Dimensione dei dati finali: {entire_data.shape}")

        return entire_data.values[:, 0], entire_data.values[:, 1:], entire_data.index.tolist()

    def __len__(self):
        return int(np.ceil((len(self.aggregated_values) - self.window_size) / self.stride) + 1)

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = np.min((len(self.aggregated_values), index * self.stride + self.window_size))
        x = self.padding_seqs(self.aggregated_values[start_index: end_index])
        y = self.padding_seqs(self.appliance_values[start_index: end_index])
        status = self.padding_seqs(self.status[start_index: end_index])

        return torch.tensor(x), torch.tensor(y), torch.tensor(status)

    def padding_seqs(self, in_array):
        if len(in_array) == self.window_size:
            return in_array
        try:
            out_array = np.zeros((self.window_size, in_array.shape[1]))
        except:
            out_array = np.zeros(self.window_size)

        length = len(in_array)
        out_array[:length] = in_array
        return out_array

    def min_max_scaling(self, data):
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_std = np.std(data, axis=0)

        if np.any(data_std == 0):
            return data
        else:
            scaled_data = (data - data_min) / (data_max - data_min)
            return scaled_data


class NILMDataset(Dataset):
    def __init__(self, args, data_folder, appliance_names, window_size=480, stride=30,  mean=None, std=None):
        # Inizializzazione degli attributi della classe
        self.data_folder = data_folder
        self.appliance_names = appliance_names
        self.window_size = window_size
        self.stride = stride
        self.chunk_size = 10000
        self.file_path = glob.glob(os.path.join(data_folder, "*.csv"))
        self.mean = mean
        self.std = std
        
        print('val train-mean', mean)
        print('val train-std', std)
        # Inizializzazione degli attributi da args
        self.threshold = [args.threshold_synd[i] for i in self.appliance_names]
        self.min_on = [args.min_on[i] for i in self.appliance_names]
        self.min_off = [args.min_off[i] for i in self.appliance_names]
        self.cutoff = [args.cutoff_synd[i]
                       for i in ['aggregate'] + self.appliance_names]
        print(self.threshold)
        # Filtraggio, aggiunta dello status e aggregazione dei dati
        self.filtered_df = self.filter_and_add_status_and_aggregate(appliance_names)
        self.aggregated_values = self.filtered_df['aggregated_value'].values
        self.appliance_values = self.filtered_df['value'].values
        self.status = self.filtered_df['status'].values
        self.timestamps = self.filtered_df['timestamp'].values
        print('VAL aggregated values before:', self.aggregated_values)
        print('VAL timestamps:', self.timestamps)

        # Normalize aggregated values if mean and std are provided
        if self.mean is not None and self.std is not None:
            self.aggregated_values = (self.aggregated_values - self.mean) / self.std

        print('Appliance Values:', self.appliance_values)
        print('Appliance status:', self.status)
        print('VAL aggregated values after:', self.aggregated_values)
        

        print('Appliance:', self.appliance_names)
        print('Sum of ons val:', np.sum(self.status, axis=0))
        print('Total length val:', self.status.shape[0])
        

    def compute_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        if not self.threshold:
            self.threshold = [10 for i in range(columns)]
        if not self.min_on:
            self.min_on = [1 for i in range(columns)]
        if not self.min_off:
            self.min_off = [1 for i in range(columns)]

        for i in range(columns):
            initial_status = data[:, i] >= self.threshold[i]
            status_diff = np.diff(initial_status)
            events_idx = status_diff.nonzero()

            events_idx = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(
                    events_idx, events_idx.size, initial_status.size)

            events_idx = events_idx.reshape((-1, 2))
            on_events = events_idx[:, 0].copy()
            off_events = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                on_events = on_events[off_duration > self.min_off[i]]
                off_events = off_events[np.roll(
                    off_duration, -1) > self.min_off[i]]

                on_duration = off_events - on_events
                on_events = on_events[on_duration >= self.min_on[i]]
                off_events = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on: off] = 1
            status[:, i] = temp_status

        return status
    
    def filter_and_add_status_and_aggregate(self, appliance_names):
        filtered_dfs = []
        appliance_names = ['watercooker' if name == 'kettle' else name for name in appliance_names]
        appliance_names = ['washing machine' if name == 'washing_machine' else name for name in appliance_names]
        
        for file in self.file_path:
            print(f"Reading file: {file}")
            for chunk in pd.read_csv(file, chunksize=self.chunk_size, usecols=['timestamp', 'appliance', 'value', 'aggregated_value']):
                filtered_df = chunk[chunk['appliance'].isin(appliance_names)].copy()
                if not filtered_df.empty:
                    # Convert 'value' column to numpy array for compute_status
                    values = filtered_df['value'].values.reshape(-1, 1)
                    # Compute status
                    status = self.compute_status(values)
                    # Add status to dataframe
                    filtered_df['status'] = status
                    # Step 1: Filter rows where 'aggregated_value' > 0
                    filtered_df = filtered_df[filtered_df['aggregated_value'] > 0]
                    
                    # Clip values to specific ranges for 'aggregated_value' and 'value'
                    if 'aggregated_value' in filtered_df.columns:
                        filtered_df['aggregated_value'] = filtered_df['aggregated_value'].clip(lower=0, upper=self.cutoff[0])
                    if 'value' in filtered_df.columns:
                        filtered_df['value'] = filtered_df['value'].clip(lower=0, upper=self.cutoff[1])
                    
                    filtered_dfs.append(filtered_df)

        if filtered_dfs:
            concatenated_df = pd.concat(filtered_dfs, ignore_index=True)
            print(f"Total filtered rows: {len(concatenated_df)}")
        else:
            concatenated_df = pd.DataFrame(columns=['timestamp', 'appliance', 'value', 'aggregated_value', 'status'])
            print("No rows found for the specified appliances.")

        return concatenated_df
    
    
    
    def __len__(self):
        return int(np.ceil((len(self.filtered_df) - self.window_size) / self.stride) + 1)
    
    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = np.min((len(self.aggregated_values), index * self.stride + self.window_size))
        
        x = self.padding_seqs(self.aggregated_values[start_index: end_index])
        y = self.padding_seqs(self.appliance_values[start_index: end_index])
        status = self.padding_seqs(self.status[start_index: end_index])

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        status_tensor = torch.tensor(status, dtype=torch.float32)

        return x_tensor, y_tensor, status_tensor
    
    def min_max_scaling(self, data):
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_std = np.std(data, axis=0)

        if np.any(data_std == 0):
            return data
        else:
            scaled_data = (data - data_min) / (data_max - data_min)
            return scaled_data
    
    def padding_seqs(self, in_array):
        if len(in_array) == self.window_size:
            return in_array
        try:
            out_array = np.zeros((self.window_size, in_array.shape[1]))
        except IndexError:
            out_array = np.zeros(self.window_size)

        length = len(in_array)
        out_array[:length] = in_array
        return out_array
     
class BERTDataset(Dataset):
    def __init__(self, args, data_folder, appliance_names, window_size=480, stride=30, mask_prob=0.2):
        self.data_folder = data_folder
        self.appliance_names = appliance_names
        self.window_size = window_size
        self.stride = stride
        self.mask_prob = mask_prob
        self.chunk_size = 10000
        self.file_path = glob.glob(os.path.join(data_folder, "*.csv"))

        # Inizializzazione degli attributi da args
        self.threshold = [args.threshold_synd[i] for i in self.appliance_names]
        self.min_on = [args.min_on[i] for i in self.appliance_names]
        self.min_off = [args.min_off[i] for i in self.appliance_names]
        self.cutoff = [args.cutoff_synd[i]
                       for i in ['aggregate'] + self.appliance_names]
        
        print(self.threshold)
        # Filter and load data with 'status' and 'aggregated_value'
        self.filtered_df = self.filter_and_add_status_and_aggregate(appliance_names)
        
        self.aggregated_values = self.filtered_df['aggregated_value'].values
        print('TRAIN aggregated values before:', self.aggregated_values)
        self.appliance_values = self.filtered_df['value'].values
        self.status = self.filtered_df['status'].values

        self.mean, self.std = self.calculate_train_mean_std()
        print('train mean', self.mean)
        print('train std', self.std)
        self.aggregated_values = (self.aggregated_values - self.mean) / self.std
        

        print('Appliance:', self.appliance_names)
        # Stampa i valori di appliance_values
        print('Appliance Values:', self.appliance_values)
        print('Appliance status:', self.status)
        print('TRAIN aggregated values after:', self.aggregated_values)

        print('Sum of ons train:', np.sum(self.status, axis=0))
        print('Total length train:', self.status.shape[0])
    
    def calculate_train_mean_std(self):
        self.mean = np.mean(self.aggregated_values)
        self.std = np.std(self.aggregated_values, ddof=1)

        return self.mean, self.std
    
    def compute_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        if not self.threshold:
            self.threshold = [10 for i in range(columns)]
        if not self.min_on:
            self.min_on = [1 for i in range(columns)]
        if not self.min_off:
            self.min_off = [1 for i in range(columns)]

        for i in range(columns):
            initial_status = data[:, i] >= self.threshold[i]
            status_diff = np.diff(initial_status)
            events_idx = status_diff.nonzero()

            events_idx = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(
                    events_idx, events_idx.size, initial_status.size)

            events_idx = events_idx.reshape((-1, 2))
            on_events = events_idx[:, 0].copy()
            off_events = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                on_events = on_events[off_duration > self.min_off[i]]
                off_events = off_events[np.roll(
                    off_duration, -1) > self.min_off[i]]

                on_duration = off_events - on_events
                on_events = on_events[on_duration >= self.min_on[i]]
                off_events = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on: off] = 1
            status[:, i] = temp_status

        return status
    
    def filter_and_add_status_and_aggregate(self, appliance_names):
        filtered_dfs = []
        appliance_names = ['watercooker' if name == 'kettle' else name for name in appliance_names]
        appliance_names = ['washing machine' if name == 'washing_machine' else name for name in appliance_names]
        
        for file in self.file_path:
            print(f"Reading file: {file}")
            for chunk in pd.read_csv(file, chunksize=self.chunk_size, usecols=['timestamp', 'appliance', 'value', 'aggregated_value']):
                filtered_df = chunk[chunk['appliance'].isin(appliance_names)].copy()
                if not filtered_df.empty:
                    # Convert 'value' column to numpy array for compute_status
                    values = filtered_df['value'].values.reshape(-1, 1)
                    # Compute status
                    status = self.compute_status(values)
                    # Add status to dataframe
                    filtered_df['status'] = status

                    # Step 1: Filter rows where 'aggregated_value' > 0
                    filtered_df = filtered_df[filtered_df['aggregated_value'] > 0]
                    
                    # Clip values to specific ranges for 'aggregated_value' and 'value'
                    if 'aggregated_value' in filtered_df.columns:
                        filtered_df['aggregated_value'] = filtered_df['aggregated_value'].clip(lower=0, upper=self.cutoff[0])
                    if 'value' in filtered_df.columns:
                        filtered_df['value'] = filtered_df['value'].clip(lower=0, upper=self.cutoff[1])
                    filtered_dfs.append(filtered_df)

        if filtered_dfs:
            concatenated_df = pd.concat(filtered_dfs, ignore_index=True)
            print(f"Total filtered rows: {len(concatenated_df)}")
        else:
            concatenated_df = pd.DataFrame(columns=['timestamp', 'appliance', 'value', 'aggregated_value', 'status'])
            print("No rows found for the specified appliances.")

        return concatenated_df

    

  
    
    def min_max_scaling(self, data):
        data_min = np.min(data)
        data_max = np.max(data)

        if data_max - data_min == 0:
            return data
        else:
            scaled_data = (data - data_min) / (data_max - data_min)
            return scaled_data

    def __len__(self):
        return int(np.ceil((len(self.filtered_df) - self.window_size) / self.stride) + 1)
    
    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = min(len(self.aggregated_values), index * self.stride + self.window_size)
        
        x = self.padding_seqs(self.aggregated_values[start_index: end_index])
        y = self.padding_seqs(self.appliance_values[start_index: end_index])
        status = self.padding_seqs(self.status[start_index: end_index])

        tokens = []
        labels = []
        on_offs = []
        for i in range(len(x)):
            prob = random.random()
            if prob < self.mask_prob:
                prob = random.random()
                if prob < 0.8:
                    tokens.append(-1)
                elif prob < 0.9:
                    tokens.append(np.random.normal())
                else:
                    tokens.append(x[i])

                labels.append(y[i])
                on_offs.append(status[i])
            else:
                tokens.append(x[i])
                labels.append(-1)  
                on_offs.append(-1)  
        
        return torch.tensor(tokens), torch.tensor(labels), torch.tensor(on_offs)

    def padding_seqs(self, in_array):
        if len(in_array) == self.window_size:
            return in_array
        else:
            out_array = np.zeros(self.window_size)
            length = len(in_array)
            out_array[:length] = in_array
            return out_array
    
    # Metodi getter per aggregated_values, mean, e std
    def get_aggregated_values(self):
        return self.aggregated_values

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std
