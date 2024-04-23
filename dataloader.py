import random
import numpy as np
import torch
import torch.utils.data as data_utils
import dataset


torch.set_default_tensor_type(torch.DoubleTensor)


class NILMDataloader():
    def __init__(self, args, train_labels, train_total_power, val_labels, val_total_power, test_labels, test_total_power, bert=False):
        self.args = args
        self.mask_prob = args.mask_prob
        self.batch_size = args.batch_size

        if bert:
            self.train_dataset, self.val_dataset, self.test_dataset = dataset.get_bert_datasets(self, train_labels, train_total_power, val_labels, val_total_power, test_labels, test_total_power, mask_prob=self.mask_prob)
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = dataset.get_datasets(self, train_labels, train_total_power, val_labels, val_total_power, test_labels, test_total_power)
        print(self.train_dataset)
    @classmethod
    def code(cls):
        return 'dataloader'

    def get_dataloaders(self):
        train_loader = self._get_loader(self.train_dataset)
        val_loader = self._get_loader(self.val_dataset)
        test_loader = self._get_loader(self.test_dataset)  # Utilizza il dataset di test
        return train_loader, val_loader, test_loader

    def _get_loader(self, dataset):
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader

class NILMDataset(data_utils.Dataset):
    def __init__(self, y, x, window_size=480, stride=30):
        super(NILMDataset, self).__init__()
        self.x = x
        self.y = y
        self.window_size = window_size
        self.stride = stride


        #trasformali in array
        self.x = np.array(self.x)
        labels_str = [str(labels) for labels in self.y]
        self.y = labels_str
        # Verifica che x e y siano liste di liste con lunghezza maggiore della finestra temporale
        '''for sample_x, sample_y in zip(self.x, self.y):
          print("Sample x length:", len(sample_x))
          print("Sample y length:", len(sample_y))
          #assert isinstance(sample_x, list) and isinstance(sample_y, list), "x e y devono essere liste di liste"
          assert len(sample_x) >= window_size, "La lunghezza di ogni lista deve essere almeno pari alla finestra temporale"
          assert all(isinstance(label, str) for label in self.y), "Le etichette devono essere stringhe"'''



    '''def __len__(self):
        return int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)'''
    # Restituisci la lunghezza complessiva del dataset
    def __len__(self):
        total_length = 0
        for sample_x in self.x:
            total_length += int(np.ceil((len(sample_x) - self.window_size) / self.stride) + 1)
        return total_length
    

    def __getitem__(self, index):
      start_index = index * self.stride
      end_index = np.min((len(self.x), index * self.stride + self.window_size))
      x = self.padding_seqs(self.x[start_index: end_index])

      # Converti le etichette da stringhe a indici numerici utilizzando il dizionario
      class_to_index = {class_name: index for index, class_name in enumerate(self.y)}
      labels_numeric = [class_to_index[label_str] for label_str in self.y[start_index: end_index]]

      labels_numeric = self.padding_seqs(labels_numeric)
      
      return torch.tensor(x), torch.tensor(labels_numeric) #torch.tensor(status)

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


class BERTDataset(data_utils.Dataset):
    def __init__(self, y, x, window_size=480, stride=30, mask_prob=0.2):
        super(BERTDataset, self).__init__()
        self.x = x  # Assegnamento dell'attributo x
        self.y = y  # Assegnamento dell'attributo y
        self.window_size = window_size
        self.stride = stride
        self.mask_prob = mask_prob
        self.columns = len(y)

        
    def __len__(self):
        return int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = np.min(
            (len(self.x), index * self.stride + self.window_size))
        x = self.padding_seqs(self.x[start_index: end_index])
        y = self.padding_seqs(self.y[start_index: end_index])

        tokens = []
        labels = []
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
            else:
                tokens.append(x[i])
                temp = np.array([-1] * self.columns)
                labels.append(temp)
        
        return torch.tensor(tokens), torch.tensor(labels)

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
