import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torch.autograd.gradcheck import zero_gradients
from tqdm import tqdm

import os
import json
import random
import numpy as np
from abc import *
from pathlib import Path

from utils import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

torch.set_default_tensor_type(torch.DoubleTensor)


class Trainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, export_root, mean, std):
        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.model = model.to(self.device)
        self.export_root = Path(export_root)
        self.cutoff = torch.tensor([args.cutoff[i] for i in args.appliance_names]).to(self.device)
        self.threshold = torch.tensor([args.threshold[i] for i in args.appliance_names]).to(self.device)

        self.cutoff_synd = torch.tensor([args.cutoff_synd[i] for i in args.appliance_names]).to(self.device)
        self.threshold_synd = torch.tensor([args.threshold_synd[i] for i in args.appliance_names]).to(self.device)
        
        # Initialize C0
        self.C0 = torch.tensor(args.c0[args.appliance_names[0]]).to(self.device)
        print('C0: {}'.format(self.C0))
        
        self.normalize = args.normalize
        self.denom = args.denom

        if self.normalize == 'mean':
            self.mean, self.std = mean, std
            self.mean = torch.tensor(self.mean).to(self.device)
            self.std = torch.tensor(self.std).to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')

    def train(self):
        val_rel_err, val_abs_err = [], []
        val_acc, val_precision, val_recall, val_f1 = [], [], [], []

        best_rel_err, _, best_acc, _, _, best_f1, _, _, _, _ = self.validate()
        self._save_state_dict()

        for epoch in range(self.num_epochs):
            self.train_bert_one_epoch(epoch + 1)

            rel_err, abs_err, acc, precision, recall, f1, predictions_energy, ground_truth_energy, predicted_status, true_status = self.validate()
            val_rel_err.append(rel_err.tolist())
            val_abs_err.append(abs_err.tolist())
            val_acc.append(acc.tolist())
            val_precision.append(precision.tolist())
            val_recall.append(recall.tolist())
            val_f1.append(f1.tolist())

            if f1.mean() + acc.mean() - rel_err.mean() > best_f1.mean() + best_acc.mean() - best_rel_err.mean():
                best_f1 = f1
                best_acc = acc
                best_rel_err = rel_err
                self._save_state_dict()
    
   
    def train_one_epoch(self, epoch):
        print(f"Using device: {self.device}")
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(seqs)
            labels = labels_energy / self.cutoff
            logits_energy = self.cutoff_energy(logits * self.cutoff)
            logits_status = self.compute_status(logits_energy)

            kl_loss = self.kl(torch.log(F.softmax(logits.squeeze() / 0.1, dim=-1) + 1e-9), F.softmax(labels.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits.contiguous().view(-1).double(),
                labels.contiguous().view(-1).double())
            margin_loss = self.margin((logits_status * 2 - 1).contiguous().view(-1).double(), 
                (status * 2 - 1).contiguous().view(-1).double())
            total_loss = kl_loss + mse_loss + margin_loss
            
            on_mask = ((status == 1) + (status != logits_status.reshape(status.shape))) >= 1
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1), 
                    labels_on.contiguous().view(-1))
                total_loss += self.C0 * loss_l1_on / total_size
            
            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()


    def train_bert_one_epoch(self, epoch):
        loss_values = []
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            seqs, labels_energy, status = batch
            seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(self.device)
            batch_shape = status.shape
            self.optimizer.zero_grad()
            logits = self.model(seqs)
            labels = labels_energy / self.cutoff_synd
            logits_energy = self.cutoff_energy_synd(logits * self.cutoff_synd)
            logits_status = self.compute_status_synd(logits_energy)
            
            # Mascheramento
            mask = (status >= 0)
            labels_masked = labels[mask]
            logits_masked = logits[mask]
            status_masked = status[mask]
            logits_status_masked = logits_status[mask]

            # Calcolo delle perdite
            kl_loss = self.kl(torch.log(F.softmax(logits_masked.squeeze() / 0.1, dim=-1) + 1e-9), F.softmax(labels_masked.squeeze() / 0.1, dim=-1))
            mse_loss = self.mse(logits_masked.contiguous().view(-1).double(),
                labels_masked.contiguous().view(-1).double())
            margin_loss = self.margin((logits_status_masked * 2 - 1).contiguous().view(-1).double(), 
                (status_masked * 2 - 1).contiguous().view(-1).double())
            total_loss = kl_loss + mse_loss + margin_loss
            
            on_mask = (status >= 0) * (((status == 1) + (status != logits_status.reshape(status.shape))) >= 1)
            if on_mask.sum() > 0:
                total_size = torch.tensor(on_mask.shape).prod()
                logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
                labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
                loss_l1_on = self.l1_on(logits_on.contiguous().view(-1), 
                    labels_on.contiguous().view(-1))
                total_loss += self.C0 * loss_l1_on / total_size
            
            total_loss.backward()
            self.optimizer.step()
            loss_values.append(total_loss.item())

            average_loss = np.mean(np.array(loss_values))
            tqdm_dataloader.set_description(f'Epoch {epoch}, loss {average_loss:.2f}')

            torch.cuda.empty_cache()  # Release GPU memory

        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()
   


    #validate with plot
    def validate(self):
        """
        Validate the model on the validation dataset and return various metrics.

        Returns:
            return_rel_err (float): Mean relative error.
            return_abs_err (float): Mean absolute error.
            return_acc (float): Mean accuracy.
            return_precision (float): Mean precision.
            return_recall (float): Mean recall.
            return_f1 (float): Mean F1 score.
            predictions_energy (list): Predicted energy values.
            ground_truth_energy (list): Ground truth energy values.
            predicted_status (list): Predicted status values.
            true_status (list): Ground truth status values.
        """
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values = [], [], [], []

        predictions_energy, ground_truth_energy, predicted_status, true_status= [], [], [], []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(self.device)

                logits = self.model(seqs)

                labels = labels_energy / self.cutoff_synd
                logits_energy = self.cutoff_energy_synd(logits * self.cutoff_synd)
                logits_status = self.compute_status_synd(logits_energy)
                logits_energy = logits_energy * logits_status

                rel_err, abs_err = relative_absolute_error(logits_energy.detach().cpu().numpy().squeeze(), 
                                                        labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status.detach().cpu().numpy().squeeze(), 
                                                                        status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                predictions_energy.extend(np.array(logits_energy.detach().cpu().numpy().squeeze()).flatten().tolist())
                ground_truth_energy.extend(np.array(labels_energy.detach().cpu().numpy().squeeze()).flatten().tolist())
                predicted_status.extend(np.array(logits_status.detach().cpu().numpy().squeeze()).flatten().tolist())
                true_status.extend(np.array(status.detach().cpu().numpy().squeeze()).flatten().tolist())
               

                average_acc = np.mean(np.array(acc_values).reshape(-1))
                average_f1 = np.mean(np.array(f1_values).reshape(-1))
                average_rel_err = np.mean(np.array(relative_errors).reshape(-1))

                tqdm_dataloader.set_description('Validation, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(
                    average_rel_err, average_acc, average_f1))
                    
                torch.cuda.empty_cache()

        return_rel_err = np.array(relative_errors).mean(axis=0)
        return_abs_err = np.array(absolute_errors).mean(axis=0)
        return_acc = np.array(acc_values).mean(axis=0)
        return_precision = np.array(precision_values).mean(axis=0)
        return_recall = np.array(recall_values).mean(axis=0)
        return_f1 = np.array(f1_values).mean(axis=0)

        
        # plot visualization
        num_samples = 50000

        if len(predicted_status) > num_samples:
            predicted_status = predicted_status[:num_samples]
            true_status = true_status[:num_samples]

        if len(predictions_energy) > num_samples:
            predictions_energy = predictions_energy[:num_samples]
            ground_truth_energy = ground_truth_energy[:num_samples]
        

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Predictions vs Ground Truth Status SynD", "Predictions vs Ground Truth Energy SynD"))

        # Status plot
        fig.add_trace(go.Scatter(y=predicted_status, mode='lines+markers', name='Predictions Status', marker=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(y=true_status, mode='lines+markers', name='Ground Truth Status', marker=dict(color='green')), row=1, col=1)

        # Energy plot
        fig.add_trace(go.Scatter(y=predictions_energy, mode='lines+markers', name='Predizioni Energy', marker=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(y=ground_truth_energy, mode='lines+markers', name='Ground Truth Energy', marker=dict(color='green')), row=2, col=1)

     
        
        fig.update_layout(title_text="Validation Results", height=800)

        # Save the plot as HTML
        fig.write_html("VAL_results.html")

        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1, predictions_energy, ground_truth_energy, predicted_status, true_status
        
    

    #test ottimizzato
    '''def test(self, test_loader):
        self._load_best_model() 
        self.model.eval() 

        total_rel_err, total_abs_err = 0, 0
        total_acc, total_precision, total_recall, total_f1 = 0, 0, 0, 0
        num_batches = len(test_loader)

        predictions_energy, ground_truth_energy, predicted_status, true_status = [], [], [], []

        with torch.no_grad():  # Disabilita la registrazione del gradiente per velocizzare il processo
            tqdm_dataloader = tqdm(test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(self.device)
                logits = self.model(seqs)  # Calcola le previsioni del modello

                labels = labels_energy / self.cutoff  # Normalizza i label
                logits_energy = self.cutoff_energy(logits * self.cutoff)
                logits_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * logits_status

                # Metrics
                acc, precision, recall, f1 = acc_precision_recall_f1_score(
                    logits_status.detach().cpu().numpy().squeeze(), 
                    status.detach().cpu().numpy().squeeze()
                )
                rel_err, abs_err = relative_absolute_error(
                    logits_energy.detach().cpu().numpy().squeeze(), 
                    labels_energy.detach().cpu().numpy().squeeze()
                )

                
                total_rel_err += np.mean(rel_err)
                total_abs_err += np.mean(abs_err)
                total_acc += np.mean(acc)
                total_precision += np.mean(precision)
                total_recall += np.mean(recall)
                total_f1 += np.mean(f1)

                # Salva i valori predetti e reali
                predictions_energy.extend(np.array(logits_energy.detach().cpu().numpy().squeeze()).flatten().tolist())
                ground_truth_energy.extend(np.array(labels_energy.detach().cpu().numpy().squeeze()).flatten().tolist())
                predicted_status.extend(np.array(logits_status.detach().cpu().numpy().squeeze()).flatten().tolist())
                true_status.extend(np.array(status.detach().cpu().numpy().squeeze()).flatten().tolist())
               

                # Visualizza le metriche ad ogni iterazione
                avg_rel_err = total_rel_err / (batch_idx + 1)
                avg_abs_err = total_abs_err / (batch_idx + 1)
                avg_acc = total_acc / (batch_idx + 1)
                avg_precision = total_precision / (batch_idx + 1)
                avg_recall = total_recall / (batch_idx + 1)
                avg_f1 = total_f1 / (batch_idx + 1)

                tqdm_dataloader.set_description(
                    'Test, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(avg_rel_err, avg_acc, avg_f1)
                )

                # Libera la memoria GPU periodicamente
                torch.cuda.empty_cache()

                # Libera i tensori non necessari
                del seqs, labels_energy, status, logits, labels, logits_energy, logits_status

            # Final metrics computation
            final_rel_err = total_rel_err / num_batches
            final_abs_err = total_abs_err / num_batches
            final_acc = total_acc / num_batches
            final_precision = total_precision / num_batches
            final_recall = total_recall / num_batches
            final_f1 = total_f1 / num_batches

        
         # Visualizza previsioni vs valori reali con plotly
        num_samples = 500000

        if len(predicted_status) > num_samples:
            predicted_status = predicted_status[:num_samples]
            true_status = true_status[:num_samples]

        if len(predictions_energy) > num_samples:
            predictions_energy = predictions_energy[:num_samples]
            ground_truth_energy = ground_truth_energy[:num_samples]


        fig = make_subplots(rows=2, cols=1, subplot_titles=("Predictions vs Ground Truth Status UK-DALE", "Predictions vs Ground Truth Energy UK-DALE"))

        # Status plot
        fig.add_trace(go.Scatter(y=predicted_status, mode='lines+markers', name='Predictions Status', marker=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(y=true_status, mode='lines+markers', name='Ground Truth Status', marker=dict(color='green')), row=1, col=1)

        # Energy plot
        fig.add_trace(go.Scatter(y=predictions_energy, mode='lines+markers', name='Predictions Energy', marker=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(y=ground_truth_energy, mode='lines+markers', name='Ground Truth Energy', marker=dict(color='green')), row=2, col=1)


        fig.update_layout(title_text="Test Results", height=800)

        # Salva il grafico come file HTML
        fig.write_html("TEST_results.html")

        # Restituisci le metriche finali
        return final_rel_err, final_abs_err, final_acc, final_precision, final_recall, final_f1'''
    

    def test(self, test_loader):
        self._load_best_model() 
        self.model.eval() 

        self._load_best_model()
        self.model.eval()
        loss_values, relative_errors, absolute_errors = [], [], []
        acc_values, precision_values, recall_values, f1_values,  = [], [], [], []

        predictions_energy, ground_truth_energy, predicted_status, true_status = [], [], [], []

        label_curve = []
        e_pred_curve = []
        status_curve = []
        s_pred_curve = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                seqs, labels_energy, status = batch
                seqs, labels_energy, status = seqs.to(self.device), labels_energy.to(self.device), status.to(self.device)
                logits = self.model(seqs)
                labels = labels_energy / self.cutoff
                logits_energy = self.cutoff_energy(logits * self.cutoff)
                logits_status = self.compute_status(logits_energy)
                logits_energy = logits_energy * logits_status

                acc, precision, recall, f1 = acc_precision_recall_f1_score(logits_status.detach(
                    ).cpu().numpy().squeeze(), status.detach().cpu().numpy().squeeze())
                acc_values.append(acc.tolist())
                precision_values.append(precision.tolist())
                recall_values.append(recall.tolist())
                f1_values.append(f1.tolist())

                rel_err, abs_err = relative_absolute_error(logits_energy.detach(
                    ).cpu().numpy().squeeze(), labels_energy.detach().cpu().numpy().squeeze())
                relative_errors.append(rel_err.tolist())
                absolute_errors.append(abs_err.tolist())

                average_acc = np.mean(np.array(acc_values).reshape(-1))
                average_f1 = np.mean(np.array(f1_values).reshape(-1))
                average_rel_err = np.mean(np.array(relative_errors).reshape(-1))

                # Save results
                predictions_energy.extend(np.array(logits_energy.detach().cpu().numpy().squeeze()).flatten().tolist())
                ground_truth_energy.extend(np.array(labels_energy.detach().cpu().numpy().squeeze()).flatten().tolist())
                predicted_status.extend(np.array(logits_status.detach().cpu().numpy().squeeze()).flatten().tolist())
                true_status.extend(np.array(status.detach().cpu().numpy().squeeze()).flatten().tolist())

                

                tqdm_dataloader.set_description('Test, rel_err {:.2f}, acc {:.2f}, f1 {:.2f}'.format(
                    average_rel_err, average_acc, average_f1))
                
                label_curve.append(labels_energy.detach().cpu().numpy().tolist())
                e_pred_curve.append(logits_energy.detach().cpu().numpy().tolist())
                status_curve.append(status.detach().cpu().numpy().tolist())
                s_pred_curve.append(logits_status.detach().cpu().numpy().tolist())


                # Libera la memoria GPU periodicamente
                torch.cuda.empty_cache()

                # Libera i tensori non necessari
                del seqs, labels_energy, status, logits, labels, logits_energy, logits_status

        label_curve = np.concatenate(label_curve).reshape(-1, self.args.output_size)
        e_pred_curve = np.concatenate(e_pred_curve).reshape(-1, self.args.output_size)
        status_curve = np.concatenate(status_curve).reshape(-1, self.args.output_size)
        s_pred_curve = np.concatenate(s_pred_curve).reshape(-1, self.args.output_size)

        self._save_result({'gt': label_curve.tolist(),
            'pred': e_pred_curve.tolist()}, 'test_result.json')

        if self.args.output_size > 1:
            return_rel_err = np.array(relative_errors).mean(axis=0)
        else:
            return_rel_err = np.array(relative_errors).mean()
        return_rel_err, return_abs_err = relative_absolute_error(e_pred_curve, label_curve)
        return_acc, return_precision, return_recall, return_f1 = acc_precision_recall_f1_score(s_pred_curve, status_curve)

        #visualize plot
        num_samples = 500000

        if len(predicted_status) > num_samples:
            predicted_status = predicted_status[:num_samples]
            true_status = true_status[:num_samples]

        if len(predictions_energy) > num_samples:
            predictions_energy = predictions_energy[:num_samples]
            ground_truth_energy = ground_truth_energy[:num_samples]


        fig = make_subplots(rows=2, cols=1, subplot_titles=("Predictions vs Ground Truth Status UK-DALE", "Predictions vs Ground Truth Energy UK-DALE"))

        # Status plot
        fig.add_trace(go.Scatter(y=predicted_status, mode='lines+markers', name='Predictions Status', marker=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(y=true_status, mode='lines+markers', name='Ground Truth Status', marker=dict(color='green')), row=1, col=1)

        # Energy plot
        fig.add_trace(go.Scatter(y=predictions_energy, mode='lines+markers', name='Predictions Energy', marker=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(y=ground_truth_energy, mode='lines+markers', name='Ground Truth Energy', marker=dict(color='green')), row=2, col=1)


        fig.update_layout(title_text="Test Results", height=800)

        # Save plot as HTML file
        fig.write_html("TEST_results.html")


        return return_rel_err, return_abs_err, return_acc, return_precision, return_recall, return_f1

        
        
    

    def cutoff_energy_synd(self, data):
        columns = data.squeeze().shape[-1]

        if self.cutoff_synd.size(0) == 0:
            self.cutoff_synd = torch.tensor(
                [3100 for i in range(columns)]).to(self.device)

        data[data < 5] = 0
        data = torch.min(data, self.cutoff_synd.double())
        return data
    
    def cutoff_energy(self, data):
        columns = data.squeeze().shape[-1]

        if self.cutoff.size(0) == 0:
            self.cutoff = torch.tensor(
                [3100 for i in range(columns)]).to(self.device)

        data[data < 5] = 0
        data = torch.min(data, self.cutoff.double())
        return data
    
    

    def compute_status_synd(self, data):
        data_shape = data.shape
        columns = data.squeeze().shape[-1]

        if self.threshold_synd.size(0) == 0:
            self.threshold_synd = torch.tensor(
                [10 for i in range(columns)]).to(self.device)
        
        status = (data >= self.threshold_synd) * 1
        return status

    
    def compute_status(self, data):
        data_shape = data.shape
        columns = data.squeeze().shape[-1]

        if self.threshold.size(0) == 0:
            self.threshold = torch.tensor(
                [10 for i in range(columns)]).to(self.device)
        
        status = (data >= self.threshold) * 1
        return status



    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum)
        else:
            raise ValueError

    def _load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(
                self.export_root.joinpath('best_acc_model.pth')))
            self.model.to(self.device)
        except:
            print('Failed to load best model, continue testing with current model...')

    def _save_state_dict(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_values(self, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        torch.save(self.model.state_dict(),
                   self.export_root.joinpath('best_acc_model.pth'))

    def _save_result(self, data, filename):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        filepath = Path(self.export_root).joinpath(filename)
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)
