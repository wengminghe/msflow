import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np

from datasets import load_dataset
from models.msflow import MSFlow
from utils.loss import MSFlowLoss
from utils.metric import auroc_accuracy_precision_recall


class Trainer(object):
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = args.mode
        self.epoch = 1
        self.epochs = args.epochs
        self.top_k = args.top_k
        self.threshold = args.threshold

        self.output_dir = args.output_dir

        if self.mode == 'train':
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.output_dir)

        self.train_dataloader = load_dataset(args, is_train=True)
        self.val_dataloader = load_dataset(args, is_train=False)

        self.model = MSFlow(**vars(args)).to(self.device)
        self.criterion = MSFlowLoss()
        self.train_params = list(self.model.fusion_flow.parameters()) + list(self.model.parallel_flows.parameters())
        self.optimizer = torch.optim.Adam(self.train_params, lr=args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_milestones,
                                                                 gamma=args.lr_gamma)

        self.auroc = -1
        self.accuracy = -1
        self.precision = -1
        self.recall = -1
        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(os.path.join(self.output_dir, 'checkpoint.pth')):
            data = torch.load(os.path.join(self.output_dir, 'checkpoint.pth'), map_location='cpu')
            data['model'] = self.convert_state_dict(data['model'])
            self.model.load_state_dict(data['model'])
            self.optimizer.load_state_dict(data['optimizer'])
            self.lr_scheduler.load_state_dict(data['lr_scheduler'])
            self.epoch = data['epoch']
            self.auroc = data['auroc']
            self.accuracy = data['accuracy']
            self.precision = data['precision']
            self.recall = data['recall']

    def convert_state_dict(self, state_dict):
        maps = {}
        for i in range(len(self.model.parallel_flows)):
            maps[self.model.fusion_flow.module_list[i].perm.shape[0]] = i
        temp = {}
        for k, v in state_dict.items():
            if 'fusion_flow' in k and 'perm' in k:
                temp[k.replace(k.split('.')[2], str(maps[v.shape[0]]))] = v
        for k, v in temp.items():
            state_dict[k] = v
        return state_dict

    def save_checkpoint(self):
        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'auroc': self.auroc,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall
        }
        torch.save(data, os.path.join(self.output_dir, 'checkpoint.pth'))

    def save_model(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, name))

    def run(self):
        # self.val_epoch()
        if self.mode == 'test':
            state_dict = torch.load(os.path.join(self.output_dir, 'best_recall_model.pth'))
            state_dict = self.convert_state_dict(state_dict)
            self.model.load_state_dict(state_dict)
            self.val_epoch()
            return
        while self.epoch <= self.epochs:
            self.train_epoch()
            self.val_epoch()
            self.epoch += 1
            self.save_checkpoint()

    def train_epoch(self):
        self.model.parallel_flows.train()
        self.model.fusion_flow.train()

        epoch_loss = 0.
        p_bar = tqdm(self.train_dataloader, desc=f'Train Epoch[{self.epoch}/{self.epochs}]', ncols=100)
        for image, _, _ in p_bar:
            if len(image.shape) == 5:
                image = image.flatten(0, 1)
            image = image.to(self.device)
            outputs = self.model(image)
            loss = self.criterion(outputs)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.train_params, 2)
            self.optimizer.step()

            epoch_loss += loss.item()
            p_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss /= len(self.train_dataloader)
        print(f'Train Epoch: [{self.epoch}/{self.epochs}] Average loss: {epoch_loss:.4f}')
        self.writer.add_scalar('train/loss', epoch_loss, self.epoch)
        # self.lr_scheduler.step()

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        label_list = []
        score_list = []
        epoch_loss = 0.
        size = 0.

        p_bar = tqdm(self.val_dataloader, desc=f'Val Epoch[{self.epoch}/{self.epochs}]', ncols=100)
        for image, label, _ in p_bar:
            if len(image.shape) == 5:
                image = image.flatten(0, 1)
            image = image.to(self.device)
            label_list.extend(label.cpu().numpy())

            if label == 0:
                outputs = self.model(image)
                loss = self.criterion(outputs)
                epoch_loss += loss.item()
                size += 1

            score, _ = self.model.predict(image, top_k=self.top_k)
            score = np.max(score)
            score_list.append(score)

        epoch_loss /= size
        auroc, accuracy, precision, recall, threshold = auroc_accuracy_precision_recall(score_list, label_list)
        print(f'Val Epoch: [{self.epoch}/{self.epochs}] Loss: {epoch_loss:.4f}, AUROC: {auroc:.2f}, '
              f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, Threshold: {threshold:.3f}')

        if self.mode == 'train':
            self.writer.add_scalar('val/loss', epoch_loss, self.epoch)
            self.writer.add_scalar('val/auroc', auroc, self.epoch)
            self.writer.add_scalar('val/accuracy', accuracy, self.epoch)
            self.writer.add_scalar('val/precision', precision, self.epoch)
            self.writer.add_scalar('val/recall', recall, self.epoch)
            self.writer.add_scalar('val/threshold', threshold, self.epoch)

            if auroc > self.auroc:
                self.save_model('best_auroc_model.pth')
                self.auroc = auroc
            if accuracy > self.accuracy:
                self.save_model('best_accuracy_model.pth')
                self.accuracy = accuracy
            if precision > self.precision:
                self.save_model('best_precision_model.pth')
                self.precision = precision
            if recall > self.recall:
                self.save_model('best_recall_model.pth')
                self.recall = recall

