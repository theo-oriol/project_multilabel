from collections import defaultdict
import torch
import numpy as np
from torcheval.metrics import MultilabelAccuracy
from torcheval.metrics import BinaryAccuracy

class MultiLabelMetricsSaver:
    def __init__(self, num_labels,valid=False):
        self.num_labels = num_labels
        self.metrics = {
            "species_macro_accuracy":[],
            "sexe_macro_accuracy":[],
            "macro_accuracy":[],
            "accuracy":[],
            "loss":[],
        }
        self.global_step = 0
        self.mode = "valid" if valid  else "train"

    def init_buffer(self):
        self.buffer_species_macro_accuracy_preds = defaultdict(list)
        self.buffer_species_macro_accuracy_labels = defaultdict(list)
        #####################################
        self.buffer_sexe_macro_accuracy_preds = defaultdict(list)
        self.buffer_sexe_macro_accuracy_labels = defaultdict(list)
        #####################################
        self.buffer_macro_accuracy_preds = []
        self.buffer_macro_accuracy_labels = []
        #####################################        
        self.buffer_loss = []

    def merge(self):
        self.species_macro_accuracy()
        self.sexe_macro_accuracy()
        self.macro_accuracy()
        self.accuracy()
        self.loss()
        self.global_step += 1


    def update(self, y_pred, y_true, loss, info):
        self.buffering_loss(loss)

        preds = (torch.sigmoid(y_pred) > 0.5).int()
        self.buffering_species_macro_accuracy(preds,y_true,info)
        self.buffering_sexe_macro_accuracy(preds,y_true,info)
        self.buffering_macro_accuracy(preds,y_true)

    def buffering_species_macro_accuracy(self,y_pred,y_true,info):
        for sid, pred, label in zip(info, y_pred, y_true.int()):
            self.buffer_species_macro_accuracy_preds[sid[0].item()].append(pred)
            self.buffer_species_macro_accuracy_labels[sid[0].item()].append(label)

    def species_macro_accuracy(self):
        species_acc = {}
        for sid in self.buffer_species_macro_accuracy_preds:
            preds = torch.stack(self.buffer_species_macro_accuracy_preds[sid])
            labels = torch.stack(self.buffer_species_macro_accuracy_labels[sid])
            metric = MultilabelAccuracy(criteria="hamming")
            metric.update(preds, labels)
            species_acc[sid] = metric.compute()
        self.metrics["species_macro_accuracy"].append(species_acc)


    def buffering_sexe_macro_accuracy(self,y_pred,y_true,info):
        for sid, pred, label in zip(info, y_pred, y_true.int()):
            self.buffer_sexe_macro_accuracy_preds[sid[1].item()].append(pred)
            self.buffer_sexe_macro_accuracy_labels[sid[1].item()].append(label)
    
    def sexe_macro_accuracy(self):
        sexe_acc = {}
        for sid in self.buffer_sexe_macro_accuracy_preds:
            preds = torch.stack(self.buffer_sexe_macro_accuracy_preds[sid])
            labels = torch.stack(self.buffer_sexe_macro_accuracy_labels[sid])
            metric = MultilabelAccuracy(criteria="hamming")
            metric.update(preds, labels)
            sexe_acc[sid] = metric.compute()
        self.metrics["sexe_macro_accuracy"].append(sexe_acc)

    def buffering_macro_accuracy(self,y_pred,y_true):
        self.buffer_macro_accuracy_preds.extend(y_pred.cpu().numpy())
        self.buffer_macro_accuracy_labels.extend(y_true.int().cpu().numpy())
    
    def macro_accuracy(self):
        labels_tensor = torch.from_numpy(np.array(self.buffer_macro_accuracy_labels))
        preds_tensor = torch.from_numpy(np.array(self.buffer_macro_accuracy_preds))
        macro_acc = []
        for cls in range(labels_tensor.shape[1]):
            if labels_tensor[:, cls].sum() > 0:

                metric = BinaryAccuracy()
                metric.update(preds_tensor[:, cls], labels_tensor[:, cls])
                macro_acc.append(metric.compute())
            
        self.metrics["macro_accuracy"].append(np.mean(macro_acc))

    

    def accuracy(self):
        labels = torch.from_numpy(np.array(self.buffer_macro_accuracy_labels))
        preds = torch.from_numpy(np.array(self.buffer_macro_accuracy_preds))
        metric = MultilabelAccuracy(criteria="hamming")
        metric.update(preds, labels)
        results = metric.compute()
        self.metrics["accuracy"].append(results)

    def buffering_loss(self,l):
        self.buffer_loss.append(l)

    def loss(self):
        l = np.mean(self.buffer_loss)
        self.metrics["loss"].append(l)
    

    def log_to_tensorboard(self,writer):
        step = self.global_step
        writer.add_scalar(f"Loss/{self.mode}", self.metrics["loss"][-1], step)
        writer.add_scalar(f"MacroAccuracy/{self.mode}", self.metrics["macro_accuracy"][-1], step)
        writer.add_scalar(f"OverallAccuracy/{self.mode}", self.metrics["accuracy"][-1], step)

        for sid, acc in self.metrics["species_macro_accuracy"][-1].items():
            writer.add_scalar(f"Species/{sid}_Accuracy/{self.mode}", acc, step)
        
        acc = np.mean(list(self.metrics["species_macro_accuracy"][-1].values()))
        writer.add_scalar(f"Species_macro/{self.mode}", acc, step)

        for sid, acc in self.metrics["sexe_macro_accuracy"][-1].items():
            writer.add_scalar(f"Sexe/{sid}_Accuracy/{self.mode}", acc, step)

      