import os
from abc import ABCMeta, abstractmethod
import hashlib
import shutil
import importlib
import warnings
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig, AutoModel
from tqdm import tqdm
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

from ..finetune import HFClassifierModel, LoRAModule
from ..utils import onehot_to_chars, one_hot_encode, NoModule, copy_if_not_exists


def train_finetuned_classifier(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, lr, wd, accumulate, num_workers, prefetch_factor, device, progress_bar=False, resume_from=None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, 
                                pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)

    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "train.log")
    log_cols = ["epoch", "val_loss", "val_acc", "val_acc_paired"]

    zero = torch.tensor(0, dtype=torch.long, device=device)[None]
    one = torch.tensor(1, dtype=torch.long, device=device)[None]

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if resume_from is not None:
        resume_checkpoint_path = os.path.join(out_dir, f"checkpoint_{resume_from}.pt")
        optimizer_checkpoint_path = os.path.join(out_dir, f"optimizer_{resume_from}.pt")
        start_epoch = resume_from + 1
        checkpoint_resume = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint_resume, strict=False)
        try:
            optimizer_resume = torch.load(optimizer_checkpoint_path)
            optimizer.load_state_dict(optimizer_resume)
        except FileNotFoundError:
            warnings.warn(f"Optimizer checkpoint not found at {optimizer_checkpoint_path}")
    else:
        start_epoch = 0

    with open(log_file, "a") as f:
        if resume_from is None:
            f.write("\t".join(log_cols) + "\n")

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(start_epoch, num_epochs):
            optimizer.zero_grad()
            model.train()
            for i, (seq, ctrl, _) in enumerate(tqdm(train_dataloader, disable=(not progress_bar), desc="train", ncols=120)):
                # seq = seq.to(device)
                # ctrl = ctrl.to(device)
                
                out_seq = model(seq)
                loss_seq = criterion(out_seq, one.expand(out_seq.shape[0])) / accumulate
                loss_seq.backward()
                out_ctrl = model(ctrl)
                loss_ctrl = criterion(out_ctrl, zero.expand(out_ctrl.shape[0])) / accumulate
                loss_ctrl.backward()

                if ((i + 1) % accumulate == 0):
                    optimizer.step()
                    optimizer.zero_grad()

            optimizer.step()
        
            val_loss = 0
            val_acc = 0
            val_acc_paired = 0
            model.eval()
            with torch.no_grad():
                for i, (seq, ctrl, _) in enumerate(tqdm(val_dataloader, disable=(not progress_bar), desc="val", ncols=120)):
                    # seq = seq.to(device)
                    # ctrl = ctrl.to(device)

                    out_seq = model(seq)
                    out_ctrl = model(ctrl)
                    loss_seq = criterion(out_seq, one.expand(out_seq.shape[0]))
                    loss_ctrl = criterion(out_ctrl, zero.expand(out_ctrl.shape[0]))
                    val_loss += (loss_seq + loss_ctrl).item()
                    val_acc += (out_seq.argmax(1) == 1).sum().item() + (out_ctrl.argmax(1) == 0).sum().item()
                    val_acc_paired += ((out_seq - out_ctrl).argmax(1) == 1).sum().item()
            
            val_loss /= len(val_dataloader.dataset) * 2
            val_acc /= len(val_dataloader.dataset) * 2
            val_acc_paired /= len(val_dataloader.dataset)

            print(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc}, val_acc_paired={val_acc_paired}")
            f.write(f"{epoch}\t{val_loss}\t{val_acc}\t{val_acc_paired}\n")
            f.flush()

            checkpoint_path = os.path.join(out_dir, f"checkpoint_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            optimizer_checkpoint_path = os.path.join(out_dir, f"optimizer_{epoch}.pt")
            torch.save(optimizer.state_dict(), optimizer_checkpoint_path)


def evaluate_finetuned_classifier(test_dataset, model, out_path, batch_size,num_workers, prefetch_factor, device, progress_bar=False):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, prefetch_factor=prefetch_factor)

    zero = torch.tensor(0, dtype=torch.long, device=device)[None]
    one = torch.tensor(1, dtype=torch.long, device=device)[None]

    model.to(device)
    
    # os.makedirs(out_dir, exist_ok=True)
    # scores_path = os.path.join(out_dir, "scores.tsv")
    # metrics_path = os.path.join(out_dir, "metrics.json")

    criterion = torch.nn.CrossEntropyLoss()

    # with open(scores_path, "w") as f:
    #     f.write("idx\tseq_pos_logit\tseq_neg_logit\tctrl_pos_logit\tctrl_neg_logit\n")

    metrics = {}
    test_loss = 0
    # test_acc = 0
    test_acc_paired = 0
    pred_log_probs = []
    labels = []
    for i, (seq, ctrl, inds) in enumerate(tqdm(test_dataloader, disable=(not progress_bar), desc="train", ncols=120)):
        with torch.no_grad():
            out_seq = model(seq)
            out_ctrl = model(ctrl)
            pred_log_probs.append(F.log_softmax(out_seq, dim=1))
            pred_log_probs.append(F.log_softmax(out_ctrl, dim=1))
            labels.append(one.expand(out_seq.shape[0]))
            labels.append(zero.expand(out_ctrl.shape[0]))
            loss_seq = criterion(out_seq, one.expand(out_seq.shape[0]))
            loss_ctrl = criterion(out_ctrl, zero.expand(out_ctrl.shape[0]))
            test_loss += (loss_seq + loss_ctrl).item()
            # test_acc += (out_seq.argmax(1) == 1).sum().item() + (out_ctrl.argmax(1) == 0).sum().item()
            test_acc_paired += ((out_seq - out_ctrl).argmax(1) == 1).sum().item()

    pred_log_probs = torch.cat(pred_log_probs, dim=0).numpy(force=True)
    pred_logits = pred_log_probs[:,1] - pred_log_probs[:,0]
    labels = torch.cat(labels, dim=0).numpy(force=True)

    test_loss /= len(test_dataloader.dataset) * 2
    test_acc_paired /= len(test_dataloader.dataset)

    test_acc = (pred_log_probs.argmax(axis=1) == labels).sum().item() / (len(test_dataloader.dataset) * 2)
    test_auroc = roc_auc_score(labels, pred_logits)
    test_auprc = average_precision_score(labels, pred_logits)
    test_mcc = matthews_corrcoef(labels, pred_log_probs.argmax(axis=1))

    metrics["test_loss"] = test_loss
    metrics["test_acc"] = test_acc
    metrics["test_acc_paired"] = test_acc_paired
    metrics["test_auroc"] = test_auroc
    metrics["test_auprc"] = test_auprc
    metrics["test_mcc"] = test_mcc

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

    
class DNABERT2LoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"zhihan1996/{model_name}"
        with NoModule("triton"):
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            config = BertConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, config=config)
            model.bert.embeddings = LoRAModule(model.bert.embeddings, lora_rank, lora_alpha, lora_dropout)
            model.bert.encoder = LoRAModule(model.bert.encoder, lora_rank, lora_alpha, lora_dropout)

        super().__init__(tokenizer, model)


class MistralDNALoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"RaphaelMourad/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model = LoRAModule(model.model, lora_rank, lora_alpha, lora_dropout)
        
        super().__init__(tokenizer, model)


class GENALMLoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"AIRI-Institute/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_base = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        gena_module_name = model_base.__class__.__module__
        cls = getattr(importlib.import_module(gena_module_name), 'BertForSequenceClassification')
        model = cls.from_pretrained(model_name, num_labels=num_labels)

        model.bert.embeddings = LoRAModule(model.bert.embeddings, lora_rank, lora_alpha, lora_dropout)
        model.bert.encoder = LoRAModule(model.bert.encoder, lora_rank, lora_alpha, lora_dropout)

        super().__init__(tokenizer, model)


class NucleotideTransformerLoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"InstaDeepAI/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels)
        model.esm = LoRAModule(model.esm, lora_rank, lora_alpha, lora_dropout)

        super().__init__(tokenizer, model)

    def _tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens.to(self.device), None


class HyenaDNALoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"LongSafari/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels)
        model.hyena = LoRAModule(model.hyena, lora_rank, lora_alpha, lora_dropout)

        super().__init__(tokenizer, model)

    def _tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens.to(self.device), None
    
    def forward(self, seqs):
        tokens, _ = self._tokenize(seqs)

        torch_outs = self.model(tokens)
        logits = torch_outs.logits

        return logits


class CaduceusLoRAModel(HFClassifierModel):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, num_labels):
        model_name = f"kuleshov-group/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels)
        model.caduceus = LoRAModule(model.caduceus, lora_rank, lora_alpha, lora_dropout)

        super().__init__(tokenizer, model)

    def _tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        encoded = self.tokenizer(seqs_str, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]

        return tokens.to(self.device), None
    
    def forward(self, seqs):
        tokens, _ = self._tokenize(seqs)

        torch_outs = self.model(tokens)
        logits = torch_outs.logits

        return logits


class LargeCNNClassifier(torch.nn.Module):
    def __init__(self, input_channels, n_filters, n_residual_convs, output_channels, seq_len, pos_channels=1, first_kernel_size=21, residual_kernel_size=3):
        super().__init__()
        self.n_residual_convs = n_residual_convs
        self.iconv = torch.nn.Conv1d(input_channels, n_filters, kernel_size=first_kernel_size)
        self.irelu = torch.nn.ReLU()

        self.pos_emb = torch.nn.Parameter(torch.zeros(seq_len, pos_channels))
        self.pos_proj = torch.nn.Linear(pos_channels, n_filters)

        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=residual_kernel_size, 
                dilation=2**i) for i in range(n_residual_convs)
        ])
        self.rrelus = torch.nn.ModuleList([
            torch.nn.ReLU() for i in range(n_residual_convs)
        ])
        self.output_layer = torch.nn.Linear(n_filters, output_channels)

        device_indicator = torch.empty(0)
        self.register_buffer("device_indicator", device_indicator)
    
    @property
    def device(self):
        return self.device_indicator.device
        
    def forward(self, x):
        x = x.to(self.device).float()

        x = x.swapaxes(1, 2)
        x = self.iconv(x)
        x = x.swapaxes(1, 2)
        p = self.pos_proj(self.pos_emb)
        x = self.irelu(x + p)
        x = x.swapaxes(1, 2)
        
        # x = self.irelu(self.iconv(x))
        
        for i in range(self.n_residual_convs):
            x_conv = self.rrelus[i](self.rconvs[i](x))
            crop_amount = (x.shape[-1] - x_conv.shape[-1]) // 2
            x_cropped = x[:,:,crop_amount:-crop_amount]
            x = torch.add(x_cropped, x_conv)
            
        x = torch.mean(x, dim=-1)
        
        final_out = self.output_layer(x)
        
        return final_out