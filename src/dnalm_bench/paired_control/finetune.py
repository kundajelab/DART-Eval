import os
from abc import ABCMeta, abstractmethod
import hashlib
import shutil
import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig, AutoModel
from tqdm import tqdm
import polars as pl
import h5py
import pyfaidx
import pyBigWig

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
    # print(one.shape) ####

    if resume_from is not None:
        start_epoch = int(resume_from.split("_")[-1].split(".")[0]) + 1
        checkpoint_resume = torch.load(resume_from)
        model.load_state_dict(checkpoint_resume)
    else:
        start_epoch = 0

    with open(log_file, "a") as f:
        if resume_from is None:
            f.write("\t".join(log_cols) + "\n")

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(start_epoch, num_epochs):
            optimizer.zero_grad()
            model.train()
            for i, (seq, ctrl, _) in enumerate(tqdm(train_dataloader, disable=(not progress_bar), desc="train")):
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
                for i, (seq, ctrl, _) in enumerate(tqdm(val_dataloader, disable=(not progress_bar), desc="val")):
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
