# from abc import ABCMeta, abstractmethod
import os
import math
import hashlib
import warnings
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import polars as pl
import pyfaidx
import h5py
from ncls import NCLS
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
# from scipy.stats import wilcoxon
from tqdm import tqdm

from ...utils import one_hot_encode

class EmbeddingsDataset(IterableDataset):
    _elements_dtypes = {
        "chr": pl.Utf8,
        "input_start": pl.UInt32,
        "input_end": pl.UInt32,
        "ccre_start": pl.UInt32,
        "ccre_end": pl.UInt32,
        "ccre_relative_start": pl.UInt32,
        "ccre_relative_end": pl.UInt32,
        "reverse_complement": pl.Boolean
    }

    def __init__(self, embeddings_h5, elements_tsv, chroms, cache_dir=None):
        super().__init__()

        self.elements_df = self._load_elements(elements_tsv, chroms)
        self.embeddings_h5 = embeddings_h5

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

            embeddings_h5_abs = os.path.abspath(embeddings_h5)
            embeddings_h5_hash = hashlib.sha256(embeddings_h5_abs.encode('utf-8')).hexdigest()
            embeddings_h5_cache_path = os.path.join(cache_dir, embeddings_h5_hash + ".fa")
            self._copy_if_not_exists(embeddings_h5, embeddings_h5_cache_path)
            self.embeddings_h5 = embeddings_h5_cache_path

    @classmethod
    def _load_elements(cls, elements_file, chroms):
        df = (
            pl.scan_csv(elements_file, separator="\t", quote_char=None, dtypes=cls._elements_dtypes)
            .with_row_count(name="region_idx")
        )
        
        if chroms is not None:
            df = df.filter(pl.col("chr").is_in(chroms))

        df = df.collect()

        return df
    
    def __len__(self):
        return self.elements_df.height

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            start = 0
            end = self.elements_df.height
        else:
            per_worker = int(math.ceil(self.elements_df.height / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, self.elements_df.height)

        df_sub = self.elements_df.slice(start, end - start)
        # print(df_sub) ####
        valid_inds = df_sub.get_column('region_idx').to_numpy().astype(np.int32)
        query_struct = NCLS(valid_inds, valid_inds + 1, valid_inds)

        chunk_start = 0
        with h5py.File(self.embeddings_h5) as h5:
            chunk_ranges = []
            for name in h5["seq"].keys():
                if name.startswith("emb_"):
                    chunk_start, chunk_end = map(int, name.split("_")[1:])
                    chunk_ranges.append((chunk_start, chunk_end))

            chunk_ranges.sort()

            if "idx_fix" in h5["seq"]:
                idx_seq_dset = h5["seq/idx_fix"][:].astype(np.int64)
                idx_seq_fixed = True
            else:
                idx_seq_fixed = False

            if "idx_fix" in h5["ctrl"]:
                idx_ctrl_dset = h5["ctrl/idx_fix"][:].astype(np.int64)
                idx_ctrl_fixed = True
            else:
                idx_ctrl_fixed = False

            for chunk_start, chunk_end in chunk_ranges:
                chunk_range = list(query_struct.find_overlap(chunk_start, chunk_end))
                if len(chunk_range) == 0:
                    continue

                seq_chunk = h5[f"seq/emb_{chunk_start}_{chunk_end}"][:]
                ctrl_chunk = h5[f"ctrl/emb_{chunk_start}_{chunk_end}"][:]

                if not idx_seq_fixed:
                    idx_seq_chunk = h5["seq/idx_var"][chunk_start:chunk_end]
                if not idx_ctrl_fixed:
                    idx_ctrl_chunk = h5["ctrl/idx_var"][chunk_start:chunk_end]

                for i, _, _ in chunk_range:
                    i_rel = i - chunk_start
                    if idx_seq_fixed:
                        seq_inds = idx_seq_dset
                    else:
                        seq_inds = idx_seq_chunk[i_rel].astype(np.int64)
                    if idx_ctrl_fixed:
                        ctrl_inds = idx_ctrl_dset
                    else:
                        ctrl_inds = idx_ctrl_chunk[i_rel].astype(np.int64)

                    seq_emb = seq_chunk[i_rel]
                    ctrl_emb = ctrl_chunk[i_rel]

                    yield torch.from_numpy(seq_emb), torch.from_numpy(ctrl_emb), torch.from_numpy(seq_inds), torch.from_numpy(ctrl_inds)


def _collate_batch(batch):
    max_seq_len = max(seq_emb.shape[0] for seq_emb, _, _, _ in batch)
    max_ctrl_len = max(ctrl_emb.shape[0] for _, ctrl_emb, _, _ in batch)
    seq_embs = torch.zeros(len(batch), max_seq_len, batch[0][0].shape[1])
    ctrl_embs = torch.zeros(len(batch), max_ctrl_len, batch[0][1].shape[1])
    for i, (seq_emb, ctrl_emb, _, _) in enumerate(batch):
        seq_embs[i,:seq_emb.shape[0]] = seq_emb
        ctrl_embs[i,:ctrl_emb.shape[0]] = ctrl_emb

    seq_inds = torch.stack([seq_inds for _, _, seq_inds, _ in batch])
    ctrl_inds = torch.stack([ctrl_inds for _, _, _, ctrl_inds in batch])

    return seq_embs, ctrl_embs, seq_inds, ctrl_inds
    

# def _detokenize(embs, inds, device):
#     gather_idx = inds[:,:,None].expand(-1,-1,embs.shape[2]).to(device)
#     seq_embeddings = torch.gather(embs, 1, gather_idx)

#     return seq_embeddings

def train_classifier(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, lr, num_workers, prefetch_factor, device, progress_bar=False, resume_from=None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_batch, 
                                  pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_batch, 
                                pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)

    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "train.log")
    log_cols = ["epoch", "val_loss", "val_acc", "val_acc_paired"]

    zero = torch.tensor(0, dtype=torch.long, device=device)[None]
    one = torch.tensor(1, dtype=torch.long, device=device)[None]
    # print(one.shape) ####

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if resume_from is not None:
        # start_epoch = int(resume_from.split("_")[-1].split(".")[0]) + 1
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
            model.train()
            for i, (seq_emb, ctrl_emb, seq_inds, ctrl_inds) in enumerate(tqdm(train_dataloader, disable=(not progress_bar), desc="train")):
                seq_emb = seq_emb.to(device)
                ctrl_emb = ctrl_emb.to(device)
                seq_inds = seq_inds.to(device)
                ctrl_inds = ctrl_inds.to(device)

                # seq_emb = _detokenize(seq_emb, seq_inds, device)
                # ctrl_emb = _detokenize(ctrl_emb, ctrl_inds, device)
                
                optimizer.zero_grad()
                out_seq = model(seq_emb, seq_inds)
                out_ctrl = model(ctrl_emb, ctrl_inds)
                loss_seq = criterion(out_seq, one.expand(out_seq.shape[0]))
                loss_ctrl = criterion(out_ctrl, zero.expand(out_ctrl.shape[0]))
                loss = loss_seq + loss_ctrl
                loss.backward()
                # clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
            
            val_loss = 0
            val_acc = 0
            val_acc_paired = 0
            model.eval()
            with torch.no_grad():
                for i, (seq_emb, ctrl_emb, seq_inds, ctrl_inds) in enumerate(tqdm(val_dataloader, disable=(not progress_bar), desc="val")):
                    seq_emb = seq_emb.to(device)
                    ctrl_emb = ctrl_emb.to(device)
                    seq_inds = seq_inds.to(device)
                    ctrl_inds = ctrl_inds.to(device)

                    # seq_emb = _detokenize(seq_emb, seq_inds, device)
                    # ctrl_emb = _detokenize(ctrl_emb, ctrl_inds, device)

                    out_seq = model(seq_emb, seq_inds)
                    out_ctrl = model(ctrl_emb, ctrl_inds)
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


def evaluate_probing_classifier(test_dataset, model, out_path, batch_size,num_workers, prefetch_factor, device, progress_bar=False):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, prefetch_factor=prefetch_factor, collate_fn=_collate_batch)

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
    for i, (seq_emb, ctrl_emb, seq_inds, ctrl_inds) in enumerate(tqdm(test_dataloader, disable=(not progress_bar), desc="train", ncols=120)):
        with torch.no_grad():
            seq_emb = seq_emb.to(device)
            ctrl_emb = ctrl_emb.to(device)
            seq_inds = seq_inds.to(device)
            ctrl_inds = ctrl_inds.to(device)

            out_seq = model(seq_emb, seq_inds)
            out_ctrl = model(ctrl_emb, ctrl_inds)

            pred_log_probs.append(F.log_softmax(out_seq, dim=1))
            pred_log_probs.append(F.log_softmax(out_ctrl, dim=1))
            labels.append(one.expand(out_seq.shape[0]))
            labels.append(zero.expand(out_ctrl.shape[0]))
            loss_seq = criterion(out_seq, one.expand(out_seq.shape[0]))
            loss_ctrl = criterion(out_ctrl, zero.expand(out_ctrl.shape[0]))
            test_loss += (loss_seq + loss_ctrl).item()
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


# class RMSNorm(torch.nn.Module):
#     """
#     Root Mean Square Layer Normalization (RMSNorm)
#     Adapted from https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L34
#     """
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#     def forward(self, x):
#         output = self._norm(x.float()).type_as(x)

#         return output
    
 
class CNNEmbeddingsClassifierBase(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(input_channels, hidden_channels, 1, padding=1)
        self.conv2 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1)
        self.conv3 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1)
        self.fc1 = torch.nn.Linear(hidden_channels, 2)

    @staticmethod
    def _detokenize(embs, inds):
        return embs

    def forward(self, embs, inds):
        x = self._detokenize(embs, inds)
        x = x.swapaxes(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.sum(dim=2)
        x = self.fc1(x)

        return x
    
class CNNEmbeddingsClassifier(CNNEmbeddingsClassifierBase):
    @staticmethod
    def _detokenize(embs, inds):
        gather_idx = inds[:,:,None].expand(-1,-1,embs.shape[2]).to(embs.device)
        seq_embeddings = torch.gather(embs, 1, gather_idx)

        return seq_embeddings

class CNNSlicedEmbeddingsClassifier(CNNEmbeddingsClassifierBase):
    @staticmethod
    def _detokenize(embs, inds):
        positions = torch.arange(embs.shape[1], device=embs.device)
        start_mask = positions[None,:] >= inds[:,0][:,None]
        end_mask = positions[None,:] < inds[:,1][:,None]
        mask = start_mask & end_mask
        seq_embeddings = embs[mask].reshape(embs.shape[0], -1, embs.shape[2])

        return seq_embeddings


class CNNSequenceBaselineClassifier(torch.nn.Module):
    def __init__(self, emb_channels, hidden_channels, kernel_size, seq_len, init_kernel_size, pos_channels):
        super().__init__()

        self.iconv = torch.nn.Conv1d(4, emb_channels, kernel_size=init_kernel_size, padding='same')
        self.pos_emb = torch.nn.Parameter(torch.zeros(seq_len, pos_channels))
        self.pos_proj = torch.nn.Linear(pos_channels, emb_channels)
        
        # self.rconvs = torch.nn.ModuleList([
        #     torch.nn.Conv1d(emb_channels, emb_channels, kernel_size=3, padding=2**i, 
        #         dilation=2**i) for i in range(1, n_layers_dil+1)
        # ])
        # self.rrelus = torch.nn.ModuleList([
        #     torch.nn.ReLU() for i in range(1, n_layers_dil+1)
        # ])

        # self.norm = nn.BatchNorm1d(emb_channels)
        # self.norm = nn.LayerNorm(emb_channels)
        # self.iconv2 = torch.nn.Conv1d(emb_channels, emb_channels, kernel_size=21, padding=10)
        self.trunk = CNNEmbeddingsClassifierBase(emb_channels, hidden_channels, kernel_size)

    def forward(self, x, _):
        x = x.swapaxes(1, 2)
        x = self.iconv(x)
        x = x.swapaxes(1, 2)
        p = self.pos_proj(self.pos_emb)
        x = F.relu(x + p)
        
        # x = x.swapaxes(1, 2)
        # for a, c in zip(self.rrelus, self.rconvs):
        #     x_conv = a(c(x))
        #     x = torch.add(x, x_conv)
        # x = x.swapaxes(1, 2)
        
        x = self.trunk(x, None)

        return x
    
    

# class CNNSequenceBaselineClassifier(torch.nn.Module):
#     def __init__(self, n_filters, n_layers):
#         super().__init__()

#         self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
#         self.irelu = torch.nn.ReLU()

#         self.rconvs = torch.nn.ModuleList([
#             torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
#                 dilation=2**i) for i in range(1, n_layers+1)
#         ])
#         self.rrelus = torch.nn.ModuleList([
#             torch.nn.ReLU() for i in range(1, n_layers+1)
#         ])

#         self.linear = torch.nn.Linear(n_filters, 2)


#     def forward(self, x, _):
#         x = x.swapaxes(1, 2)

#         x = self.irelu(self.iconv(x))
#         for a, c in zip(self.rrelus, self.rconvs):
#             x_conv = a(c(x))
#             x = torch.add(x, x_conv)

#         x = torch.mean(x[:,:,37:-37], dim=2)

#         return x


# class CNNSequenceBaselineClassifier(torch.nn.Module):
#     def __init__(self, input_channels, hidden_channels, kernel_size, n_layers_trunk):
#         super().__init__()

#         self.iconv = torch.nn.Conv1d(4, input_channels, kernel_size=21, padding=10)
#         self.irelu = torch.nn.ReLU()

#         self.rconvs = torch.nn.ModuleList([
#             torch.nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=2**i, 
#                 dilation=2**i) for i in range(1, n_layers_trunk+1)
#         ])
#         self.rrelus = torch.nn.ModuleList([
#             torch.nn.ReLU() for i in range(1, n_layers_trunk+1)
#         ])

#         self.conv1 = torch.nn.Conv1d(input_channels, hidden_channels, 1, padding=1)
#         self.conv2 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1)
#         self.conv3 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1)
#         self.fc1 = torch.nn.Linear(hidden_channels, 2)

#     def forward(self, x, _):
#         x = x.swapaxes(1, 2)

#         x = self.irelu(self.iconv(x))
#         for a, c in zip(self.rrelus, self.rconvs):
#             x_conv = a(c(x))
#             x = torch.add(x, x_conv)

#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.sum(dim=2)
#         x = self.fc1(x)

#         return x