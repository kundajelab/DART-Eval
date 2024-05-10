import os
import math
import heapq
import hashlib
import warnings
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import polars as pl
import pyBigWig
import h5py
from ncls import NCLS
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from tqdm import tqdm

from ..utils import copy_if_not_exists, log1mexp

class AssayEmbeddingsDataset(IterableDataset):
    _elements_dtypes = {
        "chr": pl.Utf8,
        "input_start": pl.UInt32,
        "input_end": pl.UInt32,
        "elem_start": pl.UInt32,
        "elem_end": pl.UInt32,
        "elem_relative_start": pl.UInt32,
        "elem_relative_end": pl.UInt32
    }

    def __init__(self, embeddings_h5, elements_tsv, chroms, assay_bw, bounds=None, crop=0, downsample_ratio=1, cache_dir=None):
        super().__init__()

        self.elements_df_all = self._load_elements(elements_tsv, chroms)
        self.embeddings_h5 = embeddings_h5
        self.assay_bw = assay_bw
        self.bounds = bounds
        self.crop = crop
        self.downsample_ratio = downsample_ratio

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

            embeddings_h5_abs = os.path.abspath(embeddings_h5)
            embeddings_h5_hash = hashlib.sha256(embeddings_h5_abs.encode('utf-8')).hexdigest()
            embeddings_h5_cache_path = os.path.join(cache_dir, embeddings_h5_hash + ".h5")
            copy_if_not_exists(embeddings_h5, embeddings_h5_cache_path)
            self.embeddings_h5 = embeddings_h5_cache_path

            bw_path_abs = os.path.abspath(assay_bw)
            bw_path_hash = hashlib.sha256(bw_path_abs.encode('utf-8')).hexdigest()
            bw_cache_path = os.path.join(cache_dir, bw_path_hash + ".bw")
            copy_if_not_exists(assay_bw, bw_cache_path)
            self.assay_bw = bw_cache_path

        self.next_epoch = 0
        self._set_epoch()

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

    def _set_epoch(self):
        segment = self.next_epoch % self.downsample_ratio
        total_elements = self.elements_df_all.height
        segment_boundaries = np.linspace(0, total_elements, self.downsample_ratio + 1).round().astype(np.int32)
        start = segment_boundaries[segment]
        end = segment_boundaries[segment + 1]

        self.elements_df = self.elements_df_all.slice(start, end - start)
        self.next_epoch += 1

    def __len__(self):
        return self.elements_df.height

    def __iter__(self):
        worker_info = get_worker_info()
        if self.bounds is not None:
            start, end = self.bounds
        elif worker_info is None:
            start = 0
            end = len(self)
        else:
            per_worker = int(math.ceil(len(self) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self))

        df_sub = self.elements_df.slice(start, end - start)
        valid_inds = df_sub.get_column('region_idx').to_numpy().astype(np.int32)
        region_idx_to_row = {v: i for i, v in enumerate(valid_inds)}
        query_struct = NCLS(valid_inds, valid_inds + 1, valid_inds)

        bw = pyBigWig.open(self.assay_bw)

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

            for chunk_start, chunk_end in chunk_ranges:
                chunk_range = list(query_struct.find_overlap(chunk_start, chunk_end))
                if len(chunk_range) == 0:
                    continue

                seq_chunk = h5[f"seq/emb_{chunk_start}_{chunk_end}"][:]

                if not idx_seq_fixed:
                    idx_seq_chunk = h5["seq/idx_var"][chunk_start:chunk_end]

                for i, _, _ in chunk_range:
                    # print(worker_info.id, i, "a") ####
                    i_rel = i - chunk_start
                    if idx_seq_fixed:
                        seq_inds = idx_seq_dset
                    else:
                        seq_inds = idx_seq_chunk[i_rel].astype(np.int64)

                    # print(worker_info.id, i, "b") ####
                    seq_emb = seq_chunk[i_rel]

                    # print(df_sub[i]) ####
                    # print(worker_info.id, i, "c") ####
                    _, chrom, region_start, region_end, _, _, _, _ = self.elements_df.row(region_idx_to_row[i])
                    # print(chrom, region_start, region_end) ####

                    # print(worker_info.id, i, "d") ####
                    track = np.nan_to_num(bw.values(chrom, region_start, region_end, numpy=True))
                    if self.crop > 0:
                        track = track[self.crop:-self.crop]

                    # print(worker_info.id, i, "e") ####
                    yield torch.from_numpy(seq_emb), torch.from_numpy(seq_inds), torch.from_numpy(track)

        bw.close()
        self._set_epoch()


class InterleavedIterableDataset(IterableDataset):
    def __init__(self, datasets):
        super().__init__()

        self.datasets = datasets

    # def set_epoch(self, epoch):
    #     for d in self.datasets:
    #         d.set_epoch(epoch)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __iter__(self):
        lengths = []
        worker_info = get_worker_info()
        for d in self.datasets:
            if worker_info is None:
                d.bounds = (0, len(d))
                lengths.append(len(d))
            else:
                per_worker = int(math.ceil(len(d) / float(worker_info.num_workers)))
                start = worker_info.id * per_worker
                end = min(start + per_worker, len(d))
                d.bounds = (start, end)
                lengths.append(end - start)

        iterators = [iter(dataset) for dataset in self.datasets]
        heap = [(0., 0, l, i) for i, l in enumerate(lengths) if l > 0]
        heapq.heapify(heap)
        while heap:
            frac, complete, length, ind = heapq.heappop(heap)
            yield_vals = list(next(iterators[ind]))
            yield_vals.append(torch.tensor(ind, dtype=torch.long))

            yield tuple(yield_vals)

            if complete + 1 < length:
                updated_record = ((complete + 1) / length, complete + 1, length, ind)
                heapq.heappush(heap, updated_record)


class PeaksEmbeddingsDataset(IterableDataset):
    _elements_dtypes = {
        "chr": pl.Utf8,
        "input_start": pl.UInt32,
        "input_end": pl.UInt32,
        "label": pl.Utf8,
    }

    def __init__(self, embeddings_h5, elements_tsv, chroms, classes, bounds=None, cache_dir=None):
        super().__init__()

        self.classes = classes
        self.elements_df = self._load_elements(elements_tsv, chroms)
        self.embeddings_h5 = embeddings_h5
        self.bounds = bounds

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

            embeddings_h5_abs = os.path.abspath(embeddings_h5)
            embeddings_h5_hash = hashlib.sha256(embeddings_h5_abs.encode('utf-8')).hexdigest()
            embeddings_h5_cache_path = os.path.join(cache_dir, embeddings_h5_hash + ".h5")
            copy_if_not_exists(embeddings_h5, embeddings_h5_cache_path)
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
        if self.bounds is not None:
            start, end = self.bounds
        elif worker_info is None:
            start = 0
            end = len(self)
        else:
            per_worker = int(math.ceil(len(self) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self))

        df_sub = self.elements_df.slice(start, end - start)
        valid_inds = df_sub.get_column('region_idx').to_numpy().astype(np.int32)
        region_idx_to_row = {v: i for i, v in enumerate(valid_inds)}
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

            for chunk_start, chunk_end in chunk_ranges:
                chunk_range = list(query_struct.find_overlap(chunk_start, chunk_end))
                if len(chunk_range) == 0:
                    continue

                seq_chunk = h5[f"seq/emb_{chunk_start}_{chunk_end}"][:]

                if not idx_seq_fixed:
                    idx_seq_chunk = h5["seq/idx_var"][chunk_start:chunk_end]

                for i, _, _ in chunk_range:
                    # print(worker_info.id, i, "a") ####
                    i_rel = i - chunk_start
                    if idx_seq_fixed:
                        seq_inds = idx_seq_dset
                    else:
                        seq_inds = idx_seq_chunk[i_rel].astype(np.int64)

                    # print(worker_info.id, i, "b") ####
                    seq_emb = seq_chunk[i_rel]

                    # print(df_sub[i]) ####
                    # print(worker_info.id, i, "c") ####
                    _, chrom, start, end, _, _, _, label = self.elements_df.row(region_idx_to_row[i])
                    label_ind = self.classes[label]

                    yield torch.from_numpy(seq_emb), torch.from_numpy(seq_inds), torch.tensor(label_ind)


def log1pMSELoss(log_predicted_counts, true_counts):
    log_true = torch.log(true_counts+1)
    return torch.mean(torch.square(log_true - log_predicted_counts), dim=-1)


def pearson_correlation(a, b):
    a = a - torch.mean(a)
    b = b - torch.mean(b)

    var_a = torch.sum(a ** 2)
    var_b = torch.sum(b ** 2)
    cov = torch.sum(a * b)

    r = cov / torch.sqrt(var_a * var_b)
    r = torch.nan_to_num(r)

    return r.item()

def counts_pearson(log_preds, targets):
    log_targets = torch.log(targets + 1)

    r = pearson_correlation(log_preds, log_targets)

    return r


def counts_spearman(log_preds, targets):
    log_targets = torch.log(targets + 1)

    preds_rank = log_preds.argsort().argsort().float()
    targets_rank = log_targets.argsort().argsort().float()

    r = pearson_correlation(preds_rank, targets_rank)

    return r


def _collate_batch(batch):
    max_seq_len = max(seq_emb.shape[0] for seq_emb, _, _, _ in batch)
    seq_embs = torch.zeros(len(batch), max_seq_len, batch[0][0].shape[1])
    for i, (seq_emb, _, _, _) in enumerate(batch):
        seq_embs[i,:seq_emb.shape[0]] = seq_emb

    seq_inds = torch.stack([seq_inds for _, seq_inds, _, _ in batch])
    tracks = torch.stack([track for _, _, track, _ in batch])
    indicators = torch.stack([indicator for _, _, _, indicator in batch])

    return seq_embs, seq_inds, tracks, indicators
    

def train_predictor(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, lr, num_workers, prefetch_factor, device, progress_bar=False, resume_from=None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_batch, 
                                  pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_batch, 
                                pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=False)

    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "train.log")
    log_cols = ["epoch", "val_loss", "val_pearson_all", "val_spearman_all", "val_pearson_peaks", "val_spearman_peaks"]

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            f.flush()

        for epoch in range(start_epoch, num_epochs):
            model.train()
            for i, (seq_emb, seq_inds, track, indicator) in enumerate(tqdm(train_dataloader, disable=(not progress_bar), desc="train")):
                seq_emb = seq_emb.to(device)
                seq_inds = seq_inds.to(device)
                track = track.to(device)
                true_counts = track.sum(dim=1)

                # print(seq_emb.shape, seq_inds.shape, track.shape, indicator.shape) ####
                
                optimizer.zero_grad()
                log1p_counts = model(seq_emb, seq_inds)
                loss = log1pMSELoss(log1p_counts, true_counts)
                loss.backward()
                optimizer.step()
            
            val_loss = 0
            val_counts_pred = []
            val_counts_true = []
            val_indicators = []
            model.eval()
            with torch.no_grad():
                for i, (seq_emb, seq_inds, track, indicator) in enumerate(tqdm(val_dataloader, disable=(not progress_bar), desc="val")):
                    seq_emb = seq_emb.to(device)
                    seq_inds = seq_inds.to(device)
                    track = track.to(device)
                    true_counts = track.sum(dim=1)
                    
                    optimizer.zero_grad()
                    log1p_counts = model(seq_emb, seq_inds)
                    loss = log1pMSELoss(log1p_counts, true_counts)

                    val_loss += loss.item()
                    val_counts_pred.append(log1p_counts)
                    val_counts_true.append(true_counts)
                    val_indicators.append(indicator)

            val_loss /= len(val_dataloader)
            val_counts_pred = torch.cat(val_counts_pred, dim=0)
            val_counts_true = torch.cat(val_counts_true, dim=0)
            val_indicators = torch.cat(val_indicators, dim=0)

            val_counts_pred_peaks = val_counts_pred[val_indicators == 0]
            val_counts_true_peaks = val_counts_true[val_indicators == 0]

            val_pearson_all = counts_pearson(val_counts_pred, val_counts_true)
            val_pearson_peaks = counts_pearson(val_counts_pred_peaks, val_counts_true_peaks)
            val_spearman_all = counts_spearman(val_counts_pred, val_counts_true)
            val_spearman_peaks = counts_spearman(val_counts_pred_peaks, val_counts_true_peaks)

            print(f"Epoch {epoch}: val_loss={val_loss}, val_pearson_all={val_pearson_all}, val_spearman_all={val_spearman_all}, val_pearson_peaks={val_pearson_peaks}, val_spearman_peaks={val_spearman_peaks}")
            f.write(f"{epoch}\t{val_loss}\t{val_pearson_all}\t{val_spearman_all}\t{val_pearson_peaks}\t{val_spearman_peaks}\n")
            f.flush()

            checkpoint_path = os.path.join(out_dir, f"checkpoint_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)


def evaluate_chromatin_model(pos_dataset, idr_dataset, neg_dataset, model, batch_size, out_path,
                                       num_workers, prefetch_factor, device, progress_bar=False, seed=0):

    torch.manual_seed(seed)
    
    model.to(device)

    model.eval()

    with torch.no_grad():
        test_loss_pos = 0
        test_counts_pred_pos = []
        test_counts_true_pos = []
        test_pos_dataloader = DataLoader(pos_dataset, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=True, prefetch_factor=prefetch_factor)
        for i, (seq_emb, seq_inds, track) in enumerate(tqdm(test_pos_dataloader, disable=(not progress_bar), desc="test_pos", ncols=120)):
            seq_emb = seq_emb.to(device)
            seq_inds = seq_inds.to(device)
            track = track.to(device)
            true_counts = track.sum(dim=1)
            
            log1p_counts = model(seq_emb, seq_inds)
            loss = log1pMSELoss(log1p_counts, true_counts)

            test_loss_pos += loss.item()
            test_counts_pred_pos.append(log1p_counts)
            test_counts_true_pos.append(true_counts)

        test_counts_pred_pos = torch.cat(test_counts_pred_pos, dim=0)
        test_counts_true_pos = torch.cat(test_counts_true_pos, dim=0)
        test_pearson_pos = counts_pearson(test_counts_pred_pos, test_counts_true_pos)
        test_spearman_pos = counts_spearman(test_counts_pred_pos, test_counts_true_pos)
        test_loss_pos /= len(test_pos_dataloader)

        test_loss_idr = 0
        test_counts_pred_idr = []
        test_counts_true_idr = []
        test_idr_dataloader = DataLoader(idr_dataset, batch_size=batch_size, num_workers=num_workers,
                                            pin_memory=True, prefetch_factor=prefetch_factor)
        for i, (seq_emb, seq_inds, track) in enumerate(tqdm(test_idr_dataloader, disable=(not progress_bar), desc="test_idr", ncols=120)):
            seq_emb = seq_emb.to(device)
            seq_inds = seq_inds.to(device)
            track = track.to(device)
            true_counts = track.sum(dim=1)
            
            log1p_counts = model(seq_emb, seq_inds)
            loss = log1pMSELoss(log1p_counts, true_counts)

            test_loss_idr += loss.item()
            test_counts_pred_idr.append(log1p_counts)
            test_counts_true_idr.append(true_counts)

        test_counts_pred_idr = torch.cat(test_counts_pred_idr, dim=0)
        test_counts_true_idr = torch.cat(test_counts_true_idr, dim=0)
        test_pearson_idr = counts_pearson(test_counts_pred_idr, test_counts_true_idr)
        test_spearman_idr = counts_spearman(test_counts_pred_idr, test_counts_true_idr)
        test_loss_idr /= len(test_idr_dataloader)

        test_loss_neg = 0
        test_counts_pred_neg = []
        test_counts_true_neg = []
        test_neg_dataloader = DataLoader(neg_dataset, batch_size=batch_size, num_workers=num_workers,
                                            pin_memory=True, prefetch_factor=prefetch_factor)
        for i, (seq_emb, seq_inds, track) in enumerate(tqdm(test_neg_dataloader, disable=(not progress_bar), desc="test_neg", ncols=120)):
            seq_emb = seq_emb.to(device)
            seq_inds = seq_inds.to(device)
            track = track.to(device)
            true_counts = track.sum(dim=1)
            
            log1p_counts = model(seq_emb, seq_inds)#.squeeze(1)  
            loss = log1pMSELoss(log1p_counts, true_counts)

            test_loss_neg += loss.item()
            test_counts_pred_neg.append(log1p_counts)
            test_counts_true_neg.append(true_counts)

        test_counts_pred_neg = torch.cat(test_counts_pred_neg, dim=0)
        test_counts_true_neg = torch.cat(test_counts_true_neg, dim=0)
        test_pearson_neg = counts_pearson(test_counts_pred_neg, test_counts_true_neg)
        test_spearman_neg = counts_spearman(test_counts_pred_neg, test_counts_true_neg)
        test_loss_neg /= len(test_neg_dataloader)

        test_loss_all = (test_loss_pos + test_loss_neg) / 2
        test_counts_pred_all = torch.cat([test_counts_pred_pos, test_counts_pred_neg], dim=0)
        test_counts_true_all = torch.cat([test_counts_true_pos, test_counts_true_neg], dim=0)
        test_pearson_all = counts_pearson(test_counts_pred_all, test_counts_true_all)
        test_spearman_all = counts_spearman(test_counts_pred_all, test_counts_true_all)

        test_counts_pred_cls = torch.cat([test_counts_pred_idr, test_counts_pred_neg], dim=0)
        test_labels = torch.cat([torch.ones_like(test_counts_pred_idr), torch.zeros_like(test_counts_pred_neg)], dim=0)
        test_auroc = roc_auc_score(test_labels.numpy(force=True), test_counts_pred_cls.numpy(force=True))
        test_auprc = average_precision_score(test_labels.numpy(force=True), test_counts_pred_cls.numpy(force=True))

        metrics = {
            "test_loss_pos": test_loss_pos,
            "test_loss_idr": test_loss_idr,
            "test_loss_neg": test_loss_neg,
            "test_loss_all": test_loss_all,
            "test_pearson_pos": test_pearson_pos,
            "test_pearson_idr": test_pearson_idr,
            "test_pearson_neg": test_pearson_neg,
            "test_pearson_all": test_pearson_all,
            "test_spearman_pos": test_spearman_pos,
            "test_spearman_idr": test_spearman_idr,
            "test_spearman_neg": test_spearman_neg,
            "test_spearman_all": test_spearman_all,
            "test_auroc": test_auroc,
            "test_auprc": test_auprc,
        }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def _collate_batch_classifier(batch):
    max_seq_len = max(seq_emb.shape[0] for seq_emb, _, _ in batch)
    seq_embs = torch.zeros(len(batch), max_seq_len, batch[0][0].shape[1])
    for i, (seq_emb, _, _) in enumerate(batch):
        seq_embs[i,:seq_emb.shape[0]] = seq_emb

    seq_inds = torch.stack([seq_inds for _, seq_inds, _ in batch])
    labels = torch.stack([label for _, _, label in batch])

    return seq_embs, seq_inds, labels

def train_peak_classifier(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, lr, num_workers, prefetch_factor, device, progress_bar=False, resume_from=None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_batch_classifier, 
                                  pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_batch_classifier, 
                                pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=False)

    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "train.log")
    log_cols = ["epoch", "val_loss", "val_pearson_all", "val_spearman_all", "val_pearson_peaks", "val_spearman_peaks"]

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    criterion = torch.nn.CrossEntropyLoss()

    with open(log_file, "a") as f:
        if resume_from is None:
            f.write("\t".join(log_cols) + "\n")
            f.flush()

        for epoch in range(start_epoch, num_epochs):
            model.train()
            
            optimizer.zero_grad()
            for i, (seq_emb, seq_inds, labels) in enumerate(tqdm(train_dataloader, disable=(not progress_bar), desc="train", ncols=120)):
                seq_emb = seq_emb.to(device)
                seq_inds = seq_inds.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                pred = model(seq_emb, seq_inds)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
            
            val_loss = 0
            val_acc = 0
            model.eval()
            with torch.no_grad():
                for i, (seq_emb, seq_inds, labels) in enumerate(tqdm(val_dataloader, disable=(not progress_bar), desc="val", ncols=120)):
                    seq_emb = seq_emb.to(device)
                    seq_inds = seq_inds.to(device)
                    labels = labels.to(device)
                    
                    pred = model(seq_emb, seq_inds)
                    loss = criterion(pred, labels)

                    val_loss += loss.item()
                    val_acc += (pred.argmax(dim=1) == labels).sum().item()

            val_loss /= len(val_dataloader.dataset)
            val_acc /= len(val_dataloader.dataset)

            print(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc}")
            f.write(f"{epoch}\t{val_loss}\t{val_acc}\n")
            f.flush()

            checkpoint_path = os.path.join(out_dir, f"checkpoint_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            optimizer_checkpoint_path = os.path.join(out_dir, f"optimizer_{epoch}.pt")
            torch.save(optimizer.state_dict(), optimizer_checkpoint_path)


def eval_peak_classifier(test_dataset, model, out_path, batch_size, 
                                    num_workers, prefetch_factor, device, progress_bar=False, seed=0):

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, 
                                pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)

    torch.manual_seed(seed)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
            
    test_loss = 0
    pred_log_probs = []
    labels = []
    model.eval()
    pred_logits = []
    with torch.no_grad():
        for i, (seq_emb, seq_inds, labels_batch) in enumerate(tqdm(test_dataloader, disable=(not progress_bar), desc="test", ncols=120)):
            seq_emb = seq_emb.to(device)
            seq_inds = seq_inds.to(device)
            labels_batch = labels_batch.to(device)
            
            pred = model(seq_emb, seq_inds)
            pred_logits.append(pred)
            loss = criterion(pred, labels_batch)

            test_loss += loss.item()
            pred_log_probs.append(F.log_softmax(pred, dim=1))
            labels.append(labels_batch)

    test_loss /= len(test_dataloader.dataset)

    pred_log_probs = torch.cat(pred_log_probs, dim=0)
    log_probs_others = log1mexp(pred_log_probs)
    log_probs_others = torch.nan_to_num(log_probs_others, neginf=-999)
    pred_log_odds = pred_log_probs - log_probs_others
    labels = torch.cat(labels, dim=0)
    test_acc = (pred_log_probs.argmax(dim=1) == labels).sum().item() / len(test_dataloader.dataset)

    pred_labels = pred_log_probs.argmax(dim=1)

    metrics = {"test_loss": test_loss, "test_acc": test_acc}

    for class_name, class_idx in test_dataloader.dataset.classes.items():
        class_log_odds = pred_log_odds[:, class_idx]
        class_preds = (class_log_odds >= 0).float()
        class_labels = (labels == class_idx).float()

        class_log_odds = class_log_odds.numpy(force=True)
        class_preds = class_preds.numpy(force=True)
        class_labels = class_labels.numpy(force=True)

        class_auroc = roc_auc_score(class_labels, class_log_odds)
        class_auprc = average_precision_score(class_labels, class_log_odds)
        class_mcc = matthews_corrcoef(class_labels, class_preds)
        class_acc = (class_preds == class_labels).sum().item() / len(class_labels)

        metrics[f"class_{class_name}_auroc"] = class_auroc
        metrics[f"class_{class_name}_auprc"] = class_auprc
        metrics[f"class_{class_name}_mcc"] = class_mcc
        metrics[f"class_{class_name}_acc"] = class_acc

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

class CNNEmbeddingsPredictorBase(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, out_channels=1):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(input_channels, hidden_channels, 1, padding=1)
        self.conv2 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1)
        self.conv3 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1)
        self.fc1 = torch.nn.Linear(hidden_channels, out_channels)

    @staticmethod
    def _detokenize(embs, inds):
        return embs

    def forward(self, embs, inds):
        torch.cuda.synchronize()
        x = self._detokenize(embs, inds)
        x = x.swapaxes(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.sum(dim=2)
        x = self.fc1(x)
        x = x.squeeze(-1)
        return x
    
class CNNEmbeddingsPredictor(CNNEmbeddingsPredictorBase):
    @staticmethod
    def _detokenize(embs, inds):
        gather_idx = inds[:,:,None].expand(-1,-1,embs.shape[2]).to(embs.device)
        seq_embeddings = torch.gather(embs, 1, gather_idx)

        return seq_embeddings

class CNNSlicedEmbeddingsPredictor(CNNEmbeddingsPredictorBase):
    @staticmethod
    def _detokenize(embs, inds):
        positions = torch.arange(embs.shape[1], device=embs.device)
        start_mask = positions[None,:] >= inds[:,0][:,None]
        end_mask = positions[None,:] < inds[:,1][:,None]
        mask = start_mask & end_mask
        seq_embeddings = embs[mask].reshape(embs.shape[0], -1, embs.shape[2])

        return seq_embeddings
    

class CNNSequenceBaselinePredictor(torch.nn.Module):
    def __init__(self, emb_channels, hidden_channels, kernel_size, seq_len, init_kernel_size, pos_channels, out_channels=1):
        super().__init__()

        self.iconv = torch.nn.Conv1d(4, emb_channels, kernel_size=init_kernel_size, padding='same')
        self.pos_emb = torch.nn.Parameter(torch.zeros(seq_len, pos_channels))
        self.pos_proj = torch.nn.Linear(pos_channels, emb_channels)
        
        self.trunk = CNNEmbeddingsPredictorBase(emb_channels, hidden_channels, kernel_size, out_channels=out_channels)

    def forward(self, x, _):
        x = x.swapaxes(1, 2)
        x = self.iconv(x)
        x = x.swapaxes(1, 2)
        p = self.pos_proj(self.pos_emb)
        x = F.relu(x + p)

        x = self.trunk(x, None)

        return x
    

