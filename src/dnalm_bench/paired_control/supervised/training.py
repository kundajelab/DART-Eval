# from abc import ABCMeta, abstractmethod
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import polars as pl
import pyfaidx
import h5py
from ncls import NCLS
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

    def __init__(self, embeddings_h5, elements_tsv, chroms):
        super().__init__()

        self.elements_df = self._load_elements(elements_tsv, chroms)
        self.embeddings_h5 = embeddings_h5


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

        df_sub = self.elements_df.slice(start, end)
        # print(df_sub) ####
        valid_inds = df_sub.get_column('region_idx').to_numpy().astype(np.int32)
        query_struct = NCLS(valid_inds, valid_inds + 1, valid_inds)

        with h5py.File(self.embeddings_h5) as h5:
            for chunk_slice, _, _ in h5["seq_emb"].iter_chunks():
                chunk_start, chunk_end = chunk_slice.start, chunk_slice.stop

                chunk_inds = list(query_struct.find_overlap(chunk_start, chunk_end))
                if len(chunk_inds) == 0:
                    continue

                seq_chunk = h5["seq_emb"][chunk_slice,:,:]
                ctrl_chunk = h5["ctrl_emb"][chunk_slice,:,:]
                for i, _, _ in chunk_inds:
                    # print(i, chunk_start) ####
                    i_rel = i - chunk_start
                    seq_emb = seq_chunk[i_rel]
                    ctrl_emb = ctrl_chunk[i_rel]

                    yield torch.from_numpy(seq_emb), torch.from_numpy(ctrl_emb)
    

def train_classifier(train_dataset, val_dataset, model, num_epochs, out_dir, batch_size, num_workers, prefetch_factor, device, progress_bar=False):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)

    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "train.log")
    log_cols = ["epoch", "val_loss", "val_acc", "val_acc_paired"]

    zero = torch.tensor(0, dtype=torch.long, device=device)[None]
    one = torch.tensor(1, dtype=torch.long, device=device)[None]
    # print(one.shape) ####

    with open(log_file, "w") as f:
        f.write("\t".join(log_cols) + "\n")

        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for i, (seq_emb, ctrl_emb) in enumerate(tqdm(train_dataloader, disable=(not progress_bar))):
                seq_emb = seq_emb.to(device)
                ctrl_emb = ctrl_emb.to(device)
                
                optimizer.zero_grad()
                out_seq = model(seq_emb)
                out_ctrl = model(ctrl_emb)
                loss_seq = criterion(out_seq, one.expand(out_seq.shape[0]))
                loss_ctrl = criterion(out_ctrl, zero.expand(out_ctrl.shape[0]))
                loss = loss_seq + loss_ctrl
                loss.backward()
                optimizer.step()
            
            val_loss = 0
            val_acc = 0
            val_acc_paired = 0
            with torch.no_grad():
                for i, (seq_emb, ctrl_emb) in enumerate(val_dataloader):
                    seq_emb = seq_emb.to(device)
                    ctrl_emb = ctrl_emb.to(device)

                    out_seq = model(seq_emb)
                    out_ctrl = model(ctrl_emb)
                    loss_seq = criterion(out_seq, one.expand(out_seq.shape[0]))
                    loss_ctrl = criterion(out_ctrl, zero.expand(out_ctrl.shape[0]))
                    val_loss += loss.item()
                    val_acc += (out_seq.argmax(1) == 1).sum().item() + (out_ctrl.argmax(1) == 0).sum().item()
                    val_acc_paired += ((out_seq - out_ctrl).argmax(1) == 1).sum().item()
            
            val_loss /= len(val_dataloader.dataset) * 2
            val_acc /= len(val_dataloader.dataset) * 2
            val_acc_paired /= len(val_dataloader.dataset)

            print(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc}, val_acc_paired={val_acc_paired}")
            f.write(f"{epoch}\t{val_loss}\t{val_acc}\t{val_acc_paired}\n")

            checkpoint_path = os.path.join(out_dir, f"checkpoint_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)

 
class CNNEmbeddingsClassifier(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(input_channels, hidden_channels, 1, padding=1)
        self.conv2 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1)
        self.conv3 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1)
        self.fc1 = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.sum(dim=2)
        x = self.fc1(x)

        return x