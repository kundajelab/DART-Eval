import os
import sys

import torch

# from ....training import AssayEmbeddingsDataset, InterleavedIterableDataset, CNNEmbeddingsPredictor, train_predictor
from ....finetune import train_finetuned_classifier, LargeCNNClassifier
from ....components import PairedControlDataset

if __name__ == "__main__":
    n_filters = 512
    n_residual_convs = 7
    output_channels = 2
    seq_len = 330

    model = LargeCNNClassifier(4, n_filters, n_residual_convs, output_channels, seq_len)

    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")

    for name, param in model.named_parameters():
        print(name, param.numel())
