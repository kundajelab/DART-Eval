import torch
import sys
import numpy as np
import json
import os
from ..evaluators import PairedControlDataset, HFZeroShotEvaluator, MaskedZeroShotScore, onehot_to_chars
sys.path.append("/users/patelas/regulatory_lm/src/regulatory_lm")
from modeling.model import *
MAPPING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
model_str_dict = {"InputEmbedder":InputEmbedder, "InputSeqOnlyEmbedder":InputSeqOnlyEmbedder, "InputBertEmbedder":InputBertEmbedder, "SimpleLMDecoder":SimpleLMDecoder, "TransformerLM": TransformerLM, "BicausalNet":BicausalNet, "DilatedConvNet":DilatedConvNet, "PeakClassifier":PeakClassifier}
FLOAT_DTYPES = {"float32":torch.float32, "float64":torch.float64, "bfloat16":torch.bfloat16, "float16":torch.float16}

work_dir = os.environ.get("DART_WORK_DIR", "")

def encode_sequence(sequence): 
    encoded_sequence = [MAPPING.get(nucleotide, 4) for nucleotide in sequence]
    return encoded_sequence

def encode_sequence_batch(seq_batch):
   return [encode_sequence(seq) for seq in seq_batch]
# class RegulatoryLMEmbeddingExtractor(HFEmbeddingExtractor, PairedControlEmbeddingExtractor):
#     _idx_mode = "fixed"

#     def __init__(self, model, batch_size, num_workers, device):
#         tokenizer = None
#         super().__init__(tokenizer, model, batch_size, num_workers, device)

#     def tokenize(self, seqs):
#         seqs_str = onehot_to_chars(seqs)
#         tokens = torch.tensor(encode_sequence(dna_seq))
#         return tokens, None


class RegulatoryLMEvaluator(HFZeroShotEvaluator, MaskedZeroShotScore):
    def __init__(self, model, dataset, batch_size, num_workers, device, category=12, mask_token=5):
        tokenizer = None
        self.category = category
        self.mask_token_override = mask_token
        super().__init__(tokenizer, model, dataset, batch_size, num_workers, device)

    @property
    def start_token(self):
        return None
    
    @property
    def end_token(self):
        return None

    def tokenize(self, seqs):
        seqs_str = onehot_to_chars(seqs)
        tokens = torch.tensor(encode_sequence_batch(seqs_str))
        return tokens, torch.tensor([0] * len(seqs)), torch.tensor([len(seqs[0])] * len(seqs)), None

    def score(self, tokens, starts, ends, attention_mask):
        tokens = tokens.to(device=self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
        lls = torch.zeros(tokens.shape[:2], device=self.device)
        for i in range(tokens.shape[1]):
            clip_mask = ((i >= starts) & (i < ends)).to(device=self.device)
            masked_tokens = tokens.clone()
            masked_tokens[:,i,...] = self.mask_token_override
            lls[:,i] = self.model_fwd(masked_tokens, attention_mask, tokens)[:,i] * clip_mask

        out = lls.sum(dim=1).numpy(force=True)

        return out       

    def model_fwd(self, tokens_in, attention_mask, tokens_out):
        category_tensor = torch.tensor([self.category]).to(device=self.device)
        with torch.no_grad():
            torch_outs = self.model(
                tokens_in, category_tensor
            )
            logits = torch_outs[0].swapaxes(1, 2)
            lls = -F.cross_entropy(logits, tokens_out, reduction="none")
        return lls


if __name__ == "__main__":
    model_dir = sys.argv[1]
    checkpoint = sys.argv[2]
    out_dir = sys.argv[3]

    model_dir = model_dir + "/" if model_dir[-1] != "/" else model_dir
    args = json.load(open(os.path.join(model_dir, "args.json"), "r"))
    saved_model_file = os.path.join(model_dir, f"checkpoint_{checkpoint}.pt")
    embedder = model_str_dict[args["embedder"]](args["embedding_size"], args["num_categories"], vocab_size=args["num_real_tokens"]+2, masking=True)
    encoder = model_str_dict[args["encoder"]](args["embedding_size"], args["num_encoder_layers"], n_filters=args["embedding_size"])
    decoder = model_str_dict[args["decoder"]](args["embedding_size"])
    if "classifier" in args:
        classifier = model_str_dict[args["classifier"]](args["embedding_size"])
        model = RegulatoryLMWithClassification(embedder, encoder, decoder, classifier)
    else:
        model = RegulatoryLM(embedder, encoder, decoder)
    model = torch.compile(model)
    model_info = torch.load(saved_model_file)
    model.load_state_dict(model_info["model_state"])

    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(work_dir, f"task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv")

    # out_dir = os.path.join(work_dir, f"task_1_ccre/zero_shot_outputs/likelihoods/{model_name}")

    chroms = [
        "chr5",
        "chr10",
        "chr14",
        "chr18",
        "chr20",
        "chr22"
    ]

    batch_size = 1024
    num_workers = 4
    seed = 0
    device = "cuda"

    dataset = PairedControlDataset(genome_fa, elements_tsv, chroms, seed)

    evaluator = RegulatoryLMEvaluator(model, dataset, batch_size, num_workers, device)
    
    metrics = evaluator.evaluate(out_dir, progress_bar=True)

    for k, v in metrics.items():
        print(f"{k}: {v}")    
