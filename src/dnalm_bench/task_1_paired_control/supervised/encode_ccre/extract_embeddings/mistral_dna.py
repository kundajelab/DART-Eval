import os

from ...embeddings import MistralDNAEmbeddingExtractor
from ....components import PairedControlDataset

work_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "Mistral-DNA-v1-1.6B-hg38"
    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(work_dir, "task_1_ccre/processed_data/ENCFF420VPZ_processed.tsv")
    chroms = None
    batch_size = 512
    num_workers = 0
    seed = 0
    device = "cuda"

    out_dir = os.path.join(work_dir, "task_1_ccre/embeddings")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}.h5")

    dataset = PairedControlDataset(genome_fa, elements_tsv, chroms, seed)
    extractor = MistralDNAEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
