import os
import sys

from ....embeddings import GENALMEmbeddingExtractor
from ....components import SimpleSequence

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "gena-lm-bert-large-t2t"
    genome_fa = os.path.join(root_output_dir,"refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(root_output_dir,"task_3_cell-type-specific/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")
    chroms = None
    batch_size = 64
    num_workers = 0
    seed = 0
    device = "cuda"

    out_path = os.path.join(root_output_dir,f"task_3_cell-type-specific/embeddings/{model_name}.h5")
    
    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = GENALMEmbeddingExtractor(model_name, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
