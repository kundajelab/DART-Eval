import os
import sys

from ....finetune import PeaksEndToEndDataset, train_finetuned_peak_classifier, GENALMLoRAModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

work_dir = os.environ.get("DART_WORK_DIR", "")
cache_dir = os.environ.get("DART_CACHE_DIR")

if __name__ == "__main__":
    resume_checkpoint = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model_name = "gena-lm-bert-large-t2t"

    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(work_dir,"task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")

    batch_size = 128
    num_workers = 4
    prefetch_factor = 2
    seed = 0
    device = "cuda"

    chroms_train = [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr7",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr13",
        "chr15",
        "chr16",
        "chr17",
        "chr19",
        "chrX",
        "chrY"
    ]
    
    chroms_val = [
        "chr6",
        "chr21"
    ]

    chroms_test = [
        "chr5",
        "chr10",
        "chr14",
        "chr18",
        "chr20",
        "chr22"
    ]

    emb_channels = 1024

    lora_rank = 8
    lora_alpha = 2 * lora_rank
    lora_dropout = 0.05

    accumulate = 1
    
    lr = 1e-4
    wd = 0.01
    num_epochs = 15

    out_dir = os.path.join(work_dir, f"task_3_peak_classification/supervised_models/fine_tuned/{model_name}")    

    os.makedirs(out_dir, exist_ok=True)

    classes = {
        "GM12878": 0,
        "H1ESC": 1,
        "HEPG2": 2,
        "IMR90": 3,
        "K562": 4
    } 

    train_dataset = PeaksEndToEndDataset(genome_fa, elements_tsv, chroms_train, classes, cache_dir=cache_dir)
    val_dataset = PeaksEndToEndDataset(genome_fa, elements_tsv, chroms_val, classes, cache_dir=cache_dir)

    model = GENALMLoRAModel(model_name, lora_rank, lora_alpha, lora_dropout, len(classes))

    train_finetuned_peak_classifier(train_dataset, val_dataset, model, 
                                    num_epochs, out_dir, batch_size, lr, wd, accumulate,
                                    num_workers, prefetch_factor, device, progress_bar=True, resume_from=resume_checkpoint)