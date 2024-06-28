import json
import pandas as pd
import sys

train_peaks = pd.read_csv(sys.argv[1], sep="\t")
valid_peaks = pd.read_csv(sys.argv[2], sep="\t")
test_peaks = pd.read_csv(sys.argv[3], sep="\t")
out_file = sys.argv[4]

train_len, valid_len, test_len = len(train_peaks), len(valid_peaks), len(test_peaks)

split_dict = {
    "train": list(range(train_len)),
    "valid": list(range(train_len, train_len + valid_len)),
    "test": list(range(train_len + valid_len, train_len + valid_len + test_len))

}

assert split_dict["valid"][0] == split_dict["train"][-1] + 1
assert split_dict["test"][0] == split_dict["valid"][-1] + 1

json.dump(split_dict, open(out_file, "w"))