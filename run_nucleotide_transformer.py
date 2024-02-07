from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="Given an input sequence and model name, run the nucleotide transformer model on the sequence.")
	parser.add_argument("--dna_sequence", type=str, required=True, help="DNA Sequence to run prediction on")
	parser.add_argument("--model_max_length", type=int, default=0, help="The length to which the input sequences are padded to.")
	parser.add_argument("--model_name", type=str, default="nucleotide-transformer-500m-human-ref", help="Options: 'nucleotide-transformer-2.5b-1000g', 'nucleotide-transformer-500m-human-ref', 'nucleotide-transformer-500m-1000g'")
	args = parser.parse_args()
	return args


def get_model_embeddings(model_name, model_max_length, dna_sequence):
	# Import the tokenizer and the model
	model_name = f"InstaDeepAI/{model_name}"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForMaskedLM.from_pretrained(model_name)

	# Choose the length to which the input sequences are padded. By default, the 
	# model max length is chosen, but feel free to decrease it as the time taken to 
	# obtain the embeddings increases significantly with it.

	if model_max_length == 0:
		max_length = tokenizer.model_max_length
	else:
		max_length = args.model_max_length

	# Tokenize input DNA Sequence
	tokens_ids = tokenizer.batch_encode_plus(dna_sequence, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]

	# Compute the embeddings
	attention_mask = tokens_ids != tokenizer.pad_token_id
	torch_outs = model(
    		tokens_ids,
    		attention_mask=attention_mask,
    		encoder_attention_mask=attention_mask,
    		output_hidden_states=True
	)

	# Compute sequences embeddings
	embeddings = torch_outs['hidden_states'][-1].detach().numpy()
	print(f"Embeddings shape: {embeddings.shape}")
	print(f"Embeddings per token: {embeddings}")

	# Add embed dimension axis
	attention_mask = torch.unsqueeze(attention_mask, dim=-1)

	# Compute mean embeddings per sequence
	mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
	print(f"Mean sequence embeddings: {mean_sequence_embeddings}")

def main():
	# Example run: python3 run_nucleotide_transformer.py "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT" 0 "nucleotide-transformer-500m-human-ref" 

	args = parse_args()
	get_model_embeddings(args.model_name, args.model_max_length, args.dna_sequence)

if __name__ == "__main__":
	main()

