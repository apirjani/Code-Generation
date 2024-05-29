import os, argparse, random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records, compute_error_rate
from prompting_utils import read_schema, extract_sql_query, save_logs, compute_freq_map
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # you can add mps
schema_prompt_addition = read_schema('data/flight_database.schema')


def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-p', '--ptype', type=int, default=0,
                        help='Prompt type')
    parser.add_argument('-m', '--model', type=str, default='gemma',
                        help='Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')
    parser.add_argument('--sample_strategy', type=str, default="relevance", choices=["random_per", "random", "relevance"],
                        help="Whether to use a LR scheduler and what type to use if so")

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, sampled_examples):
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
    '''
    # TODO
    prompt = f"You are a state-of-the-art English-to-SQL translator. You are provided with information about the database schema to help you:\n"
    prompt += schema_prompt_addition
    if k == 0:
        prompt += f"Given the following text, generate the corresponding SQL query and always use full city names in the queries.\n\n"
        prompt += f"Text: {sentence}\n"
        prompt += "SQL: "
        return prompt
    else:
        prompt += f"\nYou are also provided with {k} examples of English sentences and their corresponding SQL queries to help you:\n"
        for i, example in enumerate(sampled_examples):
            prompt += f"Example {i+1}: {example[0]}. The corresponding SQL query is: {example[1]}\n"
        prompt += f"\nGiven the following text, generate the corresponding SQL query and always use full city names in the queries.\n\n"
        prompt += f"Text: {sentence}\n"
        prompt += "SQL: "
        return prompt

def pad_tensor(tensor_list, pad_length):
    """Pad the tensors in the list to the same length with zeros."""
    return [torch.cat([t, torch.zeros(pad_length - len(t), dtype=torch.float32)]) for t in tensor_list]


def exp_kshot(tokenizer, model, inputs, k, train_x, train_y, select_strategy, training_freq_map=None):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
    '''
    raw_outputs = []
    extracted_queries = []

    pbar = tqdm(enumerate(inputs), total=len(inputs), desc="Processing inputs")

    for i, sentence in pbar:
        extracted_query = ""
        j = 0
        pbar.set_description(f"Processing sentence {i+1}/{len(inputs)}")
        while not extracted_query and j < 5:
            if select_strategy == "random":
                sampled_examples = random.sample(list(zip(train_x, train_y)), k)
            elif select_strategy == "relevance":
                sentence_freqs = compute_freq_map(sentence)
                sentence_tensor = torch.tensor(sentence_freqs, dtype=torch.float32)
                # Determine the maximum length of frequency maps
                max_length = max(len(sentence_tensor), max(len(t) for t in training_freq_map))
                padded_sentence_tensor = pad_tensor([sentence_tensor], max_length)[0]
                padded_training_tensors = pad_tensor(training_freq_map, max_length)
                similarities = [F.cosine_similarity(padded_sentence_tensor.unsqueeze(0), freq.unsqueeze(0)) for freq in padded_training_tensors]
                similarities_tensor = torch.tensor(similarities)
                sampled_indices = torch.topk(torch.tensor(similarities_tensor), k).indices
                sampled_examples = [(train_x[i], train_y[i]) for i in sampled_indices]
            prompt = create_prompt(sentence, k, sampled_examples) # Looking at the prompt may also help
            # print(f"Prompt: {prompt}")
            # print("Tokenizing the prompt...")
            input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            # print("Calling model.generate...")
            outputs = model.generate(**input_ids, max_new_tokens=512, temperature=0.7, do_sample=True) # You should set MAX_NEW_TOKENS
            # print("Decoding the outputs...")
            response = tokenizer.decode(outputs[0]) # How does the response look like? You may need to parse it
            # print(f"Model response: {response}")
            raw_outputs.append(response)

            # Extract the SQL query
            extracted_query = extract_sql_query(response)
            if not extracted_query:
                j += 1

        extracted_queries.append(extracted_query)
    return raw_outputs, extracted_queries


def eval_outputs(queries, gt_sql_path, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluate the outputs of the model by computing the metrics.

    Add/modify the arguments and code as needed.
    '''
    # TODO
    save_queries_and_records(queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(gt_sql_path, model_sql_path, gt_record_path, model_record_path)
    error_rate = compute_error_rate(model_error_msgs)
    return sql_em, record_em, record_f1, model_error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    '''
    Args:
        * model_name (str): Model name ("gemma" or "codegemma").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)
    
    To access to the model on HuggingFace, you need to log in and review the 
    conditions and access the model's content.
    '''
    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        # Native weights exported in bfloat16 precision, but you can use a different precision if needed
        model = GemmaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
        ).to(DEVICE)
    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # 4-bit quantization
            )
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16,
                                                        config=nf4_config).to(DEVICE)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16).to(DEVICE)
    return tokenizer, model


def main():
    '''
    Note: this code serves as a basic template for the prompting task. You can but 
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    '''
    args = get_args()
    shot = args.shot
    ptype = args.ptype
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name
    select_strategy = args.sample_strategy

    set_random_seeds(args.seed)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)


    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)

    if select_strategy == "relevance":
        precomputed_freq_map = compute_freq_map(train_x)
        training_tensors = [torch.tensor(freqs, dtype=torch.float32) for freqs in precomputed_freq_map]

    for eval_split in ["dev", "test"]:
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)

        print(f"Running on {eval_split}...")

        if select_strategy == "relevance":
            raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x[:5], shot, train_x, train_y, select_strategy, training_tensors)
        
        elif select_strategy == "random":
            raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x[:5], shot, train_x, train_y, select_strategy)
        

        # You can add any post-processing if needed
        # You can compute the records with `compute_records``

        gt_sql_path = os.path.join(f'data/{eval_split}.sql')
        dev_gt_records = os.path.join(f'records/{eval_split}_gt_records.pkl')
        model_sql_path = os.path.join(f'results/gemma_{experiment_name}_dev.sql')
        model_record_path = os.path.join(f'records/gemma_{experiment_name}_dev.pkl')
        if eval_split == "dev":
            print("Evaluating model outputs...")
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                queries=extracted_queries,
                gt_sql_path=gt_sql_path,
                model_sql_path=model_sql_path,
                gt_record_path=dev_gt_records,
                model_record_path=model_record_path
            )
            print(f"{eval_split} set results: ")
            print(f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"{eval_split} set results: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        else:
            print("Saving test results...")
            model_sql_path = os.path.join(f'results/gemma_{experiment_name}_test.sql')
            model_record_path = os.path.join(f'records/gemma_{experiment_name}_test.pkl')
            save_queries_and_records(extracted_queries, model_sql_path, model_record_path)

        # Save logs, if needed
        log_path = "prompting_logs/logs.txt" # to specify
        save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)


if __name__ == "__main__":
    main()