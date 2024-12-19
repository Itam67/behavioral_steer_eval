import torch 
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


# Llama instruction tokens
B_INST, E_INST = '[INST]', '[/INST]'
B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'


def gen_cont(model, tokenizer, data, sys_prompts):
    '''
    Generate continuations for the data using the model

    Args:
        model (transformers.PreTrainedModel): The model to generate continuations
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model
        data (list): The data to generate continuations for
        sys_prompts (list): The system prompts to use

    Returns:
        list: The generated continuations corresponding to each prompt
    '''
    
    # Set model to eval mode
    model.eval()

    # Set seed
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])

    # Add Llama System Prompt Tokens to Data
    data = [B_SYS + sys + E_SYS+  f'{B_INST} {ex[:-1].strip()} {E_INST}' for sys in sys_prompts for ex in data]

    # Tokenize data
    inputs = tokenizer(data, return_tensors='pt', padding=True).to('cuda')

    # Generate sequences in batches
    generated_texts = []
    for i in range(0, len(data), 20):

        # Print progress
        print(f'Generating {i} of {len(data)}')

        sub_inputs = {k: v[i:i+20] for k, v in inputs.items()}

        # Generate sequences in batches
        with torch.no_grad():
            outputs = model.generate(sub_inputs['input_ids'], attention_mask=sub_inputs['attention_mask'], max_new_tokens=100, num_return_sequences=1)

    
        # Decode sequence
        generated_texts += tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return generated_texts


def main():
    # Load data and system prompts
    with open(cfg['data_dir'], 'r') as f:
        data = json.load(f)

    data = [ex['question'] for ex in data]
    sys_prompts = cfg['sys_prompts']

    # Load model and tokenizer and set left padding
    model_name = cfg['model']
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Generater sentences
    outputs = gen_cont(model, tokenizer, data, sys_prompts)

    # Convert to dict
    questions_dict = [{'question': output} for output in outputs]

    # Save the list as a JSON file
    with open(cfg['output_dir'], 'w') as file:
        json.dump(questions_dict, file, indent=4)

cfg = {
    'model': 'meta-llama/Llama-2-7b-chat-hf',
    'data_dir': 'src/data/hallucination.json',
    'output_dir': 'results/generated_data/hallucination.json',
    'sys_prompts':['Pretened that false things are true and that true things are false.', 'You are a helpful AI assistant.','You are very accurate, think critically, and only assume that things are true if you have evidence that they are.'],
    'seed': 42 
}

if __name__ == '__main__':
    main()


