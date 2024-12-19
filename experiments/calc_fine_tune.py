import torch 
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.likelihood_utils import mask_tokens

    
def get_model_like(model, tokenizer, data):
    '''
    Calculate the log likelihood of the data under normal/fine-tuned model

    Args:
        model (transformers.PreTrainedModel): The model
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model
        data (list): The data to calculate likelihoods

    Returns:
        list: The log likelihoods of the data
    '''

    # Tokenize data
    inputs = tokenizer(data, return_tensors='pt', padding=True).to('cuda')

    likelihoods = []

    # Run the model while hooking with the steer vector
    for i in range(0, len(data), 20):

        # Print progress
        print(f'Calculating {i} of {len(data)}')

        # Get the current batch
        sub_inputs = {k: v[i:i+20] for k, v in inputs.items()}

        # Generate sequences in batches
        with torch.no_grad():
            outputs = model(sub_inputs['input_ids'], attention_mask=sub_inputs['attention_mask'], labels=sub_inputs['input_ids'])

  
        # Calculate experimental log likelihood of continuation

        # Get the mask for the tokens
        unpadded_indices =  mask_tokens(sub_inputs['input_ids'])

        # Get the correct indices to meausre
        shift_unpadded_indices = unpadded_indices[..., 1:].contiguous()
        shift_labels = sub_inputs['input_ids'][..., 1:].contiguous()
        
        # Calculate the log likelihood
        exp_ll = outputs.logits[:,:-1].softmax(-1).log().gather(-1, shift_labels.unsqueeze(-1)).flatten(-2) * shift_unpadded_indices
        likelihoods += exp_ll.mean(-1).tolist()

    return likelihoods


def main():

    # Load data and system prompts
    with open(cfg['data_dir'], 'r') as f:
        data = json.load(f)
    data = [ex['question'] for ex in data]

    # Get the likelihoods for the two models
    total_like = []
    for model_name in [cfg['control_model'], cfg['exp_model']]:

        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(cfg['control_model'], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        # Set the seed
        torch.manual_seed(cfg['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg['seed'])

        # Partition data into neg and pos
        partioned_data = [data[i:i+len(data)//2] for i in range(0, len(data), len(data)//2)]

        likelihoods = []

        # Get the likelihoods for each partition
        for part in partioned_data:

            # Get the steered likelihoods
            likelihoods+=get_model_like(model, tokenizer, part)

        total_like.append(likelihoods)


    control_likelihoods = total_like[0]
    tuned_likelihoods = total_like[1]

    # Save the likelihoods for each partition as toch tensors
    torch.save(torch.tensor(control_likelihoods), cfg['output_dir'] + '_control.pt')
    torch.save(torch.tensor(tuned_likelihoods), cfg['output_dir'] + '_exp.pt')

cfg = {
    'control_model': 'meta-llama/Llama-2-7b-chat-hf',
    'exp_model': 'likenneth/honest_llama2_chat_7B',
    'data_dir': 'results/generated_data/hallucination.json',
    'output_dir': 'results/steeredness/truthful',
    'seed': 42
}

if __name__ == '__main__':
    main()


