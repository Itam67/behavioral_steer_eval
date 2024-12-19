import torch 

# Llama2 ONLY: Find the [/INST] token position
def mask_tokens(tokens):
    '''
    Masks the tokens before the [/INST] token

    Args:
        tokens (torch.Tensor): The tokenized data

    Returns:
        torch.Tensor: Mask that isolates continuations
    '''
    starts = []
    for batch in tokens:
        # 29914 index + 2 is end of the [/INST] token
        starts.append((batch == 29914).nonzero(as_tuple=True)[0].item()+2)

    # Mask the tokens
    range = torch.arange(tokens.shape[1]).expand(tokens.shape[0], -1)
    mask = range > torch.tensor(starts).unsqueeze(1)

    return mask.to('cuda')