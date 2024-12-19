import torch
from src.metric_utils import plot_metric, calc_steer_metric

def main():
    # Load data and system prompts
    control = torch.load(cfg['ctrl_like_dir'])
    experimental = torch.load(cfg['exp_like_dir'])

    # Plot results and save them
    plot_metric(control, experimental, cfg['plt_title'], cfg['ctrl_group_name'], cfg['exp_group_name'])
    
    # Calculate metric results and save them
    steer_metric_result = calc_steer_metric(control, experimental)

    f = open(f'results/{cfg['plt_title']}/{cfg['plt_title']}.txt', 'w')
    for t in steer_metric_result:
        line = ' '.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()

cfg = {
    'ctrl_like_dir': 'results/corrigible/corrigible_control.pt',
    'exp_like_dir': 'results/corrigible/corrigible_exp.pt',
    'ctrl_group_name': 'Llama 2 7B',
    'exp_group_name': 'Corrigible Steer Llama 2',
    'plt_title': 'Corrigble Steering CAA'
}

if __name__ == '__main__':
    main()