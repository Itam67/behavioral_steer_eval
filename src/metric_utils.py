import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_metric(ctrl, exp, title, ctrl_name, exp_name):
    # First half represents behavior matching continuation
    # Second half represents behavior not matching continuation
    
    # Baseline
    ctrl_pos = ctrl[:int(ctrl.shape[0]/2)]
    ctrl_neg = ctrl[int(ctrl.shape[0]/2):]

    # Experimental
    exp_pos = exp[:int(exp.shape[0]/2)]
    exp_neg = exp[int(exp.shape[0]/2):]

    # Sort both control groups
    ctrl_pos_ind = torch.argsort(ctrl_pos)
    ctrl_neg_ind = torch.argsort(ctrl_neg)
    ctrl_srt = torch.cat([ctrl_neg[ctrl_neg_ind], ctrl_pos[ctrl_pos_ind]])

    # Rearrange experimental group to match the control sorting
    exp_srt = torch.cat([exp_neg[ctrl_neg_ind], exp_pos[ctrl_pos_ind]])

    # Rescale
    ctrl_srt = (-1 * ctrl_srt) / ctrl_srt.max()
    exp_srt = (-1 * exp_srt) / exp_srt.max()

    # Set plot limits
    plt.xlim(-2,ctrl.shape[0]+1)
    plt.ylim(torch.cat([ctrl_srt,exp_srt]).min()-0.2,torch.cat([ctrl_srt,exp_srt]).max()+0.2)

    # Calculate Quartiles
    negative_samples = ctrl_srt[:len(ctrl_srt)//2]  # Get top negative side
    positive_samples = ctrl_srt[len(ctrl_srt)//2:]  # Bottom Positive side

    # Select the first 25% 50% 75% of indices
    neg_bounds = [np.percentile(negative_samples.numpy(), 75), max(negative_samples).item()]
    pos_bounds = [min(positive_samples).item(), np.percentile(positive_samples.numpy(), 25)]

    cmap = ListedColormap([   '#ffeb99'])  # You can modify these colors
    plt.fill_betweenx(neg_bounds, -2, negative_samples.shape[0]-1, color=cmap(0), alpha=0.5)
    plt.fill_betweenx(pos_bounds, negative_samples.shape[0]-1, ctrl.shape[0]+1, color=cmap(0), alpha=0.5)

    # Create the scatter plot
    plt.scatter(np.arange(exp.shape[0]), ctrl_srt, color='blue', label=ctrl_name, s=50, alpha=0.8)
    plt.scatter(np.arange(exp.shape[0]), exp_srt, color='red', label=exp_name, s=50, alpha=0.8)

    # Add a dotted vertical line
    plt.axvline(x=negative_samples.shape[0]-1, color='black', linestyle='--', linewidth=1)

    plt.legend(loc='upper left')

    # Show the plot
    plt.savefig(f'results/{title}/{title}.png')

    plt.show()



def calc_steer_metric(ctrl, exp):


  # Split and renormalize into behavior groups

  ctrl_pos = ctrl[:int(ctrl.shape[0]/2)] / (-1*ctrl.max())
  ctrl_neg = ctrl[int(ctrl.shape[0]/2):] / (-1*ctrl.max())

  exp_pos = exp[:int(exp.shape[0]/2)] / (-1*exp.max())
  exp_neg = exp[int(exp.shape[0]/2):] / (-1*exp.max())

  # Sort groups based on control
  ctrl_pos_ind = torch.argsort(ctrl_pos)
  ctrl_neg_ind = torch.argsort(ctrl_neg, descending=True)

  ctrl_pos = ctrl_pos[ctrl_pos_ind]
  ctrl_neg = ctrl_neg[ctrl_neg_ind]

  exp_pos = exp_pos[ctrl_pos_ind]
  exp_neg = exp_neg[ctrl_neg_ind]

  match_diff = exp_pos - ctrl_pos
  neg_diff = ctrl_neg - exp_neg

  # Measure metric include more and more quartiles from Q4 to Q1
  scores = []
  for i in range(1,5):
    match_sub = match_diff[:int(match_diff.shape[0] * i/4)]
    neg_sub = neg_diff[:int(neg_diff.shape[0] * i/4)]


    # Give credit for positive difference in match and negative difference in non-match
    scores.append((round(match_sub.mean().item(),4), round(neg_sub.mean().item(),4)))

  return scores