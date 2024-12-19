# Towards Reliable Evaluation of Behavior Steering Interventions in LLMs

This repository contains the code for the paper **"Towards Reliable Evaluation of Behavior Steering Interventions in LLMs"**. It includes:

1. Code for the novel benchmark introduced in the paper.
2. Tools to generate responses to prompts and evaluate the behavior of language models under interventions.

---

## Pipeline Overview

### Important Files
- **`experiments/calc_fine_tune.py`**: Computes likelihoods of continuations for fine-tuned models loaded from Hugging Face.
- **`experiments/calc_steer.py`**: Evaluates likelihoods of continuations for models with steering vectors applied.
- **`src/pipeline.py`**: Produces figures and metric data based on the computed likelihoods.

---

## How to Run the Pipeline

### Step 1: Configure Parameters
1. **Experiment Parameters**
   - Modify the cfg parameters at the bottom of `experiments/calc_fine_tune.py` or `experiments/calc_steer.py` according to your needs.
   - Set the output directory (behavior_name is a placeholder):
     ```python
     'output_dir':'results/behavior_name'
     ```

2. **Figure Parameters**
   - Update `src/pipeline.py` cfg parameters with the following paths:
     ```python
     'ctrl_like_dir':'results/behavior_name/behavior_name_control.pt'
     'exp_like_dir':'results/behavior_name/behavior_name_exp.pt'
     ```

---

### Step 2: Run the Scripts
- Navigate to the `scripts/` directory.
- Select and run the relevant script for your experiment. Outputs will be stored in the `results/behavior_name/` directory.

---

## Notes

- This repository is tailored for **Llama 2 7B**. If you plan to adapt it for other model families or tokenizers:
  - Update the **mask_tokens** function in `src/likelihood_utils.py` and ensure they align with the target tokenizer.
  - Review and update the **model loading steps** accordingly in the calc files.

---

## Citation
If you use this code, please cite the paper:

@misc{pres2024reliableevaluationbehaviorsteering,
      title={Towards Reliable Evaluation of Behavior Steering Interventions in LLMs}, 
      author={Itamar Pres and Laura Ruis and Ekdeep Singh Lubana and David Krueger},
      year={2024},
      eprint={2410.17245},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.17245}, 
}
