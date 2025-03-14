# Universality of Truth Representation: A Study of Llama3 and DeepSeek-R1
This repository contains the code and datasets necessary to reproduce the results presented in our paper "Universality of Truth Representation: A Study of Llama3 and DeepSeek-R1", which extends the work from the NeurIPS 2024 paper "Truth is Universal: Robust Detection of Lies in LLMs" by Bürger et al.

### Overview
### Our project:

- Reproduces the TTPD (Training of Truth and Polarity Direction) results on LLaMA3-8B

- Extends the TTPD framework to DeepSeek R1-distill-llama-8B

- Compares both models to analyze architectural influences on truth encoding

- Evaluates generalization performance across different statement types

### Setup
```bash
# Create and activate a new Python environment
conda create --name truth_representation python=3.11
conda activate truth_representation
```
### Clone the repository and install requirements
```bash
git clone https://github.com/Nikta-Kiani/universality-of-truth.git
cd universality-of-truth
pip install -r requirements.txt
```
### Generating Activations
This repository provides all datasets used in the paper, but not the associated activation vectors due to their large size. You'll need to generate these activations before running any code:

### For LLaMA3-8B:
```bash
python generate_acts.py --model_family Llama3 --model_size 8B --model_type chat --layers 12 --datasets all_topic_specific --device cuda -l info
```
### For DeepSeek R1-distill-llama-8B:
```bash
python generate_acts.py --model_family r1 --model_size 8B --model_type chat --layers 12 --datasets all_topic_specific --device cuda -l info
```
### For layer-wise analysis:

```bash
python generate_acts.py --model_family Llama3 --model_size 8B --model_type chat --layers -1 --datasets cities neg_cities sp_en_trans neg_sp_en_trans --device cuda -l info
python generate_acts.py --model_family r1 --model_size 8B --model_type chat --layers -1 --datasets cities neg_cities sp_en_trans neg_sp_en_trans --device cuda -l info
#Note: At least 16GB of GPU RAM is required to run each of these models.
```

### Repository Structure
### Main Scripts
- generate_acts.py: For generating activations for both models

- utils.py: Helper functions for loading activations and data processing

- probes.py: Various classifiers (TTPD, LR, CCS, MM) for true/false statement detection

- config.ini: Configuration file for model paths

- truth_directions.py: Reproduces the core TTPD analysis, projections onto the truth subspace, and layer-wise analysis for both models

### Output Folders
- outputs/llama3/: Contains results and visualizations for LLaMA3

- outputs/r1/: Contains results and visualizations for DeepSeek R1

### Results
Our findings confirm that:

- Both LLaMA3-8B and DeepSeek R1-distill-llama-8B encode truthfulness in a two-dimensional subspace

- Layer 12 contains the strongest truth signal in both models

- TTPD consistently outperforms baseline methods (LR, CCS, MM) across different statement types

- PCA analysis validates the two-dimensional nature of the truth subspace

## Credits
The core methodology is based on the paper ["Truth is Universal: Robust Detection of Lies in LLMs"](https://arxiv.org/abs/2407.12831) by Bürger et al.

All datasets and methods are derived from the [Truth is Universal GitHub repository](https://github.com/sciai-lab/Truth_is_Universal).

The `DataManager` class, `generate_acts.py` script, and CCS implementation are adapted from the [Geometry of Truth](https://github.com/saprmarks/geometry-of-truth) GitHub repository by Samuel Marks.

The datasets were primarily collected from the original paper.


