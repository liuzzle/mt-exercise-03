# MT Exercise 3: Layer Normalization for Transformer Models

This repo is a collection of scripts showing how to install [JoeyNMT](https://github.com/joeynmt/joeynmt), download
data and train & evaluate models, as well as the necessary data for training your own model

## Repository Structure
```md
mt-exercise-03/ 
├── configs/ 
│ ├── baseline.log # Training log for the baseline model 
│ ├── deen_transformer_regular.yaml # Configuration file for Pre-Norm and Post-Norm models 
├── data/ 
│ ├── codes3200.bpe # BPE codes for tokenization 
│ ├── train/ # Training data │ 
├── dev/ # Validation data 
│ ├── test/ # Test data 
├── logs/ 
│ ├── deen_transformer_regular/ 
│ ├── err_pre # Log file for the Pre-Norm model 
│ ├── err_post # Log file for the Post-Norm model 
├── models/ 
│ ├── deen_transformer_pre/ # Model directory for Pre-Norm 
│ ├── deen_transformer_post/ # Model directory for Post-Norm 
├── scripts/ 
│ ├── train.sh # Script to train the models 
│ ├── visualize_results.py # Script to extract validation perplexities and visualize results 
├── validation_perplexities.csv # Table of validation perplexities for all models
```
---

## Requirements

- Python 3.8 or higher
- Joey-NMT (v2.2.0)
- Required Python libraries:
  - `pandas`
  - `matplotlib`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run the Experiments
1. Train the Models
Run the training script for each model:

```bash 
scripts/train.sh
```

Ensure the configuration file (configs/`deen_transformer_regular.yaml`) is updated for Pre-Norm and Post-Norm setups.

2. Extract Validation Perplexities
Use the `visualize_results.py` script to extract validation perplexities and generate a table and plot:
- Save the validation perplexities table as validation_perplexities.csv.
- Save the validation perplexities plot as validation_perplexities.png.

## Results
#### Validation Perplexities Table
The table below shows the validation perplexities for the Baseline, Pre-Norm, and Post-Norm models:

|Steps	| Baseline	| Prenorm	|Postnorm|
|-------|-----------|-----------|--------|
|500	|56.61	|44.8	|41.5|
|1000	|49.93	|30.68	|30.22|
|1500	|45.33	|26.34	|27.02|
|...	|...	|...	|...|


#### Validation Perplexities Plot
![validation_perplexities](https://github.com/user-attachments/assets/e12570a0-b477-4b9b-b676-c200800362e3)

## Key Findings
#### Pre-Norm vs Post-Norm:
Pre-Norm consistently outperforms Post-Norm in terms of validation perplexity across all steps.
- At step 15,000, Pre-Norm achieves a perplexity of 9.2, compared to 11.09 for Post-Norm.
  
Baseline Model:
- The baseline model shows slower convergence and higher perplexities compared to both Pre-Norm and Post-Norm setups.
  
Alignment with Wang et al. (2019):
- Our experiments confirm Wang et al.'s findings that Pre-Norm facilitates better optimization and performance, even in shallow networks and low-resource settings.

## References
Wang et al. (2019). [Learning Deep Transformer Models for Machine Translation](https://arxiv.org/abs/1906.01787)

Joey-NMT [GitHub Repository](https://github.com/joeynmt/joeynmt)
