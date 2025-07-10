# simple_args

This repository contains the implementation of the ARGS algorithm, including environment setup, training and evaluation scripts for reward and cost models, and a guide to run the ARGS inference.

## Table of Contents

1. [Setup](#setup)
2. [Reward & Cost Model Training](#reward--cost-model-training)
3. [Reward & Cost Model Evaluation](#reward--cost-model-evaluation)
4. [Running ARGS](#running-args)

## Setup

### 1. Create Conda Environment

```bash
# Create a new conda environment
conda create -n args_env python=3.15.3 -y
conda activate args_env
```

2. Install Dependencies

```bash
# Using requirements.txt
pip install -r requirements.txt
```



Reward & Cost Model Training

1. Train Reward Model

```bash
python rm_train.py 
```

3. Train Cost Model

```bash
python cm_train.py
```

Reward & Cost Model Evaluation

1. Evaluation Script

```bash
python rm_eval.py
python cm_eval.py
```

2. Evaluation Metrics
	•	Reward Model: Accuracy
	•	Cost Model: Accuracy

Running ARGS

1. Inference Script

```bash
python args.py
```

2. Output
	•	Generated responses will be saved in output/sft_vs_args.csv

⸻


