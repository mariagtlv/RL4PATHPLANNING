# RL4PATHPLANNING

## Overview

The aim of this research is to initiate the study of **Network Architecture Search (NAS)** techniques and **Large-Scale Multiobjective Optimization Problems (LSMOP)**, with the goal of integrating both fields to develop what we refer to as **Evolutionary Large-Scale Multiobjective Network Architectures**.

This repository brings together and extends several well-known NAS approaches, combining ideas from reinforcement learning, evolutionary methods, and differentiable architecture search.

---

## Related Repositories

This work is based on and inspired by the following repositories:

- **HA-ENAS**  
  https://github.com/BIMK/HA-ENAS

- **REP**  
  https://github.com/fyqsama/REP

- **DARTS**  
  https://github.com/quark0/darts

- **NAS-RL** *(not the original implementation, but follows the methodology described in the paper)*  
  https://github.com/ajayn1997/Neural-Architecture-Search-using-Reinforcement-Learning

---

## Repository Structure

### `/referencias`

Contains NAS-related methods that were considered during the development of this project.  
This folder stores the scientific articles associated with each method.

---

### `/HA-ENAS`

Implementation of the HA-ENAS method.

- **Architecture search**
  - Run `search/cifar10_search.py` to search for the best architectures.

- **Training**
  - Run `validation/hanet_train.py` to train the best architecture found during the search phase.

- **Testing**
  - Run `validation/hanet_test.py` to evaluate the trained architecture.

- **Results**
  - Stored in:
    - `results/results.xlsx`
    - `/analysis_output`

---

### `/REP-main`

Implementation based on the REP framework.

#### `/CNN`

- **Search**
  - Run `train_search.py` to perform the architecture search process.

- **Training**
  - Run `adv_train.py` to train the architectures found during the search.

#### `/GNN`

- **Search**
  - Run `train_search.py` to perform the architecture search process.

- **Training / Fine-tuning**
  - Run `train4tune.py` to train and fine-tune the searched architectures.

#### Results

- All experimental results are stored in the `/rep_results` directory.
  
### DARTS

#### `/DARTS`

Implementation based on **Differentiable Architecture Search (DARTS)**.

This module follows the original DARTS pipeline, which separates the search and evaluation phases using proxy models and weight sharing.

#### Results

- All experimental results are stored in the `/results` directory.

#### Usage

All commands are executed from the `DARTS` directory unless otherwise specified.

**1. Architecture Search (Proxy Models)**

Architecture search is performed using smaller proxy networks on CIFAR-10.

```bash
cd cnn
python train_search.py --unrolled
```

**2. Architecture Evaluation (Full Model)**

Once the architecture is selected, it is trained from scratch using the full-sized model.

```bash
cd cnn
python train.py --auxiliary --cutout
```

**3. Evaluation of a Pretrained DARTS Model**

A pretrained DARTS model can be evaluated on CIFAR-10 using:

```bash
cd cnn
python test.py --auxiliary --model_path cifar10_model.pt
```

---

### NAS-RL

#### `/NAS-RL`

Implementation of **Neural Architecture Search using Reinforcement Learning**, following the methodology described in the original NAS-RL literature (controller–child network framework).

This implementation extends the original approach by incorporating additional evaluation metrics such as F1-score and robustness-aware rewards.

#### Results

- All experimental results are stored in `NAS-RL/results`.
- During execution, intermediate results are also saved as `controller_search_results.parquet` and `child_training_results.parquet`. These files contain architecture genotypes, validation accuracy, F1-score, and reward values.

#### Docker Support

The NAS-RL module includes a **Dockerfile**, **docker-compose.yml**, and a dedicated `requirements.txt` file to ensure reproducibility.

#### Build the Docker image

```bash
docker compose build
```

or, if using older Docker versions:

```bash
docker-compose build
```

#### Run the container

```bash
docker compose up
```

or:

```bash
docker-compose up
```

This will start the container with all required dependencies installed.

#### Running NAS-RL

Once inside the container (or if running locally with the correct environment):

```bash
python train.py
```
---

## Notes

* Each NAS approach (HA-ENAS, REP, DARTS, NAS-RL) follows its original methodological assumptions, but has been adapted to support multi-objective evaluation and large-scale experimental analysis.
* Results across methods are stored separately to facilitate reproducibility and comparative analysis.

## Requirements

Install dependencies using:

```bash
pip install -r file_name.txt
```

- requirements.txt: general requirements.
- requirements_ha_enas.txt: requirements for HA-ENAS.
- requirements_rep.txt: requirements for REP.
- requirements_darts.txt: requirements for DARTS.
- requirements_nas_rl.txt: requirements for NAS_RL.
