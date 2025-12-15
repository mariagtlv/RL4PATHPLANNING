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

## Requirements

Install dependencies using:

```bash
pip install -r file_name.txt
```

- requirements.txt: general requirements.
- requirements_ha_enas.txt: requirements for HA-ENAS.
