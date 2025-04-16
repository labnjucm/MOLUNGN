# MOLUNGN
# A Multi-Omics Lung Cancer Graph Network for Biomarker Discovery and Lung Cancer Classification

## Overview

This repository provides the official implementation of the MOLUNGN model described in our recent study. MOLUNGN is a graph-attention-based deep learning framework developed to integrate multi-omics data (mRNA expression, miRNA expression, DNA methylation profiles) for accurate lung cancer staging classification and stage-specific biomarker identification.

Specifically, MOLUNGN integrates omics-specific Graph Attention Networks (OSGAT) and a Multi-Omics View Correlation Discovery Network (MOVCDN), providing a powerful computational approach to identify critical biomarkers associated with lung cancer progression and facilitating comprehensive systems biology analyses.

---

## Installation

To replicate the MOLUNGN computational environment, please use the provided `environment.yml` file, which contains all necessary dependencies and configurations.

### Step 1: Clone the Repository

```bash
git clone https://github.com/labnjucm/MOLUNGN.git
cd MOLUNGN 
```


### Step 2: Environment Setup (via Anaconda)
We highly recommend using Anaconda for environment management:

```bash
conda env create -f environment.yml
conda activate molungn
```
### Step 3: Verify Installation
Ensure all dependencies are correctly installed by running:
```bash
conda list
```

### Data Preparation
Our study utilized publicly available multi-omics data from LUAD and LUSC cohorts. You can obtain these datasets from:

The Cancer Genome Atlas (TCGA): https://portal.gdc.cancer.gov

Please follow these steps to preprocess the data:

Download original omics datasets (mRNA expression, DNA methylation, miRNA expression).

Preprocess data into normalized feature matrices according to the procedures described in our paper.

Place processed data into the data/ directory following this structure:

```bash
data/
├── LUAD/
│   ├── mRNA.csv
│   ├── miRNA.csv
│   └── DNA_methylation.csv
├── LUSC/
│   ├── mRNA.csv
│   ├── miRNA.csv
│   └── DNA_methylation.csv
└── clinical_labels.csv
```

Ensure your data follows exactly the formats provided in the sample files in this repository.

Running MOLUNGN
After data preparation, here is an example to run the MOLUNGN model with the following command:

```bash
python labels_divide.py 
```

Outputs and Results
Results including evaluation metrics, identified biomarkers, and model performance figures will be stored in the results/ directory automatically upon model completion:

Please refer to these results for biomarker information and quantitative evaluation metrics detailed in our published paper.

Repository Structure
```bash
MOLUNGN/
├── data_LUAD
├── data_LUSC
├── results/
├── environment.yml
├── labels_divide.py
└── README.md
```
### Contact and Support
For questions, concerns, or suggestions regarding MOLUNGN, please contact:

Corresponding Author: [Daifeng Zhang] ([zhangshiyu654@gmail.com])

Lab homepage: https://github.com/labnjucm

License
This project is licensed under the MIT License. See the LICENSE file for details.

We hope MOLUNGN provides valuable insights and supports your research on lung cancer classification and biomarker discovery!
