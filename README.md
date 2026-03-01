# vibes_pipe_mre_seg
deep learning segmentation pipeline
=======
# Vibes-pipe
## Reproducible Data Pipeline for MRE Brain Segmentation 
A structured, scanner-aware preprocessing and training pipeline for Magnetic Resonance Elastography segmentation. The model is able to learn from NLI outputs and retrain the segmentation. 

This project consists of:
1. Dataset freezing
2. Configuration preprocessing (YAML-driver)
3. Training Execution 


## Project Structure 
vibes_pipe/
├── configs/
│   ├── config.yaml
│   └── preprocess.yaml
├── experiments/
│   └── mre_data_prep_test/
│       ├── pairs.json
│       ├── raw_data/
│       │   ├── G009/  (G009_*.mat, G009_*.nii)
│       │   ├── G039/  (G039_*.mat, G039_*.nii)
│       │   └── S001/  (S001_*.mat, S001_*.nii)
│       └── workspace_root/
│           ├── manifest.json
│           ├── train/
│           │   ├── G009/  (X.mat, GT.mat, NLI_output.mat)
│           │   ├── G039/  (X.mat, GT.mat, NLI_output.mat)
│           │   ├── S001/  (X.mat, GT.mat, NLI_output.mat)
│           │   └── subj01/ (X.mat, GT.mat, NLI_output.mat)
│           └── val/
│               └── subj02/ (X.mat, GT.mat)
├── src/
│   └── vibes_pipe/
│       ├── cli/
│       │   ├── pipeline_cli.py
│       │   ├── helpers_data_prep.py
│       │   └── import_subject.py
│       ├── data/
│       │   ├── dataset.py
│       │   ├── preprocess.py
│       │   ├── io_mat.py
│       │   ├── manifest.py
│       │   └── make_pairs_from_subject_folders.py
│       └── utils/
│           └── config.py
├── notebooks/
├── tests/
│   ├── test_cli.py
│   └── prep_test.json
├── README.md
└── requirements.txt

## Vibes Lab Data Maintenance (In Progress)
To ensure the pipeline work seamlessly with the 
ac6ea44 (Upgrade clarity and reduce duplication in data preparation logic. Added functions to extract spacing in nifti file and embeded the spacing info into data preprocessing.)
