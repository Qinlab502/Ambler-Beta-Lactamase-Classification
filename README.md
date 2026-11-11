# Ambler Beta-Lactamase Classification

This project uses ESM-2 embeddings and five classifiers (MLP, CNN, LSTM, SVM, and Random Forest) to predict Ambler classes for beta-lactamase protein sequences. The ensemble uses soft voting by default.

## Project structure

```text
.
├── data/
│   └── ambler_dataset.fasta       # Training dataset
├── model/
│   ├── ambler_models/             # Model, training, and prediction code
│   ├── train_cnn.py
│   ├── train_lstm.py
│   ├── train_mlp.py
│   ├── train_rf.py
│   ├── train_svm.py
│   └── predict_ensemble.py
├── pretrained_model/
│   └── esm2_t33_650M_UR50D/       # Download ESM-2 weights here
├── results/                       # Existing prediction results
└── requirement.yaml               # Conda environment
```

## Setup

Create and activate the Conda environment from the project root:

```bash
conda env create -f requirement.yaml
conda activate ambler-classifier
```

Download the ESM-2 model `facebook/esm2_t33_650M_UR50D` from Hugging Face and place its model files in:

```text
pretrained_model/esm2_t33_650M_UR50D/
```

The model files are not included in this repository.

## Train models

Run commands from the project root. For example:

```bash
python .\model\train_cnn.py
python .\model\train_lstm.py
python .\model\train_mlp.py
python .\model\train_rf.py
python .\model\train_svm.py
```

Each script runs five-fold cross-validation. Checkpoints and evaluation reports are written under the corresponding `model/runs_*` directory. Deep learning folds contain `best_model.pt`; SVM and Random Forest folds contain `model.pkl`.

## Predict with the ensemble

### Use the author-provided weights

To run prediction without training the classifiers, first download the trained weights from the [Zenodo record](https://doi.org/10.5281/zenodo.24110438) (DOI: `10.5281/zenodo.24110438`) and extract them into the corresponding directories under `model/`:

```text
model/
├── runs_mlp/
├── runs_cnn/
├── runs_lstm/
├── runs_svm/
└── runs_rf/
```

Keep the fold subdirectories and checkpoint filenames from the archive. The prediction script expects `best_model.pt` for MLP/CNN/LSTM and `model.pkl` for SVM/RF. Also download the ESM-2 model as described in [Setup](#setup); the Zenodo classifier weights do not replace the ESM-2 backbone.

Then provide a FASTA file:

```bash
python .\model\predict_ensemble.py --input .\data\ambler_dataset.fasta
```

By default, the script uses soft voting, looks for models in `model/runs_mlp`, `model/runs_cnn`, `model/runs_lstm`, `model/runs_svm`, and `model/runs_rf`, and saves `predictions.json` and `predictions.tsv` in `model/ensemble_results/`. Run the script with `--help` to see options for overriding model and output paths.

### Train your own weights

Alternatively, run the five training scripts above. They save checkpoints and evaluation reports under the matching `model/runs_*` directories.
