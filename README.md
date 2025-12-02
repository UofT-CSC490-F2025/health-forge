# HealthForge

HealthForge is a synthetic electronic health record (EHR) generator trained on the MIMIC-IV dataset. This repository contains the full training and sampling pipeline for the model.

A detailed report describing the architecture, problem statement, and use cases can be found here:  
**https://drive.google.com/file/d/1BOO-jEnDC6tqnNZ0St-O6AglnjoBW4yh/view?usp=sharing**

Training for this project was performed using Modal, with datasets and checkpoints stored in AWS S3.

The core training function, `train_from_pkl`, located at `health_forge/final_project/train.py`, takes a configuration dictionary, model initialization, EHR vectors, text embeddings, and the autoencoder checkpoint. It can be run fully locally, but scripts for local model initialization and file downloading are not included in this repository.

If you want to train using Modal and S3 (recommended), follow the steps below.

---

## Installation

```bash
cd health_forge/final_project
pip install -r requirements.txt
```

---

## Training the Diffusion Model

### 1. Set up Modal CLI and AWS S3

- Modal documentation: https://modal.com/docs/guide
- AWS documentation: https://aws.amazon.com/getting-started/
- Refer to `health_forge/final_project/modal_train_app.py` for the bucket and file names used in our setup. You may want to modify these to match your own S3 structure.

### 2. Upload training data and the autoencoder model to S3

- Training data used in this project can be downloaded here: https://drive.google.com/file/d/1JfUhMCgSzXCtnPw4pvrW3EqxwwzplF7-/view?usp=sharing
- **Important:** The dataset is derived from MIMIC-IV. Please follow the MIMIC license requirements and cite the original dataset.

### 3. Configure your model

Modify the configuration in `health_forge/final_project/config.yaml` to adjust hyperparameters or model settings. The existing configuration reflects those used in our experiments.

### 4. Run the training job using Modal

```bash
modal run ./modal_train_app.py
```

### 5. Outputs

The best model checkpoint, along with latent mean and standard deviation files, will be uploaded to your S3 bucket.

---

## Sampling (Local)

1. Download the following files from S3 (or your preferred storage):

   - Diffusion model `.pt` file
   - Autoencoder `.pt` file
   - `latent_mean.npy`
   - `latent_std.npy`

2. Place all downloaded files into `health_forge/final_project`.

3. Run the sampling script:

```bash
python sample.py config_path model_path
```

---

## Training the Autoencoder Model

### 1. Set up Modal CLI and AWS S3

Follow the setup instructions above for Modal and S3.  
You may edit paths in `modal_train_autoencoder.py` to match your bucket structure.

### 2. Upload patient vector data to the S3 base directory

- Training data used in this project can be downloaded here: https://drive.google.com/file/d/1JfUhMCgSzXCtnPw4pvrW3EqxwwzplF7-/view?usp=sharing
- **Important:** This dataset is derived from MIMIC-IV. Follow license and citation requirements.

### 3. Run autoencoder training

```bash
modal run ./modal_train_autoencoder.py
```

### 4. Outputs

The best autoencoder checkpoint will be uploaded to your S3 bucket.

<!-- Pytest Coverage Comment:Begin -->

<!-- Pytest Coverage Comment:End -->
