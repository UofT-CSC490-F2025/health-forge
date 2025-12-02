# HealthForge

Healthforge is a synthetic electronic health record generator trained on MIMIC-IV. This repository contains the training and sampling code for the model.

The link to the report documenting the architecture, problem statement, and use cases of the model can be found here:

The training for this project was done exclusively using Modal with the training data and results being stored on an AWS S3 instance.

The main training loop function (train_from_pkl) located in healthforge/final_project/train.py, is designed to take the config, initialized model, training data, text embeddings, and paths to the trained autoencoder (see docstring for more details). It can be run completely locally, however the script to initiate the model and download the necessary files are not included in this repository.

If you want to train using Modal and an S3 (recommended), here is how you can train and sample using our existing scripts.

Installing requirements:

> > cd health_forge/final_project
> > pip install -r requirements.txt

Training Diffusion Model:

1. Set up Modal CLI and AWS S3
   - Modal Documentation: https://modal.com/docs/guide
   - AWS Documentation: https://aws.amazon.com/getting-started/
   - You can refer to health_forge/final_project/modal_train_app.py for our bucket and file names, it may be a good idea to change those to fit the structure of your S3.
2. Upload training data and autoencoder model to S3

   - Training Data we used can be found here: (Google Drive Link)
   - IMPORTANT: It is derived from MIMIC-IV, please follow MIMIC's license and cite the original MIMIC source.

3. Configure your model
   - Modify the health_forge/final_project/config.yaml file to your liking (they have our configuration currently)
4. In health_forge/final_project
   > > modal run ./modal_train_app.py
5. The best model checkpoint along with latent data mean and standard deviation will be uploaded to your S3.

Sampling (Local):

1. Download the trained diffusion .pt file from S3 (or your preferred source)
2. Download the trained autoencoder .pt file from S3 (or your preferred source)
3. Download the latent mean and standard deviation files from S3 (or your preferred source)
4. Place all files above in health_forge/final_project
5. Run the sampling script:
   > > python run sample.py config_path model_path

Training Autoencoder Model:

1. Set up Modal CLI and AWS S3
   - Modal Documentation: https://modal.com/docs/guide
   - AWS Documentation: https://aws.amazon.com/getting-started/
   - You can refer to health_forge/final_project/modal_train_app.py for our bucket and file names, it may be a good idea to change those to fit the structure of your S3.
2. Upload patient vector data to base directory in S3

   - Training Data we used can be found here: (Google Drive Link)
   - IMPORTANT: It is derived from MIMIC-IV, please follow MIMIC's license and cite the original MIMIC source.

3. In health_forge/final_project
   > > modal run ./modal_train_app.py
4. The best model checkpoint along with latent data mean and standard deviation will be uploaded to your S3.

<!-- Pytest Coverage Comment:Begin -->
<!-- Pytest Coverage Comment:End -->
