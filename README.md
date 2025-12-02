# HealthForge

HealthForge is a synthetic electronic health record (EHR) generator trained on the MIMIC-IV dataset. This repository contains the full training and sampling pipeline for the model.

A detailed report describing the architecture, problem statement, and use cases can be found here:  
**[Insert link]**

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

<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/README.md"><img alt="Coverage" src="https://img.shields.io/badge/Coverage-80%25-green.svg" /></a><details><summary>Coverage Report </summary><table><tr><th>File</th><th>Stmts</th><th>Miss</th><th>Cover</th><th>Missing</th></tr><tbody><tr><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_utils.py">data_utils.py</a></td><td>56</td><td>2</td><td>96%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_utils.py#L19">19</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_utils.py#L21">21</a></td></tr><tr><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_sample_app.py">modal_sample_app.py</a></td><td>73</td><td>73</td><td>0%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_sample_app.py#L1-L168">1&ndash;168</a></td></tr><tr><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_train_app.py">modal_train_app.py</a></td><td>73</td><td>73</td><td>0%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_train_app.py#L1-L146">1&ndash;146</a></td></tr><tr><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_train_app_unguided.py">modal_train_app_unguided.py</a></td><td>68</td><td>68</td><td>0%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_train_app_unguided.py#L1-L146">1&ndash;146</a></td></tr><tr><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_train_autoencoder.py">modal_train_autoencoder.py</a></td><td>121</td><td>7</td><td>94%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_train_autoencoder.py#L58">58</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_train_autoencoder.py#L200">200</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_train_autoencoder.py#L208-L210">208&ndash;210</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/modal_train_autoencoder.py#L213-L214">213&ndash;214</a></td></tr><tr><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/sample.py">sample.py</a></td><td>105</td><td>18</td><td>83%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/sample.py#L21-L22">21&ndash;22</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/sample.py#L145-L166">145&ndash;166</a></td></tr><tr><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/train.py">train.py</a></td><td>73</td><td>73</td><td>0%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/train.py#L1-L172">1&ndash;172</a></td></tr><tr><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/trainer.py">trainer.py</a></td><td>91</td><td>5</td><td>95%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/trainer.py#L82-L87">82&ndash;87</a></td></tr><tr><td colspan="5"><b>data_processing</b></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/EHR_data_processing_modal_final.py">EHR_data_processing_modal_final.py</a></td><td>211</td><td>38</td><td>82%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/EHR_data_processing_modal_final.py#L49">49</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/EHR_data_processing_modal_final.py#L173">173</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/EHR_data_processing_modal_final.py#L209">209</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/EHR_data_processing_modal_final.py#L327-L372">327&ndash;372</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/EHR_data_processing_modal_final.py#L377-L378">377&ndash;378</a></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/export_column_labels.py">export_column_labels.py</a></td><td>124</td><td>38</td><td>69%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/export_column_labels.py#L32-L48">32&ndash;48</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/export_column_labels.py#L74-L101">74&ndash;101</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/export_column_labels.py#L191">191</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/export_column_labels.py#L216-L219">216&ndash;219</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/export_column_labels.py#L223">223</a></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/merge_batch_vector.py">merge_batch_vector.py</a></td><td>40</td><td>7</td><td>82%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/merge_batch_vector.py#L28-L29">28&ndash;29</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/merge_batch_vector.py#L55-L58">55&ndash;58</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/data_processing/merge_batch_vector.py#L62">62</a></td></tr><tr><td colspan="5"><b>test/data_processing</b></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/data_processing/test_data_processing.py">test_data_processing.py</a></td><td>121</td><td>2</td><td>98%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/data_processing/test_data_processing.py#L113">113</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/data_processing/test_data_processing.py#L200">200</a></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/data_processing/test_export_column_labels.py">test_export_column_labels.py</a></td><td>90</td><td>5</td><td>94%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/data_processing/test_export_column_labels.py#L40">40</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/data_processing/test_export_column_labels.py#L72-L75">72&ndash;75</a></td></tr><tr><td colspan="5"><b>test/vector_tagging</b></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge.py">test_judge.py</a></td><td>155</td><td>2</td><td>99%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge.py#L66-L76">66&ndash;76</a></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge_loop.py">test_judge_loop.py</a></td><td>129</td><td>9</td><td>93%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge_loop.py#L72">72</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge_loop.py#L87-L88">87&ndash;88</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge_loop.py#L91">91</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge_loop.py#L137">137</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge_loop.py#L213">213</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge_loop.py#L224">224</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge_loop.py#L240">240</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/test/vector_tagging/test_judge_loop.py#L258">258</a></td></tr><tr><td colspan="5"><b>vector_tagging/JudgeLLM</b></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/JudgeLLM/label_judge.py">label_judge.py</a></td><td>69</td><td>1</td><td>99%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/JudgeLLM/label_judge.py#L42">42</a></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/JudgeLLM/modal_label_judge.py">modal_label_judge.py</a></td><td>166</td><td>18</td><td>89%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/JudgeLLM/modal_label_judge.py#L120-L134">120&ndash;134</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/JudgeLLM/modal_label_judge.py#L263-L264">263&ndash;264</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/JudgeLLM/modal_label_judge.py#L409-L410">409&ndash;410</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/JudgeLLM/modal_label_judge.py#L440-L441">440&ndash;441</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/JudgeLLM/modal_label_judge.py#L464">464</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/JudgeLLM/modal_label_judge.py#L506">506</a></td></tr><tr><td colspan="5"><b>vector_tagging/LLM</b></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/LLM/tagging_loop.py">tagging_loop.py</a></td><td>90</td><td>10</td><td>89%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/LLM/tagging_loop.py#L109-L119">109&ndash;119</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/LLM/tagging_loop.py#L156">156</a></td></tr><tr><td>&nbsp; &nbsp;<a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/LLM/vector_tagger.py">vector_tagger.py</a></td><td>124</td><td>35</td><td>72%</td><td><a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/LLM/vector_tagger.py#L15-L68">15&ndash;68</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/LLM/vector_tagger.py#L164">164</a>, <a href="https://github.com/UofT-CSC490-F2025/health-forge/blob/main/vector_tagging/LLM/vector_tagger.py#L269-L273">269&ndash;273</a></td></tr><tr><td><b>TOTAL</b></td><td><b>2424</b></td><td><b>484</b></td><td><b>80%</b></td><td>&nbsp;</td></tr></tbody></table></details>

<!-- Pytest Coverage Comment:End -->
