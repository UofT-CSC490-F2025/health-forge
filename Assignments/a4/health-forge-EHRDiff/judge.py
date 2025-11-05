import argparse
import numpy as np
import torch
from runners import generate_base
from omegaconf import OmegaConf
import os
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Judge:
    def __init__(self, config):
        self.config = config
    
    def generate_samples(self):

        workdir = os.path.join(self.config.setup.root_dir, self.config.setup.workdir)
        sample_dir = os.path.join(workdir, "samples")

        generate_base.evaluation(self.config, workdir)
        sample_file_path = os.path.join(sample_dir, "all_x.npy")
        assert os.path.exists(sample_file_path), "[ERROR] Failed to generate samples"

        synthetic_samples = np.load(sample_file_path)


        return synthetic_samples
    
    # this score will be used in the calculation of the reward during RLVR. the calculations made are verifiable rewards.
    def score_samples(self, synthetic_samples):
        dataset_dir = os.path.join(self.config.setup.root_dir, self.config.setup.dataset_dir)
        assert os.path.exists(dataset_dir), "[ERROR] Training dataset not found"
        train_samples = np.load(dataset_dir)

        # Step 1: Compute mahalanobis distance per sample (https://en.wikipedia.org/wiki/Mahalanobis_distance)
        mu = np.mean(train_samples, axis=0)
        cov = np.cov(train_samples, rowvar=False)
        diff = synthetic_samples - mu
        mah_dist = np.sqrt(np.sum((diff @ np.linalg.pinv(cov)) * diff, axis=1) + 1e-9) # Add small offset to avoid nans when computing sqrt

        # Step 2: Compute distance from each sample to nearest neighbor in training distribution 
        n_neighbors = int(0.1 * (train_samples.shape[0]))
        neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=0.1)
        neigh.fit(train_samples)
        neigh_dists, neigh_idx  = neigh.kneighbors(synthetic_samples, return_distance=True)
        neigh_dists = np.mean(neigh_dists, axis=1)

        # Step 3: Create weighted difference of two metrics
        mah_weight, knn_weight = 0.7, 0.3
        score_per_sample = - (mah_weight * mah_dist) + (knn_weight * neigh_dists)

        return score_per_sample





    def train(self):
        # Generate synthetic samples
        print("Generating synthetic samples...")
        synthetic_samples = self.generate_samples()

        # Score synthetic samples using VR metric
        print("Scoring synthetic samples...")
        vr_scores = self.score_samples(synthetic_samples)
        vr_scores = np.clip(vr_scores, 0, 1)  # make sure VR scores are in 0-1

        # Convert VR scores to 1-10 for easier comparison with LLM
        vr_labels = [float(score * 10) for score in vr_scores]

        # Train baseline LLM with arbitrary scoring
        print("Training baseline LLM with arbitrary scoring...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Convert EHR vectors to text
        texts = [self.vector_to_text(x) for x in synthetic_samples]

        # Generate initial arbitrary scores for the LLM (random between 1-10)
        import random
        arbitrary_labels = [random.randint(1, 10) for _ in synthetic_samples]

        batch_size = 8
        epochs = 1
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_labels = arbitrary_labels[i:i+batch_size]
                batch_vr = vr_labels[i:i+batch_size]  # the "true" VR scores

                inputs = tokenizer(
                    [f"Rate from 1 to 10 how realistic this synthetic patient record looks:\n{text}" for text in batch_texts],
                    return_tensors="pt", truncation=True, padding=True, max_length=512
                ).to(device)

                target_ids = tokenizer(
                    [str(int(l)) for l in batch_labels],
                    return_tensors="pt", padding=True, truncation=True, max_length=5
                ).input_ids.to(device)

                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=target_ids)
                
                # RL-style penalty: multiply loss by |LLM_score - VR_score| / 10
                batch_pred = outputs.logits.argmax(-1).float()  # predicted token IDs
                # Convert token IDs to approximate numeric scores (simple heuristic)
                batch_pred_scores = batch_pred[:, 0]  # first token
                batch_vr_tensor = torch.tensor(batch_vr, device=device)
                penalty = torch.abs(batch_pred_scores - batch_vr_tensor)
                loss = outputs.loss + penalty.mean()

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"[Baseline LLM + VR penalty] Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(texts):.4f}")

        print("Training complete! LLM now scores arbitrarily but penalized by VR metric differences.")

    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, type=str, choices=["train", "eval"])
    parser.add_argument('--workdir', required=True)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("-output_dir", type=str, default="./judge_output")
    parser.add_argument("--model_cfg", default="", help="Config for base diffusion model")
    parser.add_argument('--root_dir', default='.')

    return parser.parse_args()   

def setup_config(args):
    config = OmegaConf.load(args.model_cfg)

    config.setup.workdir = args.workdir
    config.setup.mode = args.mode
    config.setup.root_dir = args.root_dir
    config.setup.dataset_dir = args.dataset_dir

    for rank in range(config.setup.n_gpus_per_node):
        config.setup.local_rank = rank
        config.setup.global_rank = rank + \
        config.setup.node_rank * config.setup.n_gpus_per_node
        config.setup.global_size = config.setup.n_nodes * config.setup.n_gpus_per_node
    
    return config


def main():
    args = parse_args()
    if args.mode == "train" and (not hasattr(args, "dataset_dir") or not hasattr(args, "model_cfg")):
        print("[ERROR] Tried to train without any input data")
        exit(-1)

    config = setup_config(args)


    if args.mode == "train":
        judge = Judge(config)
        judge.train()
        return


if __name__ == "__main__":
    main()
        