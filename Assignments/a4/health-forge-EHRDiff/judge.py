import argparse
import numpy as np
import torch
from runners import generate_base
from omegaconf import OmegaConf
import os
from sklearn.neighbors import NearestNeighbors


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
        
        # Step 3: Create weighted sum of two metrics
        mah_weight, knn_weight = 0.7, 0.3
        score_per_sample = (mah_weight * mah_dist) + (knn_weight * neigh_dists)

        return score_per_sample





    def train(self):
        # Generate synthetic samples
        # Score synthetic samples
        # Instantiate LLM
        # Train LLM with RL based on scored samples
        synthetic_samples = self.generate_samples()
        score_per_sample = self.score_samples(synthetic_samples)
        print(score_per_sample.shape)
        


        pass
    


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
        