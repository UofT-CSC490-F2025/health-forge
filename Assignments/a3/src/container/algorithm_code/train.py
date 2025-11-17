import sys
import main as EHRDiff

if __name__ == '__main__':
    sys.argv = [
        "main.py",
        "--config", "./configs/mimic/health_forge_train_edm.yaml",
        "--workdir", "opt/code/ml", # Where sagemaker mounts stuff (and where we output)
        "--mode", "train",
    ]

    EHRDiff.execute()