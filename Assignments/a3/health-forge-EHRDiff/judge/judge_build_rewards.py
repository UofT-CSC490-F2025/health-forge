# judge_build_rewards_fixed.py
# Correct per-sample reward computation

import numpy as np, json, argparse, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class RandomFourierMMD:
    def __init__(self, d, num_features=2048, gamma=1.0, seed=0):
        rng = np.random.RandomState(seed)
        self.W = rng.normal(scale=np.sqrt(2 * gamma), size=(d, num_features))
        self.b = rng.uniform(0, 2 * np.pi, size=(num_features,))
        self.z_scale = np.sqrt(2.0 / num_features)

    def rff(self, X):
        return self.z_scale * np.cos(X @ self.W + self.b)

    def mmd(self, X, Y):
        Zx, Zy = self.rff(X), self.rff(Y)
        mu_x, mu_y = Zx.mean(0), Zy.mean(0)
        return float(np.sum((mu_x - mu_y) ** 2))

class ShadowMI:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=200)

    def fit(self, real_feats, synth_feats):
        X = np.vstack([real_feats, synth_feats])
        y = np.concatenate([np.ones(len(real_feats)), np.zeros(len(synth_feats))])
        self.clf.fit(X, y)

    def score(self, feats):
        return self.clf.predict_proba(feats)[:, 1]  # p(real)

def compute_rewards(
    real, synth,
    lam=1.0,             # ↓ much smaller than 5.0 → preserves variation
    kappa=0.5,           # ↓ gentler privacy penalty
    w_dist=0.3,          # ↓ so global MMD contributes less flat signal
    w_priv=0.7,          # ↑ emphasize per-sample privacy differences
    local_mmd_frac=0.2,  # fraction of real samples to use per-sample
    noise_std=0.05,      # ↑ add more random diversity
    seed=0
):
    """
    Compute per-sample rewards with higher dynamic range.
    Combines per-sample MMD (local distance) and privacy rewards.
    """
    rng = np.random.RandomState(seed)
    d = real.shape[1]
    mmd_est = RandomFourierMMD(d, num_features=2048, gamma=1.0, seed=seed)

    # --- Train membership-inference model for privacy term ---
    mi_model = ShadowMI()
    half = len(real) // 2
    mi_model.fit(real[:half], real[half:])

    # --- Compute per-sample local MMD instead of one global value ---
    n_real_subset = max(1, int(local_mmd_frac * len(real)))
    r_dists = []
    for i in range(len(synth)):
        # sample a random small subset of real samples for each synthetic example
        idx = rng.choice(len(real), n_real_subset, replace=False)
        local_mmd = mmd_est.mmd(real[idx], synth[i:i+1])
        r_dists.append(np.exp(-lam * local_mmd))
    r_dists = np.array(r_dists)

    # --- Privacy term (per-sample) ---
    priv_scores = mi_model.score(synth)
    r_priv = np.exp(-kappa * priv_scores)

    # --- Add controlled noise to r_dist to increase diversity ---
    noise = rng.normal(0, noise_std, size=len(synth))
    r_dists = np.clip(r_dists + noise, 0, 1)

    # --- Combine with new weights ---
    r_total = w_dist * r_dists + w_priv * r_priv

    # --- Optional normalization to [0,1] ---
    r_total = (r_total - r_total.min()) / (r_total.max() - r_total.min() + 1e-8)

    return r_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--outdir", default="judge/data")
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--kappa", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    real = np.load(args.real)
    synth = np.load(args.synth)

    rewards = compute_rewards(real, synth, lam=args.lam, kappa=args.kappa)

    labeled = [{"ehr_id": int(i), "reward": float(r)} for i, r in enumerate(rewards)]
    tr, te = train_test_split(labeled, test_size=0.2, random_state=42)
    tr, va = train_test_split(tr, test_size=0.125, random_state=42)

    def dump(name, data):
        path = os.path.join(args.outdir, f"{name}.jsonl")
        with open(path, "w") as w:
            for row in data:
                w.write(json.dumps(row) + "\n")

    dump("train", tr)
    dump("val", va)
    dump("test", te)
    print(f"Saved reward-labeled sets to {args.outdir}")
    print(f"Reward stats: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")

if __name__ == "__main__":
    main()
