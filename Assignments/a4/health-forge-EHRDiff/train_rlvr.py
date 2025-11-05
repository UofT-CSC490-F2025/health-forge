import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
from part3_baseline_llm import compute_realism_scores, vector_to_text  # import from your module


# ---- Helper: Extract a score from model output ----
def parse_score(text):
    text = text.strip()
    if "10" in text:
        return 10
    digits = [int(c) for c in text if c.isdigit()]
    return digits[0] if digits else 5


# ---- Helper: sample model outputs (policy action) ----
###################################################################
# WILL TODO: THIS PROMPTING NEEDS TO MATCH BASELINE LLM PROMPTING #
###################################################################

@torch.no_grad()
def sample_scores(texts, tokenizer, model, device):
    prompts = [f"Rate from 1 to 10 how realistic this synthetic patient record looks:\n{t}" for t in texts]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    gen = model.generate(
        **enc,
        max_new_tokens=5,
        do_sample=True,
        top_p=0.9,
        temperature=0.9
    )

    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
    scores = [parse_score(x) for x in decoded]

    return gen, enc, torch.tensor(scores, dtype=torch.float32, device=device)


# ---- Helper: compute sequence log prob ----
# needed because RLVR performs a policy-gradient update, which is based on log-probs, not standard loss
def seq_logprob(model, input_ids, attn_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    logits = outputs.logits
    token_mask = (labels != -100).float()

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * token_mask

    return token_logp.sum(dim=1)  # sum over tokens → shape [batch]


# ---- RLVR Trainer ----
class RLVRTrainer:
    def __init__(self, baseline_dir, lr=1e-5, kl_beta=0.05, device="cuda"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(baseline_dir)

        # the LLM baseline model that we are improving
        self.policy = AutoModelForSeq2SeqLM.from_pretrained(baseline_dir).to(device)

        # the frozen reference model for KL penalty
        self.ref = AutoModelForSeq2SeqLM.from_pretrained(baseline_dir).to(device).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.kl_beta = kl_beta

    def train_epoch(self, X_real, X_synth, batch=16, max_samples=256):
        # compute verifiable realism scores
        vr_scores = compute_realism_scores(X_real, X_synth)
        vr_scores = torch.tensor(vr_scores, dtype=torch.float32, device=self.device)

        texts = [vector_to_text(x) for x in X_synth[:max_samples]]
        avg_loss = 0
        n = 0

        for i in tqdm(range(0, len(texts), batch), desc="RLVR Training"):
            batch_texts = texts[i:i+batch]
            batch_vr = vr_scores[i:i+batch]

            # 1) Sample model output
            gen_ids, enc_inputs, llm_scores = sample_scores(batch_texts, self.tokenizer, self.policy, self.device)

            # prepare labels for logprob calc
            labels = gen_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            # 2) Convert VR realism score (0–1) → target rating (1–10)
            vr_target = 1 + 9 * batch_vr

            # 3) Compute reward based on alignment
            reward = 1 - (torch.abs(llm_scores - vr_target) / 9.0)
            reward = reward.clamp(0, 1)
            # this makes it so that only better than average samples get positive advantage
            advantage = reward - reward.mean()

            # 4) Policy & KL loss
            logp_policy = seq_logprob(self.policy, enc_inputs.input_ids, enc_inputs.attention_mask, labels)
            with torch.no_grad():
                logp_ref = seq_logprob(self.ref, enc_inputs.input_ids, enc_inputs.attention_mask, labels)

            policy_loss = -(advantage * logp_policy).mean()
            kl_loss = (logp_policy - logp_ref).mean()
            loss = policy_loss + self.kl_beta * kl_loss

            # 5) Optimize
            self.opt.zero_grad()
            loss.backward()
            clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()

            avg_loss += loss.item() * len(batch_texts)
            n += len(batch_texts)

        return avg_loss / n

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"\n[RLVR Model Saved] → {path}")


# ---- Entry point ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", default="baseline")
    parser.add_argument("--real_data", required=True, help="Path to real training data .npy")
    parser.add_argument("--synth_data", required=True, help="Path to synthetic data .npy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out_dir", default="rlvr_judge")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    trainer = RLVRTrainer(args.baseline_dir, device=args.device)

    X_real = np.load(args.real_data)
    X_synth = np.load(args.synth_data)

    for epoch in range(args.epochs):
        loss = trainer.train_epoch(X_real, X_synth)
        print(f"Epoch {epoch+1} Loss = {loss:.4f}")

    trainer.save(args.out_dir)


if __name__ == "__main__":
    main()