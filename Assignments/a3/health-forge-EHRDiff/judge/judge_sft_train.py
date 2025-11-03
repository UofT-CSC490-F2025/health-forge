# judge_sft_train.py
# Supervised fine-tuning of the LLM Judge on reward-labeled EHR data

import os, json, yaml, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch import nn
from tqdm import tqdm

# ------------------------
# Config (inline or YAML)
# ------------------------
CFG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "train_file": "rewards_data/train.jsonl",
    "val_file": "rewards_data/val.jsonl",
    "save_dir": "judge_sft",
    "batch_size": 8,
    "lr": 2e-5,
    "epochs": 5,
    "max_seq_len": 1536
}

# ------------------------
# Dataset
# ------------------------
class RewardDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1536):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        text = "EHR: " + json.dumps(row.get("ehr", {"ehr_id": row.get("ehr_id")}), separators=(",",":"))
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "reward": torch.tensor(row["reward"], dtype=torch.float32)
        }

def collate_fn(batch, pad_token_id):
    input_ids = nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=pad_token_id
    )
    attention_mask = nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0
    )
    rewards = torch.stack([b["reward"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "reward": rewards}

# ------------------------
# Model with scalar head
# ------------------------
class ScalarJudge(nn.Module):
    def __init__(self, base_name):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(base_name)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = out.hidden_states[-1][:, -1, :]   # final token
        return self.head(last_hidden).squeeze(-1)

# ------------------------
# Main
# ------------------------
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tok = AutoTokenizer.from_pretrained(CFG["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = RewardDataset(CFG["train_file"], tok, CFG["max_seq_len"])
    val_ds = RewardDataset(CFG["val_file"], tok, CFG["max_seq_len"])

    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"],
                          shuffle=True, collate_fn=lambda b: collate_fn(b, tok.pad_token_id))
    val_dl = DataLoader(val_ds, batch_size=CFG["batch_size"],
                        shuffle=False, collate_fn=lambda b: collate_fn(b, tok.pad_token_id))

    model = ScalarJudge(CFG["model_name"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    mse = nn.MSELoss()

    total_steps = CFG["epochs"] * len(train_dl)
    scheduler = get_scheduler("cosine", optimizer=opt,
                              num_warmup_steps=int(0.05 * total_steps),
                              num_training_steps=total_steps)

    model.train()
    for epoch in range(CFG["epochs"]):
        epoch_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            for k in ("input_ids", "attention_mask", "reward"):
                batch[k] = batch[k].to(device)
            preds = model(batch["input_ids"], batch["attention_mask"])
            loss = mse(preds, batch["reward"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} mean loss: {epoch_loss/len(train_dl):.4f}")

        # quick val check
        model.eval(); val_loss=0
        with torch.no_grad():
            for batch in val_dl:
                for k in ("input_ids","attention_mask","reward"):
                    batch[k]=batch[k].to(device)
                preds = model(batch["input_ids"], batch["attention_mask"])
                val_loss += mse(preds, batch["reward"]).item()
        print(f"Validation MSE: {val_loss/len(val_dl):.4f}")
        model.train()

    os.makedirs(CFG["save_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CFG["save_dir"], "pytorch_model.bin"))
    tok.save_pretrained(CFG["save_dir"])

    with open(os.path.join(CFG["save_dir"], "train_config.json"), "w") as f:
        json.dump(CFG, f, indent=2)

    print(f"Saved model weights and tokenizer to {CFG['save_dir']}")

if __name__ == "__main__":
    main()
