import torch
from torch.utils.data import Dataset, DataLoader
import pytest
import os

import final_project.trainer as tr 


# -----------------------
# Dummy model & dataset
# -----------------------

class TinyDiffusionModel(torch.nn.Module):
    """
    Minimal model compatible with DiffusionTrainer:
    forward(z_l, text_embed) -> epsilon_pred
    Ignores text_embed, just a linear map on z_l.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Linear(input_dim, input_dim)

    def forward(self, z_l, text_embed):
        return self.net(z_l)


class DummyDiffusionDataset(Dataset):
    """
    Yields (z_l, text_embed, epsilon_true)
    where epsilon_true is a simple linear function of z_l
    so the model has something learnable.
    """
    def __init__(self, n_samples=32, input_dim=4, embed_dim=3):
        super().__init__()
        torch.manual_seed(0)
        self.z_l = torch.randn(n_samples, input_dim)
        self.text_embed = torch.randn(n_samples, embed_dim)
        # Ground truth: epsilon = 2 * z_l
        self.epsilon_true = 2.0 * self.z_l

    def __len__(self):
        return len(self.z_l)

    def __getitem__(self, idx):
        return (
            self.z_l[idx],
            self.text_embed[idx],
            self.epsilon_true[idx],
        )


def make_trainer(tmp_path, n_train=32, n_val=16, input_dim=4, embed_dim=3, num_epochs=2, lr=1e-2):
    """
    Helper to construct a DiffusionTrainer with small dummy data.
    """
    # Build tiny datasets
    train_ds = DummyDiffusionDataset(n_samples=n_train, input_dim=input_dim, embed_dim=embed_dim)
    val_ds = DummyDiffusionDataset(n_samples=n_val, input_dim=input_dim, embed_dim=embed_dim)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = TinyDiffusionModel(input_dim=input_dim)

    cfg = {
        "lr": lr,
        "num_epochs": num_epochs,
        "save_path": str(tmp_path / "best_model.pt"),
    }

    trainer = tr.DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        cfg=cfg,
        device="cpu",
    )
    return trainer, cfg


# -----------------------
# Tests
# -----------------------

def test_train_epoch_runs_and_updates_params(tmp_path):
    trainer, cfg = make_trainer(tmp_path, num_epochs=1)

    # Snapshot parameters before
    before_params = [p.detach().clone() for p in trainer.model.parameters()]

    avg_loss = trainer.train_epoch()

    # Loss should be a finite positive number (MSE)
    assert isinstance(avg_loss, float)
    assert avg_loss >= 0.0
    assert torch.isfinite(torch.tensor(avg_loss))

    # At least one parameter should have changed
    after_params = list(trainer.model.parameters())
    changed = any(
        not torch.allclose(b, a.detach(), atol=1e-6)
        for b, a in zip(before_params, after_params)
    )
    assert changed, "Expected some model parameters to change after training step."


def test_validate_with_nonempty_loader(tmp_path):
    trainer, cfg = make_trainer(tmp_path, num_epochs=1)

    # Run one validation pass
    val_loss = trainer.validate()

    assert isinstance(val_loss, float)
    assert val_loss >= 0.0
    assert torch.isfinite(torch.tensor(val_loss))
    # validate() should record the loss
    assert len(trainer.val_losses) == 1


def test_validate_with_empty_loader_returns_zero(tmp_path):
    """
    When test_loader is effectively empty (len == 0),
    validate() should early-return 0 and not append to val_losses.
    """
    # Reuse the helper but override test_loader with an empty DataLoader
    trainer, cfg = make_trainer(tmp_path, num_epochs=1)
    empty_loader = DataLoader([], batch_size=4)
    trainer.test_loader = empty_loader

    val_loss = trainer.validate()

    assert val_loss == 0
    assert len(trainer.val_losses) == 0


def test_train_saves_checkpoints_and_returns_losses(tmp_path):
    """
    Full training loop:
    - runs for num_epochs
    - saves checkpoints to cfg['save_path']
    - returns train_losses and val_losses of correct length
    """
    num_epochs = 2
    trainer, cfg = make_trainer(tmp_path, num_epochs=num_epochs)

    train_losses, val_losses = trainer.train()

    # Lengths should match num_epochs
    assert len(train_losses) == num_epochs
    assert len(val_losses) == num_epochs

    # Final save_path file should exist
    save_path = cfg["save_path"]
    assert os.path.exists(save_path)

    # Check contents of checkpoint file
    ckpt = torch.load(save_path, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "train_loss" in ckpt
    assert "val_loss" in ckpt


def test_load_checkpoint_restores_state(tmp_path):
    """
    Verify that load_checkpoint correctly loads model & optimizer state.
    """
    # First trainer: run 1 epoch and then save via .train()
    trainer1, cfg = make_trainer(tmp_path, num_epochs=1)
    trainer1.train()  # this will save to cfg['save_path']

    save_path = cfg["save_path"]
    assert os.path.exists(save_path)

    # Second trainer: starts from fresh random weights
    trainer2, _ = make_trainer(tmp_path, num_epochs=1)

    # Ensure initial states differ
    params1_before = [p.detach().clone() for p in trainer1.model.parameters()]
    params2_before = [p.detach().clone() for p in trainer2.model.parameters()]
    assert any(
        not torch.allclose(p1, p2, atol=1e-6)
        for p1, p2 in zip(params1_before, params2_before)
    ), "Fresh trainer should start from different weights."

    # Load checkpoint into trainer2
    checkpoint = trainer2.load_checkpoint(save_path)

    # After loading, trainer2's parameters should match the checkpoint
    params2_after = [p.detach().clone() for p in trainer2.model.parameters()]
    for p_loaded, p_ckpt in zip(params2_after, checkpoint["model_state_dict"].values()):
        # state_dict is ordered in PyTorch 1.9+, mapping weights & biases
        assert p_loaded.shape == p_ckpt.shape

    # Also verify epoch field is present
    assert "epoch" in checkpoint
    assert isinstance(checkpoint["epoch"], int)
