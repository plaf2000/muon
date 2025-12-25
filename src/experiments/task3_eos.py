"""
Task 3: Edge-of-Stability (EoS) quantity tracking.

Part (A) EoS quantity:
- GD/SGD (no preconditioner): eos = eta * lambda_max(H_t)
- AdamW/Muon (preconditioned): true quantity is eta * lambda_max(H_eff,t)
  with H_eff = P^{-1/2} H P^{-1/2}.
  We don't compute H_eff here yet, we still log a baseline eos = eta * lambda_max(H_t)
  and label it. (Part B will implement preconditioned curvature.)
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import yaml
import wandb
import math

from src.models import MLP, TinyViT
from src.optimizers import Muon
from src.utils.data import load_mnist, load_cifar10
from src.geometry.hessian import compute_lambda_max

import matplotlib.pyplot as plt

def load_config(config_path: str) -> Dict[str, Any]:

    # Load experiment configuration from YAML file.

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_str: str = "auto") -> torch.device:
    
    # Get available device ("auto", "cuda", "mps", "cpu").
    
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def create_model(model_config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Create model based on repo conventions:
      model:
        type: mlp | tiny_vit
        config: {...}
    """
    model_type = model_config["type"]
    cfg = model_config.get("config", {})

    if model_type == "mlp":
        model = MLP(
            input_size=cfg.get("input_size", 784),
            hidden_sizes=cfg.get("hidden_sizes", [128, 64]),
            num_classes=cfg.get("num_classes", 10),
            dropout=cfg.get("dropout", 0.0),
        )
    elif model_type == "tiny_vit":
        model = TinyViT(
            img_size=cfg.get("img_size", 32),
            patch_size=cfg.get("patch_size", 4),
            num_classes=cfg.get("num_classes", 10),
            embed_dim=cfg.get("embed_dim", 128),
            depth=cfg.get("depth", 4),
            num_heads=cfg.get("num_heads", 4),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            dropout=cfg.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def load_data(dataset_cfg: Dict[str, Any], seed: int, device: torch.device):
    """
    Load MNIST/CIFAR10 using repo utilities.
    Supports full-batch training and an optional train_subset_size.
    """
    name = dataset_cfg.get("name", "mnist")
    root = dataset_cfg.get("root", "./data")
    full_batch = dataset_cfg.get("full_batch", True)
    subset = dataset_cfg.get("train_subset_size", None)

    if name == "mnist":
        train_x, train_y = load_mnist(root=root, train=True, download=True, full_batch=full_batch)
        test_x, test_y = load_mnist(root=root, train=False, download=True, full_batch=full_batch)
    elif name == "cifar10":
        train_x, train_y = load_cifar10(root=root, train=True, download=True, full_batch=full_batch)
        test_x, test_y = load_cifar10(root=root, train=False, download=True, full_batch=full_batch)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if subset is not None and subset > 0 and subset < len(train_x):
        torch.manual_seed(seed)
        idx = torch.randperm(len(train_x))[:subset]
        idx = torch.sort(idx)[0]
        train_x, train_y = train_x[idx], train_y[idx]

    return train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)


def create_optimizer(model: nn.Module, optimizer_cfg: Dict[str, Any]):
    """
    Matches the repo style:
      optimizer:
        type: gd | sgd | adamw | muon
        config: {...}

    Returns:
      - SGD/AdamW: a single optimizer
      - Muon: tuple (MuonOpt, AdamWOpt) like basic_training.py
    """
    opt_type = optimizer_cfg["type"]
    cfg = optimizer_cfg.get("config", {})

    if opt_type in ["gd", "sgd"]:
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.get("lr", 0.01),
            momentum=cfg.get("momentum", 0.0),
            weight_decay=cfg.get("weight_decay", 0.0),
        )

    if opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 0.01),
        )

    if opt_type == "muon":
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if (p.ndim >= 2 and
                "output_layer" not in name and
                "head" not in name and
                "embed" not in name.lower() and
                "bias" not in name):
                muon_params.append(p)
            else:
                adamw_params.append(p)

        muon_opt = Muon(
            muon_params,
            lr=cfg.get("lr", 0.02),
            momentum=cfg.get("momentum", 0.95),
            ns_depth=cfg.get("ns_depth", 5),
            use_rms=cfg.get("use_rms", False),
            use_orthogonalization=cfg.get("use_orthogonalization", True),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=cfg.get("adamw_lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
        return (muon_opt, adamw_opt)

    raise ValueError(f"Unknown optimizer type: {opt_type}")


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def _zero_opt(opt):
    if isinstance(opt, tuple):
        for o in opt:
            o.zero_grad()
    else:
        opt.zero_grad()


def _step_opt(opt):
    if isinstance(opt, tuple):
        for o in opt:
            o.step()
    else:
        opt.step()

def _filter_xy(xs, ys):
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        if isinstance(y, float) and math.isnan(y):
            continue
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y

def plot_task3A(history: Dict[str, List[float]], save_path: Path, title: str) -> None:
    epochs = history["epoch"]

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["test_loss"], label="test_loss")
    plt.xlabel("epoch")
    plt.title(title + " (loss)")
    plt.legend()
    plt.tight_layout()
    loss_path = save_path.with_name(save_path.stem + "_loss.png")
    plt.savefig(loss_path)
    plt.close()

    lam_x, lam_y = _filter_xy(epochs, history["lambda_max"])
    eos_x, eos_y = _filter_xy(epochs, history["eos_value"])

    plt.figure()
    if lam_x:
        plt.plot(lam_x, lam_y, marker="o", label="lambda_max(H)")
    if eos_x:
        plt.plot(eos_x, eos_y, marker="o", label="eos_value")
    plt.xlabel("epoch")
    plt.title(title + " (sharpness / eos) [tracked epochs]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.with_name(save_path.stem + "_sharpness_eos.png"))
    plt.close()


def run_eos_analysis(config: Dict[str, Any], opt_name: str = "single") -> Dict[str, List[float]]:
    """
    Runs Edge-of-Stability analysis (Part A only):
      - trains full-batch
      - computes lambda_max(H_t) at track_frequency
      - computes eos quantity for GD/SGD as eta * lambda_max(H_t)
      - for AdamW/Muon logs baseline eta * lambda_max(H_t) (H_eff not yet computed)
      - logs everything to W&B (MANDATORY)
      - saves results + plots locally + upload plots to W&B
    """
   
    # Enforce mandatory W&B
    log_cfg = config.get("logging", {}).get("wandb", {})
    if not log_cfg.get("enabled", False):
        raise RuntimeError(
            "W&B logging is mandatory for Task 3.\n"
            "Set:\n"
            "logging:\n"
            "  wandb:\n"
            "    enabled: true\n"
        )

    # Force a clean “not logged in” error early
    try:
        wandb.ensure_configured()
    except Exception as e:
        raise RuntimeError(
            "W&B is required for Task 3 but you are not logged in.\n"
            "Run: wandb login\n"
            "or set WANDB_API_KEY in your environment.\n"
        ) from e


    # Setup
    exp = config["experiment"]
    seed = int(exp.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(exp.get("device", "auto"))

    train_x, train_y, test_x, test_y = load_data(config["dataset"], seed, device)
    model = create_model(config["model"], device)

    train_cfg = config.get("training", {})
    num_epochs = int(train_cfg.get("num_epochs", 100))

    # Loss
    loss_fn_name = train_cfg.get("loss_fn", "cross_entropy")
    if loss_fn_name != "cross_entropy":
        raise ValueError("Only cross_entropy supported right now.")
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer_cfg = config["optimizer"]
    opt_type = optimizer_cfg["type"]
    opt_cfg = optimizer_cfg.get("config", {})
    optimizer = create_optimizer(model, optimizer_cfg)

    # EoS tracking frequency
    eos_cfg = config.get("eos_analysis", {})
    track_freq = int(eos_cfg.get("track_frequency", 1))

    # Hessian power-iteration params
    h_cfg = config.get("hessian", {})
    max_iter = int(h_cfg.get("max_iter", 50))
    tol = float(h_cfg.get("tol", 1e-6))

    # Step size (eta)
    lr = float(opt_cfg.get("lr", 0.01))

    # Output dirs
    out_cfg = config.get("output", {})
    save_root = Path(out_cfg.get("save_dir", "./results/task3"))
    save_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_type = optimizer_cfg["type"]

    run_name = f"{exp.get('name','task3_eos')}__{opt_name}__{timestamp}"

    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Group all optimizers under the same group in W&B
    group_name = exp.get("name", "task3_eos")

    wandb.init(
        project=log_cfg.get("project", "muon"),
        entity=log_cfg.get("entity", None),
        name=run_name,
        group=group_name,
        job_type=opt_name,   
        config=config,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lambda_max": [],
        "eos_value": [],
        "eos_kind": [],
    }

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        _zero_opt(optimizer)

        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        _step_opt(optimizer)

        tr_loss = float(loss.item())
        tr_acc = float(accuracy(logits.detach(), train_y))

        model.eval()
        with torch.no_grad():
            test_logits = model(test_x)
            te_loss = float(loss_fn(test_logits, test_y).item())
            te_acc = float(accuracy(test_logits, test_y))

        lam_max = float("nan")
        eos_val = float("nan")
        eos_kind = ""

        if epoch % track_freq == 0 or epoch == num_epochs - 1:
            lam = compute_lambda_max(
                model, loss_fn, train_x, train_y,
                max_iter=max_iter, tol=tol, device=device
            )
            lam_max = float(lam)

            # Part A: correct EoS for GD/SGD
            if opt_type in ["gd", "sgd"]:
                eos_val = lr * lam_max
                eos_kind = "eta*lambda_max(H)"
            else:
                # placeholder baseline until H_eff is implemented
                eos_val = lr * lam_max
                eos_kind = "baseline eta*lambda_max(H) (H_eff not computed yet)"

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["lambda_max"].append(lam_max)
        history["eos_value"].append(eos_val)
        history["eos_kind"].append(eos_kind)

        wandb.log(
            {
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "test_loss": te_loss,
                "test_acc": te_acc,
                "lambda_max": None if torch.isnan(torch.tensor(lam_max)) else lam_max,
                "eos_value": None if torch.isnan(torch.tensor(eos_val)) else eos_val,
            },
            step=epoch
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{num_epochs} "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"test_loss={te_loss:.4f} test_acc={te_acc:.4f} "
                f"lambda_max={lam_max:.4f} eos={eos_val:.4f}"
            )


    # Save results locally
    results_path = run_dir / f"{run_name}_results.pt"
    torch.save({"history": history, "config": config}, results_path)

    cfg_path = run_dir / f"{run_name}_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


    # Plots + W&B images
    base_plot_path = vis_dir / f"{run_name}_task3A.png"
    plot_task3A(history, base_plot_path, title=f"Task 3(A): {opt_type}")

    loss_img = vis_dir / f"{run_name}_task3A_loss.png"
    sharp_img = vis_dir / f"{run_name}_task3A_sharpness_eos.png"

    if loss_img.exists():
        wandb.log({"plot_loss": wandb.Image(str(loss_img))})
    if sharp_img.exists():
        wandb.log({"plot_sharpness_eos": wandb.Image(str(sharp_img))})

    wandb.finish()

    print(f"Saved run ({opt_name}) to: {run_dir}")
    return history


def run_curvature_analysis(config: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Part (B) will compute preconditioned curvature / directional Rayleigh quotients.
    Not implemented yet.
    """
    raise NotImplementedError("Curvature analysis (Part B) not implemented yet.")


def main():
    parser = argparse.ArgumentParser(description="Task 3: Edge-of-Stability Analysis (Part A)")
    parser.add_argument("--config", type=str, default="configs/task3_eos.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Multi-run mode (preferred)
    optimizers = config.get("optimizers", None)
    if optimizers:
        print("=" * 80)
        print("Task 3(A): Running multiple optimizers (one W&B run each)")
        print("Optimizers:", ", ".join(optimizers.keys()))
        print("=" * 80)

        summary = {}
        for opt_name, opt_cfg in optimizers.items():
            run_cfg = dict(config)
            run_cfg["optimizer"] = opt_cfg  # inject into the single-run code path

            try:
                run_eos_analysis(run_cfg, opt_name=opt_name)
                summary[opt_name] = "success"
            except Exception as e:
                print(f"[FAILED] {opt_name}: {e}")
                summary[opt_name] = "failed"

        print("\n" + "=" * 80)
        print("Task 3(A) Summary")
        print("=" * 80)
        for k, v in summary.items():
            print(f"{k}: {v}")
        return

    # Single-run fallback (if you keep `optimizer:` in YAML)
    if "optimizer" not in config:
        raise KeyError(
            "Config must contain either `optimizer:` (single run) "
            "or `optimizers:` (multi run)."
        )

    run_eos_analysis(config, opt_name=config["optimizer"]["type"])

if __name__ == "__main__":
    main()

