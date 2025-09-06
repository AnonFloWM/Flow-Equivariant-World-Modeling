import argparse
import pathlib
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose, Normalize
from torchvision.utils import make_grid
from moviepy.editor import ImageSequenceClip

import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm
import os

# -----------------------------------------------------------------------------
#  Dataset & model imports
# -----------------------------------------------------------------------------
from dynamic_mnist_world_dataset import MNISTWorldDynamicDataset_FromFiles, get_train_val_splits  # local file generated earlier
from models import FlowEquivariantRNN, LinearRNNBaseline  # add LinearRNNBaseline

# -----------------------------------------------------------------------------
#  Auxiliary helpers
# -----------------------------------------------------------------------------

def save_checkpoint(state: dict, ckpt_dir: pathlib.Path, epoch: int):
    ckpt_dir = pathlib.Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fname = ckpt_dir / f"checkpoint_best.pt"
    torch.save(state, fname)
    print(f"[✓] Saved checkpoint to {fname}")


def log_sequences(model, data_source, device, step, dataset_type="val", max_videos=4, max_batches=1):
    """Log sequences to wandb for qualitative monitoring.
    
    Args:
        model: The model to evaluate
        data_source: Either a DataLoader or a dict containing batch data
        device: Device to run inference on
        step: Step number for wandb logging
        dataset_type: Type of dataset ("train" or "val")
        max_videos: Maximum number of videos to log
        max_batches: Maximum number of batches to process (only used if data_source is DataLoader)
    """
    model.eval()
    logged = 0
    
    with torch.no_grad():
        # Handle DataLoader case
        if hasattr(data_source, '__iter__') and not isinstance(data_source, dict):
            # data_source is a DataLoader
            for batch_idx, batch in enumerate(data_source):
                if batch_idx >= max_batches or logged >= max_videos:
                    break
                    
                inp = batch["input_seq"].to(device)
                in_actions = batch["input_actions"].to(device)
                tgt_actions = batch["target_actions"].to(device)
                tgt = batch["target_seq"].to(device)
                
                pred = model(inp, in_actions, tgt_actions)
                
                # Log videos from this batch
                batch_size = inp.size(0)
                for b in range(min(batch_size, max_videos - logged)):
                    _log_single_sequence(tgt[b], pred[b], step, dataset_type, logged)
                    logged += 1
                    if logged >= max_videos:
                        break
                        
        else:
            # data_source is a batch dict
            inp = data_source["input_seq"].to(device)
            in_actions = data_source["input_actions"].to(device)
            tgt_actions = data_source["target_actions"].to(device)
            tgt = data_source["target_seq"].to(device)
            
            pred = model(inp, in_actions, tgt_actions)
            
            # Log videos from this batch
            batch_size = inp.size(0)
            for b in range(min(batch_size, max_videos)):
                _log_single_sequence(tgt[b], pred[b], step, dataset_type, b)
                logged += 1
                if logged >= max_videos:
                    break
    
    model.train()



def log_sequence_predictions_new(
    input_seq, target_seq, output_seq,
    split_name,    
    num_samples: int = 4,          # number of sequences to visualise
    vmax_diff: float = 1.0,        # clip range for the signed difference plot
    subsample_t: int = 1,          # subsample the time dimension by this factor
    device: torch.device | None = None,
    log_all_frames: bool = False,
):
    """
    Visualise ground–truth, prediction, and signed error for a handful of sequences.

    """
    T = target_seq.shape[1]
    T = T // subsample_t

    num_samples = min(num_samples, target_seq.shape[0])

    # --- iterate over the first num_samples sequences ------------------------
    for idx in range(num_samples):
        gt_seq   = target_seq[idx].detach().cpu().squeeze()       # (T, H, W)
        pred_seq = output_seq[idx].detach().cpu().squeeze()   # (T, H, W)
        diff_seq = pred_seq - gt_seq                     # signed error

        # ----------- set up a long thin figure --------------------------------
        fig_height = 3          # one row per line, in inches
        fig_width  = max(6, T)
        fig, axes  = plt.subplots(
            3, T,
            figsize=(fig_width, fig_height),
            gridspec_kw={"wspace": 0.005, "hspace": 0.03},  # Reduced spacing between elements
        )

        # make axes always iterable in both dims
        if T == 1:
            axes = axes.reshape(3, 1)

        # ----------- plot -----------------------------------------------------
        for t in range(T):
            # top row – ground truth
            axes[0, t].imshow(gt_seq[t*subsample_t], cmap="gray", vmin=0, vmax=1)
            # centre row – predictions
            axes[1, t].imshow(pred_seq[t*subsample_t], cmap="gray", vmin=0, vmax=1)
            # bottom row – signed difference
            axes[2, t].imshow(
                diff_seq[t*subsample_t],
                cmap="bwr",
                vmin=-vmax_diff,
                vmax=vmax_diff,
            )

            # cosmetic clean-up
            for r in range(3):
                axes[r, t].axis("off")

        # label the rows once (left-most subplot)
        axes[0, 0].set_ylabel("GT",    rotation=0, labelpad=20, fontsize=10)
        axes[1, 0].set_ylabel("Pred",  rotation=0, labelpad=15, fontsize=10)
        axes[2, 0].set_ylabel("Error", rotation=0, labelpad=18, fontsize=10)

        # optional overall title
        fig.suptitle(f"{split_name} sample {idx}", fontsize=12)

        # ----------- log to wandb & close -------------------------------------
        wandb.log({f"{split_name}sequence_{idx}": wandb.Image(fig)})

        if log_all_frames:
            # log all frames as an mp4 video
            frames = pred_seq.numpy()
            # scale to 0-255
            frames = (frames * 255.0).clip(0, 255)
            frames = frames.astype(np.uint8)
            frames = np.stack([frames]*3, axis=-1)     # (T, H, W, 3) grayscale -> RGB
            clip = ImageSequenceClip(list(frames), fps=10)  # adjust fps as needed
            video_dir = './mnist_videos'
            video_path = f"{video_dir}/{wandb.run.name}/{split_name}_sequence_{idx}_video.mp4"
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            clip.write_videofile(video_path, codec="libx264")
            wandb.log({f"{split_name}sequence_{idx}_video": wandb.Video(video_path, fps=10, format="mp4")})
            print(f"Logged video to {video_path}")


        plt.close(fig)


def _log_single_sequence(true_seq, pred_seq, step, dataset_type, sample_idx):
    """Helper function to log a single sequence with spacing between frames."""
    frames = []
    spacing = 4  # Number of pixels for spacing
    for t in range(true_seq.size(0)):
        gt_frame = true_seq[t, 0].cpu()   # [H, W]
        pred_frame = pred_seq[t, 0].cpu() # [H, W]
        H, W = gt_frame.shape

        # Create a vertical spacer (all zeros, same height, width=spacing)
        spacer = torch.zeros(H, spacing, dtype=gt_frame.dtype, device=gt_frame.device)

        # Concatenate: [gt_frame | spacer | pred_frame]
        grid = torch.cat([gt_frame, spacer, pred_frame], dim=1)  # [H, W+spacing+W]
        grid = (grid * 255.0).clamp(0, 255).byte()
        grid_rgb = grid.unsqueeze(0).repeat(3, 1, 1)     # [3, H, 2*W+spacing]
        frames.append(grid_rgb)

    video = torch.stack(frames)  # [T, 3, H, 2*W+spacing]
    wandb.log({f"qualitative/{dataset_type}_sample_{sample_idx}": wandb.Video(video.numpy(), fps=4, format="gif")})

# -----------------------------------------------------------------------------
#  Epoch loops
# -----------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimiser, device, epoch, args, n_seq_log_iters=100, val_loader=None, global_step=0):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        inputs = batch["input_seq"].to(device)
        in_actions = batch["input_actions"].to(device)
        tgt_actions = batch["target_actions"].to(device)
        targets = batch["target_seq"].to(device)

        optimiser.zero_grad()
        preds = model(inputs, in_actions, tgt_actions, target_seq=targets, teacher_forcing_ratio=args.teacher_forcing)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        running_loss += loss.item() * inputs.size(0)
        
        # Update progress bar with current loss
        current_loss = loss.item()
        avg_loss = running_loss / ((step + 1) * inputs.size(0))
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})

        # --- wandb per-step logging (optional) ---
        current_global_step = global_step + step
        wandb.log({"train/step_loss": loss.item()})
        
        # Log sequences during training every n_seq_log_iters
        if step % n_seq_log_iters == 0 and val_loader is not None:
            # Log validation sequences
            log_sequences(model, val_loader, device, current_global_step, dataset_type="val", max_videos=2, max_batches=1)
            
            # Log training sequences from current batch
            batch_data = {"input_seq": inputs, "target_seq": targets, "input_actions": in_actions, "target_actions": tgt_actions}
            log_sequences(model, batch_data, device, current_global_step, dataset_type="train", max_videos=2)

            # Log the linear layer weights as an image for linearRNN (to visualzed learned equivariance)
            # linear_weights = model.cell.linear_u.weight.cpu()
            # wandb.log({"model/linear_weights": wandb.Image(linear_weights)}, step=current_global_step)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def eval_epoch(model, loader, criterion, device, args):
    model.eval()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for step, batch in enumerate(pbar):
            inputs = batch["input_seq"].to(device)
            in_actions = batch["input_actions"].to(device)
            tgt_actions = batch["target_actions"].to(device)
            targets = batch["target_seq"].to(device)

            preds = model(inputs, in_actions, tgt_actions)
            loss = criterion(preds, targets)
            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar with current loss
            current_loss = loss.item()
            avg_loss = running_loss / ((step + 1) * inputs.size(0))
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})

    return running_loss / len(loader.dataset)

def eval_len_generalization(model, dataloader, device, input_frames, subsample_t=4, log_all_frames=False):
    """
    Returns:
        mean_err  – numpy array [T]  (MSE at each future step, averaged over test set)
        std_err   – numpy array [T]  (sample‑wise std at each step)
    """
    model.eval()
    first_pass = True
    with torch.no_grad():
        n_sequences = 0
        pbar = tqdm(dataloader, desc="Evaluating Length Generalization", leave=False)
        for batch in pbar:
            inputs = batch["input_seq"].to(device)
            in_actions = batch["input_actions"].to(device)
            tgt_actions = batch["target_actions"].to(device)
            targets = batch["target_seq"].to(device)

            preds = model(inputs, in_actions, tgt_actions)

            # MSE per example per timestep  →  [B, T]
            per_ex_t = ((preds - targets)**2).mean(dim=(2, 3, 4))  # assume (B,T,C,H,W)
            if first_pass:
                sum_err  = per_ex_t.sum(dim=0)          # [T]
                sum_err2 = (per_ex_t**2).sum(dim=0)     # [T]
                first_pass = False

                log_sequence_predictions_new(inputs, targets, preds, split_name="len_gen", num_samples=10, device=device, subsample_t=subsample_t, log_all_frames=log_all_frames)
            else:
                sum_err  += per_ex_t.sum(dim=0)
                sum_err2 += (per_ex_t**2).sum(dim=0)

            n_sequences += per_ex_t.size(0)
            
            # Update progress bar with current batch size
            pbar.set_postfix({"loss": per_ex_t.mean()})

    mean = sum_err / n_sequences
    var  = sum_err2 / n_sequences - mean**2
    std  = torch.sqrt(torch.clamp(var, min=0.0))
    return mean.cpu().numpy(), std.cpu().numpy()

# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Flow-Equivariant RNN on MNIST-World with wandb logging")

    # Dataset params
    parser.add_argument("--world_size", type=int, default=50)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--num_digits", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--target_seq_len", type=int, default=20)

    # Training params
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_seq_log_iters", type=int, default=100, help="Log sequences every N iterations during training")
    parser.add_argument("--teacher_forcing", type=float, default=0.0, help="Teacher forcing ratio (0.0 to 1.0)")
    parser.add_argument("--cell_type", type=str, default="v_channels", help="Cell type: 'v_channels' or 'action_concat'")
    parser.add_argument("--use_mlp_decoder", action="store_true", default=False)
    parser.add_argument("--use_mlp_encoder", action="store_true", default=False)
    parser.add_argument("--model_type", type=str, default="fernn", choices=["fernn", "baseline"], help="Model type: 'fernn' or 'baseline'")
    parser.add_argument("--decoder_conv_layers", type=int, default=1, help="Number of decoder conv layers")
    parser.add_argument("--data_dir", type=str, default="./data/mnist_world/", help="Directory with the dataset")
    parser.add_argument("--split", type=str, default="dynamic_training", help="Split to use for training")
    
    # Evaluation params
    parser.add_argument("--eval_only", action="store_true", default=False, help="Whether to evaluate only")
    parser.add_argument("--run_len_gen", action="store_true", default=False, help="Whether to generate the run length generalization experiments")
    
    # FERNN params
    parser.add_argument("--v_range", type=int, default=0, help="Velocity range for FERNN")
    parser.add_argument("--no_self_motion_equivariance", action="store_true", default=False, help="Whether to use self-motion equivariance")

    # wandb params
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_project", default="mnist-world")
    parser.add_argument("--wandb_dir", default="./wandb/")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--model_save_dir", type=str, default='./checkpoints/', help='Directory to save model checkpoints')
    parser.add_argument("--load_model", type=str, default=None, help='Path to a saved model checkpoint to load for training or evaluation')

    # Dataset params
    parser.add_argument("--max_overlap_frac", type=float, default=None, help="Maximum overlap fraction between digits, None means ignore this, 0.0 means no overlap")
    parser.add_argument("--unique_digits", action="store_true", default=False, help="Whether to use unique digits in the dataset, 0 through 9")

    args = parser.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    #  Initialise wandb
    # ------------------------------------------------------------------
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.run_name,
        dir=args.wandb_dir,
        config=vars(args),
    )

    # ------------------------------------------------------------------
    #  Data
    # ------------------------------------------------------------------
    transform = None

    train_split, val_split = get_train_val_splits(args.split)

    print(f"Using train split: {train_split} and val split: {val_split}")

    train_ds = MNISTWorldDynamicDataset_FromFiles(
        root_dir=args.data_dir,
        split=train_split,
        seq_len=args.seq_len,
        target_seq_len=args.target_seq_len,
        random_window=False,
    )

    val_ds = MNISTWorldDynamicDataset_FromFiles(
        root_dir=args.data_dir,
        split=val_split,
        seq_len=args.seq_len,
        target_seq_len=args.target_seq_len,
        random_window=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
    )

    if args.run_len_gen:
        if val_split.endswith("vis"):
            len_gen_split = val_split
        else:   
            len_gen_split = val_split + "_200"

        len_gen_val_ds = MNISTWorldDynamicDataset_FromFiles(
            root_dir=args.data_dir,
            split=len_gen_split,
            seq_len=args.seq_len,
            target_seq_len=200 - args.seq_len,
            random_window=False,
        )
        len_gen_val_loader = DataLoader(
            len_gen_val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
        )

    # ------------------------------------------------------------------
    #  Model / optimiser / loss
    # ------------------------------------------------------------------
    if args.model_type == "fernn":
        model = FlowEquivariantRNN(input_channels=1, 
                                   hidden_channels=args.hidden_channels, 
                                   world_size=args.world_size, 
                                   window_size=args.window_size,
                                   use_mlp_decoder=args.use_mlp_decoder,
                                   use_mlp_encoder=args.use_mlp_encoder,
                                   decoder_conv_layers=args.decoder_conv_layers,
                                   cell_type=args.cell_type,
                                   v_range=args.v_range,
                                   self_motion_equivariance=not args.no_self_motion_equivariance)
    elif args.model_type == "baseline":
        model = LinearRNNBaseline(input_channels=1,
                                  hidden_dim=args.hidden_channels,
                                  window_size=args.window_size,
                                  output_channels=1,
                                  use_mlp_decoder=args.use_mlp_decoder)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    model.to(device)

    # Load model if specified
    if args.load_model is not None:
        print(f"Loading model from {args.load_model}")
        try:
            checkpoint = torch.load(args.load_model, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                # Loading from a checkpoint with model_state, optim_state, etc.
                model.load_state_dict(checkpoint['model_state'])
                print(f"Successfully loaded model weights from checkpoint.")
                
                # Log the loaded checkpoint info
                if 'epoch' in checkpoint:
                    print(f"Checkpoint was saved at epoch {checkpoint['epoch']}")
                if 'val_loss' in checkpoint:
                    print(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")
                    
            else:
                # Loading from a model state dict directly
                model.load_state_dict(checkpoint)
                print(f"Successfully loaded model weights from state dict.")
                
        except Exception as e:
            print(f"Failed to load model from {args.load_model}: {e}")
            print("Exiting due to failed model load.")
            exit(1)

    # Log total parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {param_count}")
    wandb.log({"model/parameter_count": param_count})

    criterion = nn.MSELoss() # nn.BCEWithLogitsLoss()
    optimiser = Adam(model.parameters(), lr=args.lr)

    # Load optimizer state if available in checkpoint
    start_epoch = 1
    if args.load_model is not None:
        try:
            checkpoint = torch.load(args.load_model, map_location=device)
            if isinstance(checkpoint, dict) and 'optim_state' in checkpoint:
                optimiser.load_state_dict(checkpoint['optim_state'])
                print(f"Successfully loaded optimizer state from checkpoint.")
                
                # Resume training from the next epoch
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resuming training from epoch {start_epoch}")
                    
        except Exception as e:
            print(f"Failed to load optimizer state: {e}")
            print("Continuing with fresh optimizer state...")

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------
    best_val = float("inf")
    global_step = 0

    # Create model save directory if it doesn't exist
    os.makedirs(args.model_save_dir, exist_ok=True)

    # Load best validation loss if resuming from checkpoint
    if args.load_model is not None:
        try:
            checkpoint = torch.load(args.load_model, map_location=device)
            if isinstance(checkpoint, dict) and 'val_loss' in checkpoint:
                best_val = checkpoint['val_loss']
                print(f"Resuming with best validation loss: {best_val:.4f}")
        except Exception as e:
            print(f"Failed to load best validation loss: {e}")

    if args.eval_only:
        model.eval()
        print("Evaluating only")
        val_loss = eval_epoch(model, val_loader, criterion, device, args)
        wandb.log({
            "val/epoch_loss": val_loss,
        })

        print("Evaluating length generalization")
        len_gen_val_loss_mean, len_gen_val_loss_std = eval_len_generalization(model, len_gen_val_loader, device, args.seq_len, log_all_frames=True)

        wandb.log({f"len_gen/len_gen_mean_t{t+1}": len_gen_val_loss_mean[t] for t in range(len(len_gen_val_loss_mean))})
        wandb.log({f"len_gen/len_gen_std_t{t+1}":  len_gen_val_loss_std[t]  for t in range(len(len_gen_val_loss_std))})
        wandb.log({f"len_gen/len_gen_mean_mean_over_time": len_gen_val_loss_mean.mean()})

        exit()


    for epoch in range(start_epoch, args.epochs + 1):
        t0 = datetime.now()

        train_loss = train_epoch(model, train_loader, criterion, optimiser, device, epoch, args, n_seq_log_iters=args.n_seq_log_iters, val_loader=val_loader, global_step=global_step)
        val_loss = eval_epoch(model, val_loader, criterion, device, args)
        
        if args.run_len_gen:
            len_gen_val_loss_mean, len_gen_val_loss_std = eval_len_generalization(model, len_gen_val_loader, device, args.seq_len)

            wandb.log({f"len_gen/len_gen_mean_t{t+1}": len_gen_val_loss_mean[t] for t in range(len(len_gen_val_loss_mean))})
            wandb.log({f"len_gen/len_gen_std_t{t+1}":  len_gen_val_loss_std[t]  for t in range(len(len_gen_val_loss_std))})
            wandb.log({f"len_gen/len_gen_mean_mean_over_time": len_gen_val_loss_mean.mean()})

            # ---- wandb line plot ----
            steps = np.arange(1, 200 - args.seq_len + 1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(steps, len_gen_val_loss_mean, label="MSE")
            ax.fill_between(steps,
                            (len_gen_val_loss_mean - len_gen_val_loss_std).clip(min=0),
                            len_gen_val_loss_mean + len_gen_val_loss_std,
                            alpha=0.3,
                            label="± std")
            ax.set_xlabel("Prediction horizon t")
            ax.set_ylabel("MSE")
            ax.set_title(f"Length generalization (seq_len = 200)")
            ax.legend()
            wandb.log({f"len_gen_curve": wandb.Image(fig)})
            plt.close(fig)

        global_step += len(train_loader)

        wandb.log({
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "epoch": epoch,
        })

        # Log sequences at end of epoch
        log_sequences(model, val_loader, device, epoch, dataset_type="val", max_videos=4, max_batches=1)
        log_sequences(model, train_loader, device, epoch, dataset_type="train", max_videos=4, max_batches=1)

        dt = (datetime.now() - t0).total_seconds()
        print(f"Epoch {epoch:03d}: train={train_loss:.4f} | val={val_loss:.4f} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            
            # Save best model using the same scheme as train.py
            model_filename = f"{args.model_type}_{wandb.run.id}"
            model_path = os.path.join(args.model_save_dir, model_filename)
            
            # Also save checkpoint for compatibility
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "val_loss": val_loss,
            }, model_path, epoch)
            print(f"Saved new best model at epoch {epoch} for {args.model_type} at {model_path}")

            # Log model path and best validation loss to wandb
            wandb.log({
                "best_model_path": model_path,
                "best_val_loss": val_loss,
            })

    print(f"[✓] Training finished. Best val loss: {best_val} at path: {model_path}")


if __name__ == "__main__":
    main()
