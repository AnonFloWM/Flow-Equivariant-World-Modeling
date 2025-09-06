import os
import math
import argparse
from typing import Tuple, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

from mnist_world_dataset import MNISTWorldDataset
import multiprocessing as mp 


"""
This file generates and saves the MNIST World (static world, moving camera) dataset
for a given number of examples.

It saves the dataset with the following structure:

- <output_dir>/
  - example_000000.mp4                    # Camera view video (input + target sequences)
  - example_000000.pt                     # Metadata file
  - example_000000_fullworld.mp4          # Full world view (if --render_full_world)
  - example_000000_fullworld_withcam.mp4  # Full world with camera window highlighted (if --render_full_world_with_cam)
  - ...

The .pt file contains the metadata for the example. The .mp4 file contains the
video of the example as the input sequence concatenated with the target sequence.

python -m scripts.generate_mnist_world   --num_examples 8000   --output_dir ./data/mnist_world/static_validation  \
    --train   --world_size 50   --num_digits 5   --window_size 32   --seq_len 50   --target_seq_len 20  --step_size 10 --num_workers 32 --shard_size 256
    
python -m scripts.generate_mnist_world   --num_examples 8000   --output_dir ./data/mnist_world/static_validation_200  \
    --train   --world_size 50   --num_digits 5   --window_size 32   --seq_len 50   --target_seq_len 150  --step_size 10 --num_workers 32 --shard_size 256

python -m scripts.generate_mnist_world   --num_examples 180000   --output_dir ./data/mnist_world/static_training  \
    --train   --world_size 50   --num_digits 5   --window_size 32   --seq_len 50   --target_seq_len 20   --step_size 10 --num_workers 32

# Example with full world view and camera window highlighted:
python -m scripts.generate_mnist_world   --num_examples 8   --output_dir ./data/mnist_world/static_validation_200_vis  \
    --train   --world_size 80   --num_digits 8   --window_size 32   --seq_len 50   --target_seq_len 150  --render_full_world --render_full_world_with_cam --step_size 10 --num_workers 1 --shard_size 8

# Example with biased rollout for straighter target sequences:
python -m scripts.generate_mnist_world   --num_examples 8   --output_dir ./data/mnist_world/static_validation_biased_200_vis_v2  \
    --train   --world_size 80   --num_digits 8   --window_size 32   --seq_len 50   --target_seq_len 150  --render_full_world --render_full_world_with_cam --straightline_biased_rollout --forward_probability 0.95 --step_size 10 --num_workers 1 --shard_size 8
"""


def save_video_mp4(frames: torch.Tensor, path: str, fps: int = 8) -> None:
    """
    Save a sequence of frames to an MP4 file using imageio-ffmpeg.

    frames: Tensor [T, C, H, W] or [T, H, W] in [0,1] or uint8 0..255
    """
    import imageio

    if isinstance(frames, torch.Tensor):
        frames_np = frames.detach().cpu().numpy()
    else:
        frames_np = np.asarray(frames)

    if frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np, 0, 1) * 255.0
        frames_np = frames_np.astype(np.uint8)

    # Ensure [T, H, W, 3]
    if frames_np.ndim == 3:  # [T, H, W]
        frames_np = np.repeat(frames_np[:, :, :, None], 3, axis=-1)
    elif frames_np.ndim == 4:
        if frames_np.shape[1] in (1, 3):
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))
        c = frames_np.shape[-1]
        if c == 1:
            frames_np = np.repeat(frames_np, 3, axis=-1)
        elif c != 3:
            raise ValueError("Frames must have 1 or 3 channels")
    else:
        raise ValueError("Unexpected frames ndim; expected 3 or 4 dims")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames_np:
            writer.append_data(frame)


def add_camera_window_box(frames: torch.Tensor, camera_centers: torch.Tensor, window_size: int, world_size: int) -> torch.Tensor:
    """
    Add a red box around the camera window on each frame.
    
    frames: Tensor [T, 1, H, W] - full world frames
    camera_centers: Tensor [T, 2] - camera center positions (cx, cy) for each frame
    window_size: int - size of the camera window
    world_size: int - size of the full world
    
    Returns: Tensor [T, 3, H, W] with red box drawn around camera window
    """
    import cv2
    
    # Convert to numpy and ensure 3 channels
    if isinstance(frames, torch.Tensor):
        frames_np = frames.detach().cpu().numpy()
    else:
        frames_np = np.asarray(frames)
    
    # Ensure [T, H, W, 3] format with proper memory layout
    if frames_np.ndim == 4:
        if frames_np.shape[1] == 1:  # [T, 1, H, W]
            frames_np = np.repeat(frames_np, 3, axis=1)  # [T, 3, H, W]
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))  # [T, H, W, 3]
    
    # Ensure contiguous memory layout for OpenCV
    frames_np = np.ascontiguousarray(frames_np)
    
    camera_centers_np = camera_centers.detach().cpu().numpy()
    half_window = window_size // 2
    
    # Draw red box on each frame
    for t in range(frames_np.shape[0]):
        frame = frames_np[t].copy()  # Make a copy to avoid modifying original
        cx, cy = camera_centers_np[t]
        
        # Calculate box coordinates
        x1 = int(cx - half_window)
        y1 = int(cy - half_window)
        x2 = int(cx + half_window)
        y2 = int(cy + half_window)
        
        # Draw rectangles for toroidal wrapping
        # Main rectangle (clamped to world bounds)
        main_x1 = max(0, min(x1, world_size - 1))
        main_y1 = max(0, min(y1, world_size - 1))
        main_x2 = max(0, min(x2, world_size - 1))
        main_y2 = max(0, min(y2, world_size - 1))
        
        # Only draw main rectangle if it has positive dimensions
        if main_x2 > main_x1 and main_y2 > main_y1:
            cv2.rectangle(frame, (main_x1, main_y1), (main_x2, main_y2), (0, 0, 255), 2)
        
        # Handle X-axis wrapping (left side)
        if x1 < 0:
            wrap_x1 = world_size + x1
            wrap_y1 = max(0, min(y1, world_size - 1))
            wrap_x2 = world_size - 1
            wrap_y2 = max(0, min(y2, world_size - 1))
            if wrap_x2 > wrap_x1 and wrap_y2 > wrap_y1:
                cv2.rectangle(frame, (wrap_x1, wrap_y1), (wrap_x2, wrap_y2), (0, 0, 255), 2)
        
        # Handle X-axis wrapping (right side)
        if x2 >= world_size:
            wrap_x1 = 0
            wrap_y1 = max(0, min(y1, world_size - 1))
            wrap_x2 = x2 - world_size
            wrap_y2 = max(0, min(y2, world_size - 1))
            if wrap_x2 > wrap_x1 and wrap_y2 > wrap_y1:
                cv2.rectangle(frame, (wrap_x1, wrap_y1), (wrap_x2, wrap_y2), (0, 0, 255), 2)
        
        # Handle Y-axis wrapping (top side)
        if y1 < 0:
            wrap_x1 = max(0, min(x1, world_size - 1))
            wrap_y1 = world_size + y1
            wrap_x2 = max(0, min(x2, world_size - 1))
            wrap_y2 = world_size - 1
            if wrap_x2 > wrap_x1 and wrap_y2 > wrap_y1:
                cv2.rectangle(frame, (wrap_x1, wrap_y1), (wrap_x2, wrap_y2), (0, 0, 255), 2)
        
        # Handle Y-axis wrapping (bottom side)
        if y2 >= world_size:
            wrap_x1 = max(0, min(x1, world_size - 1))
            wrap_y1 = 0
            wrap_x2 = max(0, min(x2, world_size - 1))
            wrap_y2 = y2 - world_size
            if wrap_x2 > wrap_x1 and wrap_y2 > wrap_y1:
                cv2.rectangle(frame, (wrap_x1, wrap_y1), (wrap_x2, wrap_y2), (0, 0, 255), 2)
        
        # Handle corner wrapping (when both X and Y wrap)
        # Top-left corner
        if x1 < 0 and y1 < 0:
            corner_x1 = world_size + x1
            corner_y1 = world_size + y1
            corner_x2 = world_size - 1
            corner_y2 = world_size - 1
            cv2.rectangle(frame, (corner_x1, corner_y1), (corner_x2, corner_y2), (0, 0, 255), 2)
        
        # Top-right corner
        if x2 >= world_size and y1 < 0:
            corner_x1 = 0
            corner_y1 = world_size + y1
            corner_x2 = x2 - world_size
            corner_y2 = world_size - 1
            cv2.rectangle(frame, (corner_x1, corner_y1), (corner_x2, corner_y2), (0, 0, 255), 2)
        
        # Bottom-left corner
        if x1 < 0 and y2 >= world_size:
            corner_x1 = world_size + x1
            corner_y1 = 0
            corner_x2 = world_size - 1
            corner_y2 = y2 - world_size
            cv2.rectangle(frame, (corner_x1, corner_y1), (corner_x2, corner_y2), (0, 0, 255), 2)
        
        # Bottom-right corner
        if x2 >= world_size and y2 >= world_size:
            corner_x1 = 0
            corner_y1 = 0
            corner_x2 = x2 - world_size
            corner_y2 = y2 - world_size
            cv2.rectangle(frame, (corner_x1, corner_y1), (corner_x2, corner_y2), (0, 0, 255), 2)
        
        # Update the frame in the array
        frames_np[t] = frame
    
    # Convert back to [T, 3, H, W] format
    frames_with_box = np.transpose(frames_np, (0, 3, 1, 2))
    return torch.from_numpy(frames_with_box)








def calculate_camera_positions_from_actions(actions: torch.Tensor, world_size: int, start_pos: Tuple[int, int] = (0, 0)) -> torch.Tensor:
    """
    Calculate camera center positions from action sequences.
    
    actions: Tensor [T, 2] - (dx, dy) actions
    world_size: int - size of the world for toroidal wrapping
    start_pos: Tuple[int, int] - starting camera position (default: (0, 0))
    
    Returns: Tensor [T, 2] - camera center positions (cx, cy) for each frame
    """
    cx, cy = start_pos
    positions = []
    
    for dx, dy in actions:
        cx = (cx + dx) % world_size  # Apply toroidal wrapping
        cy = (cy + dy) % world_size
        positions.append([cx, cy])
    
    return torch.tensor(positions, dtype=torch.long)


def create_full_world_frames(world: torch.Tensor, camera_centers: torch.Tensor, window_size: int, world_size: int) -> torch.Tensor:
    """
    Create full world frames showing the static world with camera window positions.
    
    world: Tensor [1, H, W] - static world
    camera_centers: Tensor [T, 2] - camera center positions
    window_size: int - size of the camera window
    world_size: int - size of the full world
    
    Returns: Tensor [T, 1, H, W] - full world frames for each timestep
    """
    # Repeat the static world for each timestep
    world_frames = world.unsqueeze(0).repeat(len(camera_centers), 1, 1, 1)
    return world_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MNIST World dataset samples and save as MP4 + .pt metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output and count
    parser.add_argument("--num_examples", type=int, required=True, help="Number of examples to generate")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs (MP4 and .pt)")

    # MNIST data root and split
    parser.add_argument("--mnist_root", type=str, default="./data", help="Torchvision MNIST root directory")
    parser.add_argument("--train", action="store_true", help="Use MNIST train split (default: test)")

    # Core dataset params (as requested)
    parser.add_argument("--world_size", type=int, default=50)
    parser.add_argument("--num_digits", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--target_seq_len", type=int, default=20)
    parser.add_argument("--step_size", type=int, default=10)

    # Optional: reproducibility and convenience
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8, help="FPS for saved videos")
    parser.add_argument("--shuffle_examples", action="store_true", help="Randomize dataset indexing per example")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--shard_size", type=int, default=1024, help="Number of samples per subfolder shard")

    # Expose a couple of useful placement controls
    parser.add_argument("--max_overlap_frac", type=float, default=None, help="Max allowed overlap fraction (0..1); None to allow overlap with wrap")
    parser.add_argument("--unique_digits", action="store_true", help="Enforce unique labels among placed digits")

    # Optionally render full-world view in addition to the camera window
    parser.add_argument("--render_full_world", action="store_true", help="If set, save full-world videos (world_size x world_size) for each example")
    parser.add_argument("--render_full_world_with_cam", action="store_true", help="If set, save full-world videos with camera window highlighted (requires --render_full_world)")
    
    # Biased rollout options for target sequence
    parser.add_argument("--straightline_biased_rollout", action="store_true", help="If set, use biased sampling for target sequence to favor forward movement")
    parser.add_argument("--forward_probability", type=float, default=0.93, help="Probability of continuing in the same direction during biased rollout (default: 0.93)")
    parser.add_argument("--constant_velocity", action="store_true", help="Use constant step size (step_size) instead of random step sizes within range for biased rollout")

    args = parser.parse_args()
    
    # Validate that render_full_world_with_cam requires render_full_world
    if args.render_full_world_with_cam and not args.render_full_world:
        parser.error("--render_full_world_with_cam requires --render_full_world to be set")
    
    # Validate forward probability is in valid range
    if args.forward_probability < 0.0 or args.forward_probability > 1.0:
        parser.error("--forward_probability must be between 0.0 and 1.0")
    
    return args


def _generate_group(group_id: int, group_count: int, start_local: int, args_dict: Dict[str, Any]) -> int:
    """
    Generate one shard group with up to group_count samples. Returns number of samples generated.
    """
    output_dir = args_dict["output_dir"]
    mnist_root = args_dict["mnist_root"]
    train = args_dict["train"]
    world_size = args_dict["world_size"]
    num_digits = args_dict["num_digits"]
    window_size = args_dict["window_size"]
    seq_len = args_dict["seq_len"]
    target_seq_len = args_dict["target_seq_len"]
    step_size = args_dict["step_size"]
    fps = args_dict["fps"]
    render_full_world = args_dict.get("render_full_world", False)
    render_full_world_with_cam = args_dict.get("render_full_world_with_cam", False)
    straightline_biased_rollout = args_dict.get("straightline_biased_rollout", False)
    forward_probability = args_dict.get("forward_probability", 0.93)
    shuffle_examples = args_dict["shuffle_examples"]
    seed = args_dict["seed"] + group_id
    max_overlap_frac = args_dict["max_overlap_frac"]
    unique_digits = args_dict["unique_digits"]

    group_dir = os.path.join(output_dir, f"{group_id}")
    os.makedirs(group_dir, exist_ok=True)

    dataset = MNISTWorldDataset(
        root=mnist_root,
        train=train,
        world_size=world_size,
        num_digits=num_digits,
        window_size=window_size,
        seq_len=seq_len,
        target_seq_len=target_seq_len,
        step_size=step_size,
        ensure_full_coverage=True,
        return_metadata=(render_full_world_with_cam or render_full_world),
        straightline_biased_rollout=straightline_biased_rollout,
        forward_probability=forward_probability,
        constant_velocity=args_dict.get("constant_velocity", False),
        download=False,
        seed=seed,
        max_overlap_frac=max_overlap_frac,
        unique_digits=unique_digits,
    )

    for local_i in range(group_count):
        index = int(torch.randint(low=0, high=len(dataset), size=(1,)).item()) if shuffle_examples else local_i
        sample = dataset[index]

        input_seq: torch.Tensor = sample["input_seq"]
        target_seq: torch.Tensor = sample["target_seq"]
        input_actions: torch.Tensor = sample["input_actions"]
        target_actions: torch.Tensor = sample["target_actions"]



        frames = torch.cat([input_seq, target_seq], dim=0)
        actions = torch.cat([input_actions, target_actions], dim=0)

        stem = f"{start_local + local_i:04d}"
        mp4_path = os.path.join(group_dir, f"{stem}.mp4")
        if not os.path.exists(mp4_path):
            save_video_mp4(frames, mp4_path, fps=fps)

        # Optionally save full-world mp4 alongside the standard one
        if render_full_world and ("world_seq" in sample):
            # Use dataset-provided full world frames and camera centers
            world_frames: torch.Tensor = sample["world_seq"]
            
            full_mp4_path = os.path.join(group_dir, f"{stem}_fullworld.mp4")
            if not os.path.exists(full_mp4_path):
                save_video_mp4(world_frames, full_mp4_path, fps=fps)
            
            # Optionally save full-world with camera window highlighted
            if render_full_world_with_cam and ("camera_centers_input" in sample) and ("camera_centers_target" in sample):
                # Concatenate camera centers for input and target sequences
                camera_centers = torch.cat([
                    sample["camera_centers_input"],
                    sample["camera_centers_target"],
                ], dim=0)

                # Add red box around camera window
                world_with_cam = add_camera_window_box(
                    world_frames,
                    camera_centers,
                    window_size,
                    world_size,
                )
                
                full_with_cam_mp4_path = os.path.join(group_dir, f"{stem}_fullworld_withcam.mp4")
                if not os.path.exists(full_with_cam_mp4_path):
                    save_video_mp4(world_with_cam, full_with_cam_mp4_path, fps=fps)

        metadata = {k: v for k, v in sample.items() if k not in ("input_seq", "target_seq", "input_actions", "target_actions", "world_seq")}
        payload = {
            "actions": actions,
            **metadata,
        }
        pt_path = os.path.join(group_dir, f"{stem}.pt")
        torch.save(payload, pt_path)

    return group_count


def _worker_setup(base_seed: int):
    # Keep numerical libraries from overspawning threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    # Give each worker a unique RNG stream, derived from base_seed, PID and time
    try:
        import time
        pid = os.getpid()
        # Mix base_seed with PID and time for uniqueness across workers
        unique_seed = int(base_seed) ^ int(pid) ^ int(time.time_ns() & 0xFFFFFFFF)
    except Exception:
        unique_seed = int(base_seed)
    try:
        np.random.seed(unique_seed)
    except Exception:
        pass
    try:
        import random
        random.seed(unique_seed)
    except Exception:
        pass

def _plan_append_tasks(output_dir: str, num_to_generate: int, shard_size: int) -> List[Tuple[int, int, int]]:
    """
    Plan non-overwriting generation tasks.
    Returns a list of (group_id, start_local, count) entries.
    """
    existing_groups: List[int] = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            p = os.path.join(output_dir, name)
            if os.path.isdir(p) and name.isdigit():
                existing_groups.append(int(name))
    tasks: List[Tuple[int, int, int]] = []
    remaining = num_to_generate

    start_gid = 0
    if existing_groups:
        last_gid = max(existing_groups)
        last_dir = os.path.join(output_dir, str(last_gid))
        os.makedirs(last_dir, exist_ok=True)
        stems = []
        for fname in os.listdir(last_dir):
            if fname.endswith('.mp4') or fname.endswith('.pt'):
                stem, _ = os.path.splitext(fname)
                if stem.isdigit():
                    stems.append(int(stem))
        next_stem = (max(stems) + 1) if stems else 0
        if next_stem < shard_size and remaining > 0:
            cnt = min(remaining, shard_size - next_stem)
            tasks.append((last_gid, next_stem, cnt))
            remaining -= cnt
        start_gid = last_gid + 1

    gid = start_gid
    while remaining > 0:
        cnt = min(remaining, shard_size)
        tasks.append((gid, 0, cnt))
        remaining -= cnt
        gid += 1
    return tasks


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Prime MNIST download once
    _ = MNISTWorldDataset(
        root=args.mnist_root,
        train=args.train,
        world_size=args.world_size,
        num_digits=args.num_digits,
        window_size=args.window_size,
        seq_len=args.seq_len,
        target_seq_len=args.target_seq_len,
        step_size=args.step_size,
        ensure_full_coverage=True,
        return_metadata=False,
        download=True,
        seed=args.seed,
        max_overlap_frac=args.max_overlap_frac,
        unique_digits=args.unique_digits,
    )

    tasks = _plan_append_tasks(args.output_dir, args.num_examples, args.shard_size)
    args_dict: Dict[str, Any] = vars(args)

    generated = 0
    if args.num_workers == 1:
        for (group_id, start_local, cnt) in tqdm(tasks):
            generated += _generate_group(group_id, cnt, start_local, args_dict)
    else:
        ctx = mp.get_context("spawn")  # avoid forking after importing torch/ffmpeg
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=ctx,
            initializer=_worker_setup,
            initargs=(args.seed,),
        ) as ex:
            futures = []
            for (group_id, start_local, cnt) in tasks:
                print(f"Starting group {group_id} at local {start_local} for {cnt} examples")
                futures.append(ex.submit(_generate_group, group_id, cnt, start_local, args_dict))

            for f in tqdm(as_completed(futures), total=len(futures)):
                try:
                    generated += f.result()
                except Exception as e:
                    import traceback
                    print("Worker failed with exception:\n", "".join(traceback.format_exception(e)))

    print(f"Saved {generated} examples to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()


