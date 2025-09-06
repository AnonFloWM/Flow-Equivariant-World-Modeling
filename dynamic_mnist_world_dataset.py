import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
from typing import Tuple, List, Optional, Dict

__all__ = ["MNISTWorldDynamicDataset", "MNISTWorldDynamicDataset_FromFiles",  "get_train_val_splits"]

# Optional backends for reading video
_HAS_DECORD = False
try:
    from decord import VideoReader, cpu as decord_cpu
    _HAS_DECORD = True
except Exception:
    pass

try:
    # torchvision fallback
    from torchvision.io import read_video, read_video_timestamps
    _HAS_TORCHVISION_VIDEO = True
except Exception:
    _HAS_TORCHVISION_VIDEO = False


def get_train_val_splits(split_type: str):
    """
    Given a split type, return the (train_split, val_split) names.

    Args:
        split_type (str): One of
            'dynamic', 'static', 'dynamic_smallworld', 'static_smallworld',
            'dynamic_smallworld_no_em', 'static_smallworld_no_em',
            'dynamic_200', 'static_200'

    Returns:
        (train_split, val_split): Tuple of split names as strings.
    """
    mapping = {
        "dynamic": ("dynamic_training", "dynamic_validation"),

        "dynamic_vis": ("dynamic_validation_biased_200_vis", "dynamic_validation_biased_200_vis"),

        "static": ("static_training", "static_validation"),

        "static_vis": ("static_validation_biased_200_vis", "static_validation_biased_200_vis"),

        "dynamic_smallworld": ("dynamic_training_smallworld", "dynamic_validation_smallworld"),

        "dynamic_smallworld_vis": ("dynamic_validation_smallworld_biased_200_vis", "dynamic_validation_smallworld_biased_200_vis"),

        "dynamic_smallworld_no_em": ("dynamic_training_smallworld_no_em", "dynamic_validation_smallworld_no_em"),

        "dynamic_smallworld_no_em_vis": ("dynamic_validation_smallworld_no_em_biased_200_vis", "dynamic_validation_smallworld_no_em_biased_200_vis"),

        "dynamic_ws80": ("dynamic_training_ws80", "dynamic_validation_ws80"),

        "dynamic_ws80_vis": ("dynamic_validation_bigworld_biased_200_vis", "dynamic_validation_bigworld_biased_200_vis"),

    }

    if split_type not in mapping:
        raise ValueError(f"Unknown split_type '{split_type}'. Valid options: {list(mapping.keys())}")
    return mapping[split_type]


def _to_grayscale_chw_uint8(frames: np.ndarray) -> np.ndarray:
    """Convert frames from (T, H, W, C) RGB/gray to (T, 1, H, W) uint8 grayscale."""
    if frames.ndim != 4:
        raise ValueError(f"Expected (T, H, W, C), got {frames.shape}")
    T, H, W, C = frames.shape
    if C == 1:
        gray = frames[..., 0]
    elif C == 3:
        # Luma-ish conversion; uint8-safe by working in float then casting back
        gray = (0.2989 * frames[..., 0] + 0.5870 * frames[..., 1] + 0.1140 * frames[..., 2]).astype(np.float32)
    else:
        # Fallback: average channels
        gray = frames.mean(axis=-1).astype(np.float32)
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    gray = np.clip(gray, 0, 255)
    gray = gray.astype(np.uint8)
    # (T, 1, H, W)
    return gray[:, None, :, :]


def _load_video_segment(path: Path, start: int, end: int) -> torch.Tensor:
    """Return frames [start:end] as float in [0,1], shape (T, 1, H, W)."""
    if _HAS_DECORD:
        vr = VideoReader(str(path), ctx=decord_cpu())
        end = min(end, len(vr))
        idx = np.arange(start, end)
        if len(idx) == 0:
            return torch.empty(0, 1, 0, 0)
        batch = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
        gray = _to_grayscale_chw_uint8(batch)  # (T, 1, H, W) uint8
        return torch.from_numpy(gray).float() / 255.0
    elif _HAS_TORCHVISION_VIDEO:
        # torchvision doesn't have exact frame index slicing here.
        # We'll read full, then slice by index (ok for small 28x28 clips).
        vid, _, _ = read_video(str(path), pts_unit="sec")
        # vid: (T, H, W, C) uint8
        vid = vid.numpy()
        gray = _to_grayscale_chw_uint8(vid)
        gray = torch.from_numpy(gray).float() / 255.0
        return gray[start:end]
    else:
        raise ImportError("Neither decord nor torchvision video I/O is available.")


def _probe_num_frames(path: Path) -> int:
    if _HAS_DECORD:
        vr = VideoReader(str(path), ctx=decord_cpu())
        return len(vr)
    elif _HAS_TORCHVISION_VIDEO:
        pts, _ = read_video_timestamps(str(path), pts_unit="sec")
        return len(pts)
    else:
        raise ImportError("Neither decord nor torchvision video I/O is available.")


class MNISTWorldDynamicDataset_FromFiles(Dataset):
    """
    Dataset that **loads** MNIST-World sequences from disk (mp4 + .pt actions)
    and returns the **same format** as `MNISTWorldDynamicDataset`:

        item = {
            "input_seq":      FloatTensor [seq_len, 1, H, W]   in [0,1]
            "input_actions":  LongTensor  [seq_len, 2]         (first row = [0, 0])
            "target_seq":     FloatTensor [target_seq_len, 1, H, W]
            "target_actions": LongTensor  [target_seq_len, 2]  (first row = [0, 0])
        }

    Directory layout (example):
        top_dir = "./data/mnist_world/"
        split   = one of:
            "dynamic_training", "dynamic_training_smallworld", "dynamic_training_smallworld_no_em",
            "dynamic_validation", "dynamic_validation_200",
            "dynamic_validation_smallworld", "dynamic_validation_smallworld_200",
            "dynamic_validation_smallworld_no_em", "dynamic_validation_smallworld_no_em_200",
            "static_training", "static_validation", "static_validation_200"
        Within the split:
            {scene_id}/
                0000.mp4, 0000.pt, 0079.mp4, 0079.pt, ..., 0948.mp4, 0948.pt

    Notes:
      * We assume each mp4 has >= (seq_len+target_seq_len) frames. If longer,
        we will take a contiguous window (randomized if `random_window=True`).
      * Actions are loaded from the companion .pt under key "actions" and
        sliced to the same [start:end] window. The **first action** of each
        returned segment is forced to [0,0] to match the generator dataset.
    """

    def __init__(
        self,
        root_dir: str = "./data/mnist_world/",
        split: str = "dynamic_training",
        seq_len: int = 50,
        target_seq_len: int = 20,
        random_window: bool = False,
        min_total_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.seq_len = int(seq_len)
        self.target_seq_len = int(target_seq_len)
        self.total_len = self.seq_len + self.target_seq_len
        self.random_window = bool(random_window)
        self.min_total_len = int(min_total_len) if min_total_len is not None else self.total_len

        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Collect all (video_path, pt_path) pairs under split
        clips: List[Tuple[Path, Path]] = []
        for scene_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name.zfill(6)):
            for mp4 in sorted(scene_dir.glob("*.mp4")):
                pt = mp4.with_suffix(".pt")
                if not pt.exists():
                    # Skip videos without actions
                    continue
                clips.append((mp4, pt))

        if len(clips) == 0:
            raise RuntimeError(f"No valid (mp4, pt) pairs found in {split_dir}")

        # Filter by length (>= min_total_len)
        self.index: List[Tuple[Path, Path, int]] = []  # (mp4, pt, num_frames)
        for mp4, pt in clips:
            self.index.append((mp4, pt, self.seq_len + self.target_seq_len))

        if len(self.index) == 0:
            raise RuntimeError(
                f"No videos with >= {self.min_total_len} frames were found in {split_dir} (total clips scanned: {len(clips)})"
            )

    def __len__(self) -> int:
        # Each file (clip) is one dataset entry. If you'd like to generate multiple
        # windows per clip, you can extend this to enumerate start indices.
        return len(self.index)

    def _choose_window(self, num_frames: int) -> Tuple[int, int]:
        if num_frames < self.total_len:
            raise ValueError(f"Clip too short: {num_frames} < {self.total_len}")
        if self.random_window:
            start = int(np.random.randint(0, num_frames - self.total_len + 1))
        else:
            start = 0
        end = start + self.total_len
        return start, end

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mp4, pt, n = self.index[idx]
        start, end = self._choose_window(n)

        # 1) Load frames [start:end] -> (T, 1, H, W) float32 in [0,1]
        frames = _load_video_segment(mp4, start, end)  # (T, 1, H, W)

        # 2) Load actions and slice to [start:end]
        meta = torch.load(pt, map_location=torch.device("cpu"), weights_only=False)
        actions = meta.get("actions", None)
        if actions is None:
            raise KeyError(f"Missing 'actions' in metadata: {pt}")
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = torch.as_tensor(actions).long()

        # Align lengths robustly (handle any tiny off-by-one inconsistencies)
        T = frames.shape[0]
        if len(actions) < end:
            # pad with zeros if actions too short
            pad = end - len(actions)
            pad_zeros = torch.zeros(pad, actions.shape[1], dtype=actions.dtype)
            actions = torch.cat([actions, pad_zeros], dim=0)
        actions = actions[start:end]
        actions = actions[:T]  # clamp just in case

        # 3) Split into input / target segments
        input_seq = frames[: self.seq_len].contiguous()
        target_seq = frames[self.seq_len : self.seq_len + self.target_seq_len].contiguous()

        input_actions = actions[: self.seq_len].clone()
        target_actions = actions[self.seq_len : self.seq_len + self.target_seq_len].clone()

        # Force first action in each segment to (0,0) to match generator dataset
        if input_actions.numel() > 0:
            input_actions[0] = 0
        if target_actions.numel() > 0:
            target_actions[0] = 0

        return {
            "input_seq": input_seq,             # (L, 1, H, W) float32 in [0,1]
            "input_actions": input_actions,     # (L, 2) long
            "target_seq": target_seq,           # (Lt, 1, H, W)
            "target_actions": target_actions,   # (Lt, 2) long
        }



class MNISTWorldDynamicDataset(Dataset):
    """
    Same as MNISTWorldDynamicDataset_FromFiles, but generates the dataset on the fly
    
    Dynamic MNIST-world:
      - A larger toroidal world with MNIST digits as moving sprites.
      - Camera does a random walk (with optional coverage requirement).
      - Digits each move with an (integer) velocity and wrap at edges.
      - Each frame is rendered by pasting digits at their current positions (sum + clamp).

    Returned item:
        "input_seq":         Tensor [seq_len, 1, window_size, window_size]
        "input_actions":     LongTensor [seq_len, 2]   (dx, dy; first is (0,0))
        "target_seq":        Tensor [target_seq_len, 1, window_size, window_size]
        "target_actions":    LongTensor [target_seq_len, 2]
    If return_metadata:
        "digit_labels":                  LongTensor [num_digits]
        "digit_init_positions":          LongTensor [num_digits, 2]        (x0, y0)
        "digit_velocities":              LongTensor [num_digits, 2]        (vx, vy)
        "digit_positions_over_time":     LongTensor [num_digits, total_T, 2]
        "camera_centers_input":          LongTensor [seq_len, 2]           (cx, cy)
        "camera_centers_target":         LongTensor [target_seq_len, 2]
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        *,
        world_size: int = 96,
        num_digits: int = 5,
        window_size: int = 28,
        seq_len: int = 50,
        target_seq_len: int = 30,
        step_size: int = 10,
        ensure_full_coverage: bool = True,
        random_start_loc: bool = False,
        transform=None,
        download: bool = True,
        random: bool = True,
        seed: int = 42,
        max_attempts: int = 100,
        return_metadata: bool = False,
        # placement/labels (same behavior as static)
        max_overlap_frac: Optional[float] = None,
        unique_digits: bool = False,
        # NEW: dynamics
        digit_vel_range: Tuple[int, int] = (-1, 1),
        moving_fraction: float = 1.0,            # fraction of digits that move
        require_nonzero_velocity: bool = False,  # if True, (vx,vy)!=(0,0) for moving digits
        # Placeholder for future bouncing dynamics
        is_bounce: bool = False,
        # optionally return full-world frames (H=W=world_size) for all timesteps
        return_full_world: bool = False,
        # biased rollout options for target sequence
        straightline_biased_rollout: bool = False,
        forward_probability: float = 0.93,
        constant_velocity: bool = False,
    ) -> None:
        super().__init__()
        assert world_size >= window_size, "world_size must be >= window_size"
        if unique_digits and num_digits > 10:
            raise ValueError("num_digits must be <= 10 when unique_digits=True")
        self.world_size = world_size
        self.num_digits = num_digits
        self.window_size = window_size
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len
        self.step_size = step_size
        self.ensure_full_coverage = ensure_full_coverage
        self.random_start_loc = random_start_loc
        self.transform = transform
        self.random = random
        self.max_attempts = max_attempts
        self.return_metadata = return_metadata
        self.return_full_world = return_full_world
        self.straightline_biased_rollout = straightline_biased_rollout
        self.forward_probability = float(np.clip(forward_probability, 0.0, 1.0))
        self.constant_velocity = constant_velocity
        if max_overlap_frac is not None and not (0.0 <= max_overlap_frac <= 1.0):
            raise ValueError("max_overlap_frac must be in [0,1]")
        self.max_overlap_frac = max_overlap_frac
        self.unique_digits = unique_digits

        # dynamics params
        vmin, vmax = digit_vel_range
        if vmin > vmax:
            raise ValueError("digit_vel_range must be (min, max) with min <= max")
        self.digit_vel_range = digit_vel_range
        self.moving_fraction = float(np.clip(moving_fraction, 0.0, 1.0))
        self.require_nonzero_velocity = require_nonzero_velocity
        self.is_bounce = is_bounce

        if self.is_bounce:
            raise NotImplementedError(
                "Bouncing dynamics is not implemented yet. Set is_bounce=False."
            )

        # Underlying MNIST dataset
        self.mnist = MNIST(root=root, train=train, download=download, transform=ToTensor())

        # label->indices (fast sampling)
        self.label_to_indices: List[List[int]] = [[] for _ in range(10)]
        targets = getattr(self.mnist, "targets", None)
        if targets is not None:
            targets = targets.tolist()
            for idx, lbl in enumerate(targets):
                self.label_to_indices[int(lbl)].append(idx)
        else:
            for idx in range(len(self.mnist)):
                _, lbl = self.mnist[idx]
                self.label_to_indices[int(lbl)].append(idx)
        for lbl in range(10):
            if len(self.label_to_indices[lbl]) == 0:
                raise RuntimeError(f"MNIST label {lbl} has no examples in this split")

        # Random-number generator
        self.rng = np.random.RandomState(seed)

        # Expose many worlds
        self._length = 60000 if train else 10000

    # ------------------------------- Dataset API -------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        # Reset target direction state for each new sample
        if hasattr(self, '_current_target_direction'):
            delattr(self, '_current_target_direction')
            
        # 1) Build initial static world state + digit patches, labels, init positions
        patches, labels, init_positions = self._sample_and_place_initial_digits()

        # 2) Choose moving subset & velocities, then precompute trajectories
        total_T = self.seq_len + self.target_seq_len
        moving_mask, velocities = self._assign_velocities()
        trajectories = self._compute_digit_trajectories(
            init_positions, velocities, moving_mask, total_T
        )  # LongTensor [num_digits, total_T, 2]

        # 3) Generate camera random walk (possibly retrying for coverage)
        for attempt in range(self.max_attempts):
            start_cx, start_cy = self._initial_center()
            (input_frames, input_actions, last_cx, last_cy,
             visited, input_centers) = self._render_sequence_with_dynamics(
                patches, trajectories, 0, self.seq_len, start_cx, start_cy,
                require_full=self.ensure_full_coverage
             )
            if self.ensure_full_coverage and not visited.all():
                continue
            break
        else:
            raise RuntimeError("Failed to generate a coverage-satisfying sequence; increase seq_len.")

        # 4) Target sequence continues camera from last input position
        (target_frames, target_actions, _, _, _,
         target_centers) = self._render_sequence_with_dynamics(
            patches, trajectories, self.seq_len, self.target_seq_len, last_cx, last_cy,
            require_full=False, is_target_sequence=True
        )

        # 5) Optional transform on both sequences
        if self.transform is not None:
            input_frames = self.transform(input_frames)
            target_frames = self.transform(target_frames)

        sample = {
            "input_seq": input_frames,    # [L, 1, W, W]
            "input_actions": torch.tensor(input_actions, dtype=torch.long),
            "target_seq": target_frames,  # [Lt, 1, W, W]
            "target_actions": torch.tensor(target_actions, dtype=torch.long),
        }
        if self.return_full_world:
            # Produce full-world frames for each timestep in [0, total_T)
            total_T = self.seq_len + self.target_seq_len
            world_frames = torch.zeros(total_T, 1, self.world_size, self.world_size)
            for t in range(total_T):
                world_t = self._render_world_from_positions(patches, trajectories[:, t, :])  # [1, H, W]
                world_frames[t] = world_t
            sample["world_seq"] = world_frames
        if self.return_metadata:
            sample.update({
                "digit_labels": labels,                                        # [num_digits]
                "digit_init_positions": init_positions,                        # [num_digits, 2]
                "digit_velocities": velocities,                                # [num_digits, 2]
                "digit_positions_over_time": trajectories,                     # [num_digits, T, 2]
                "camera_centers_input": input_centers,                         # [L, 2]
                "camera_centers_target": target_centers,                       # [Lt, 2]
            })
        return sample

    # ------------------------------- Helpers -----------------------------------

    def _rng(self):
        return np.random if self.random else self.rng

    def _sample_digit_with_label(self, label: int, rand) -> Tuple[torch.Tensor, int]:
        indices = self.label_to_indices[label]
        idx = int(rand.choice(indices))
        digit_tensor, lbl = self.mnist[idx]
        return digit_tensor, int(lbl)

    def _sample_and_place_initial_digits(self):
        """
        Returns:
          - patches: list[Tensor 1x28x28] length num_digits
          - labels:  LongTensor [num_digits]
          - init_positions: LongTensor [num_digits, 2] (x0,y0)
        Placement uses the same two modes as your static dataset:
          * with max_overlap_frac -> bounded overlap without wrap
          * else -> wrapped placement (original behavior)
        """
        labels: List[int] = []
        positions: List[Tuple[int, int]] = []
        patches: List[torch.Tensor] = []
        rand = self._rng()

        # Choose labels (unique or not)
        if self.unique_digits:
            available = list(range(10))
            rand.shuffle(available)
            chosen = available[: self.num_digits]
        else:
            chosen = [int(rand.randint(0, 10)) for _ in range(self.num_digits)]

        if self.max_overlap_frac is not None:
            # No-wrap bounded-overlap placement
            max_frac = self.max_overlap_frac
            max_overlap_pixels = int(round(max_frac * 28 * 28))
            occ = torch.zeros(self.world_size, self.world_size, dtype=torch.bool)

            for lbl in chosen:
                digit, lbl_sampled = self._sample_digit_with_label(lbl, rand)
                mask = (digit[0] > 0)
                placed = False
                max_x = self.world_size - 28
                max_y = self.world_size - 28
                for _attempt in range(3000):
                    if max_x < 0 or max_y < 0:
                        break
                    x0 = int(rand.randint(0, max_x + 1))
                    y0 = int(rand.randint(0, max_y + 1))
                    occ_sub = occ[y0:y0+28, x0:x0+28]
                    overlap_pixels = (occ_sub & mask).sum().item()
                    if overlap_pixels <= max_overlap_pixels:
                        occ[y0:y0+28, x0:x0+28] |= mask
                        labels.append(lbl_sampled)
                        positions.append((x0, y0))
                        patches.append(digit)
                        placed = True
                        break
                if not placed:
                    raise RuntimeError("Failed to place digits; relax constraints or enlarge world.")
        else:
            # Wrapped placement allowing overlap
            for lbl in chosen:
                digit, lbl_sampled = self._sample_digit_with_label(lbl, rand)
                x0 = int(rand.randint(0, self.world_size))
                y0 = int(rand.randint(0, self.world_size))
                labels.append(lbl_sampled)
                positions.append((x0, y0))
                patches.append(digit)

        labels_t = torch.tensor(labels, dtype=torch.long)
        positions_t = torch.tensor(positions, dtype=torch.long)
        return patches, labels_t, positions_t

    def _assign_velocities(self):
        """
        Assign velocities to a moving subset of digits.
        Returns:
          moving_mask: BoolTensor [num_digits]
          velocities:  LongTensor [num_digits, 2]
        """
        rand = self._rng()
        n = self.num_digits
        # Choose moving subset
        k = int(round(self.moving_fraction * n))
        indices = list(range(n))
        rand.shuffle(indices)
        moving_idx = set(indices[:k])

        # Sample integer velocities in range
        vmin, vmax = self.digit_vel_range
        vx = [int(rand.randint(vmin, vmax + 1)) for _ in range(n)]
        vy = [int(rand.randint(vmin, vmax + 1)) for _ in range(n)]
        if self.require_nonzero_velocity:
            for i in range(n):
                if i in moving_idx:
                    # resample until (vx,vy)!=(0,0)
                    tries = 0
                    while vx[i] == 0 and vy[i] == 0 and tries < 50:
                        vx[i] = int(rand.randint(vmin, vmax + 1))
                        vy[i] = int(rand.randint(vmin, vmax + 1))
                        tries += 1

        moving_mask = torch.zeros(n, dtype=torch.bool)
        if k > 0:
            moving_mask[list(moving_idx)] = True

        velocities = torch.stack([torch.tensor(vx), torch.tensor(vy)], dim=-1).long()
        # Zero out velocities for non-moving digits
        velocities[~moving_mask] = 0
        return moving_mask, velocities
    
    def _compute_digit_trajectories(self, init_positions, velocities, moving_mask, total_T: int):
        """
        returns: LongTensor [num_digits, total_T, 2]
        """
        n = init_positions.shape[0]
        if total_T <= 0:
            return torch.zeros(n, 0, 2, dtype=torch.long)

        # [total_T, 1] steps 0..T-1
        steps = torch.arange(total_T, dtype=torch.long).view(-1, 1)  # [T,1]
        # broadcast: [n, T, 2] = [n,1,2] + [T,1]*[n,1,2]
        traj = init_positions[:, None, :] + steps[None, :, :] * velocities[:, None, :]
        traj %= self.world_size  # wrap
        return traj


    # ------------------------ camera + rendering ------------------------

    def _initial_center(self) -> Tuple[int, int]:
        rand = self._rng()
        if self.random_start_loc:
            cx = int(rand.randint(0, self.world_size))
            cy = int(rand.randint(0, self.world_size))
        else:
            cx, cy = 0, 0
        return cx, cy

    def _render_sequence_with_dynamics(
        self,
        patches: List[torch.Tensor],
        trajectories: torch.Tensor,  # [num_digits, total_T, 2]
        t0: int,
        length: int,
        start_cx: int,
        start_cy: int,
        *,
        require_full: bool,
        is_target_sequence: bool = False,
    ):
        """
        Render frames t in [t0, t0+length), updating camera per step and digits from trajectories.
        Returns:
          frames:          Tensor [length, 1, W, W]
          actions:         List[(dx,dy)] length
          last_cx,last_cy: ints
          visited:         BoolTensor [world_size, world_size]
          centers:         LongTensor [length, 2] camera centers per frame
        """
        frames = torch.zeros(length, 1, self.window_size, self.window_size)
        actions: List[Tuple[int, int]] = []
        visited = torch.zeros(self.world_size, self.world_size, dtype=torch.bool)
        centers = torch.zeros(length, 2, dtype=torch.long)

        cx, cy = start_cx, start_cy
        half = self.window_size // 2
        rand = self._rng()

        for i in range(length):
            t = t0 + i
            # action
            if i == 0:
                actions.append((0, 0))
            else:
                if require_full and not visited.all():
                    dx, dy = self._biased_step_toward_unseen(cx, cy, visited, rand)
                elif is_target_sequence and self.straightline_biased_rollout:
                    # Apply biased rollout for target sequence
                    dx, dy = self._biased_step_for_target_sequence(rand)
                else:
                    dx = int(rand.randint(-self.step_size, self.step_size + 1))
                    dy = int(rand.randint(-self.step_size, self.step_size + 1))
                actions.append((dx, dy))
                cx = (cx + dx) % self.world_size
                cy = (cy + dy) % self.world_size

            centers[i, 0] = cx
            centers[i, 1] = cy

            # render world at time t from trajectories
            world_t = self._render_world_from_positions(patches, trajectories[:, t, :])
            frame = self._extract_window(world_t, cx, cy)
            frames[i] = frame

            # mark visited pixels of this frame (camera coverage)
            rows = (torch.arange(self.window_size) + (cy - half)) % self.world_size
            cols = (torch.arange(self.window_size) + (cx - half)) % self.world_size
            visited[rows[:, None], cols] = True

        return frames, actions, cx, cy, visited, centers

    def _render_world_from_positions(self, patches: List[torch.Tensor], positions_t: torch.Tensor):
        """
        positions_t: LongTensor [num_digits, 2] (x,y) at single time step
        returns: Tensor [1, H, W]
        """
        world = torch.zeros(1, self.world_size, self.world_size)
        for patch, (x, y) in zip(patches, positions_t.tolist()):
            self._paste_with_wrap(world, patch, int(x), int(y))
        world.clamp_(0.0, 1.0)
        return world

    # ----------------------- reused/ported helpers ----------------------

    def _paste_with_wrap(self, canvas: torch.Tensor, patch: torch.Tensor, x0: int, y0: int):
        assert patch.shape[1] == 28 and patch.shape[2] == 28
        tmp = torch.zeros_like(canvas)
        tmp[:, 0:28, 0:28] = patch
        rolled = torch.roll(tmp, shifts=(y0, x0), dims=(1, 2))
        canvas += rolled
        return canvas

    def _biased_step_toward_unseen(self, cx: int, cy: int, visited: torch.Tensor, rand) -> Tuple[int, int]:
        if visited.all():
            dx = int(rand.randint(-self.step_size, self.step_size + 1))
            dy = int(rand.randint(-self.step_size, self.step_size + 1))
            return dx, dy

        unseen_mask = ~visited
        unseen_indices = unseen_mask.nonzero(as_tuple=False)
        if unseen_indices.numel() == 0:
            dx = int(rand.randint(-self.step_size, self.step_size + 1))
            dy = int(rand.randint(-self.step_size, self.step_size + 1))
            return dx, dy

        idx = int(rand.randint(0, unseen_indices.shape[0]))
        ty, tx = unseen_indices[idx].tolist()

        dx_raw = self._shortest_toroidal_delta(cx, tx, self.world_size)
        dy_raw = self._shortest_toroidal_delta(cy, ty, self.world_size)

        def clip(delta, max_step):
            if delta > 0: return min(delta, max_step)
            if delta < 0: return max(delta, -max_step)
            return 0

        dx = int(clip(dx_raw, self.step_size))
        dy = int(clip(dy_raw, self.step_size))

        if dx == 0 and dy == 0 and not visited.all():
            if abs(dx_raw) >= abs(dy_raw) and dx_raw != 0:
                dx = 1 if dx_raw > 0 else -1
            elif dy_raw != 0:
                dy = 1 if dy_raw > 0 else -1
        return dx, dy

    def _shortest_toroidal_delta(self, current: int, target: int, size: int) -> int:
        delta = (target - current) % size
        if delta > size / 2:
            delta -= size
        return int(delta)

    def _biased_step_for_target_sequence(self, rand) -> Tuple[int, int]:
        """
        Generate biased step for target sequence to favor forward movement.
        Uses the forward_probability to continue in the same direction.
        """
        if not hasattr(self, '_current_target_direction'):
            # Initialize with random direction
            if self.constant_velocity:
                # Use constant step size (half of step_size) for more controlled movement
                half_step = self.step_size // 2
                dx = int(rand.choice([-half_step, half_step]))
                dy = int(rand.choice([-half_step, half_step]))
            else:
                # Use random step size within range
                dx = int(rand.randint(-self.step_size, self.step_size + 1))
                dy = int(rand.randint(-self.step_size, self.step_size + 1))
            
            if dx != 0 or dy != 0:  # Avoid zero movement
                self._current_target_direction = (dx, dy)
            else:
                # If we got zero, try again
                return self._biased_step_for_target_sequence(rand)
        else:
            # Decide whether to continue in current direction or change
            if rand.random() < self.forward_probability:
                # Continue in current direction
                dx, dy = self._current_target_direction
            else:
                # Change direction - choose a new random direction
                if self.constant_velocity:
                    half_step = self.step_size // 2
                    # Use constant step size for more predictable movement
                    directions = [
                        [0, half_step], [0, -half_step], 
                        [half_step, 0], [-half_step, 0],  # cardinal directions
                        # [half_step, half_step], [half_step, -half_step], 
                        # [-half_step, half_step], [-half_step, -half_step]  # diagonal directions
                    ]
                else:
                    # Use variable step sizes
                    directions = [
                        [0, 1], [0, -1], [1, 0], [-1, 0],  # cardinal directions
                        [1, 1], [1, -1], [-1, 1], [-1, -1]  # diagonal directions
                    ]
                
                # Exclude the current direction and its opposite
                current = self._current_target_direction
                opposite = [-current[0], -current[1]]
                available_dirs = [d for d in directions if d != current and d != opposite]
                
                if available_dirs:
                    new_direction = available_dirs[rand.randint(0, len(available_dirs))]
                    dx, dy = new_direction
                    self._current_target_direction = new_direction
                else:
                    # Fallback: just use random direction
                    if self.constant_velocity:
                        dx = int(rand.choice([-half_step, half_step]))
                        dy = int(rand.choice([-half_step, half_step]))
                    else:
                        dx = int(rand.randint(-self.step_size, self.step_size + 1))
                        dy = int(rand.randint(-self.step_size, self.step_size + 1))
                    self._current_target_direction = (dx, dy)
        
        return dx, dy

    def _extract_window(self, world: torch.Tensor, cx: int, cy: int) -> torch.Tensor:
        half = self.window_size // 2
        rolled = torch.roll(world, shifts=(-cy + half, -cx + half), dims=(1, 2))
        frame = rolled[:, : self.window_size, : self.window_size]
        return frame.clone()


# ------------------------------- Video utils -------------------------------
def _save_video_mp4(frames, path: str, fps: int = 8):
    """
    Save a sequence of frames to an MP4 file using imageio-ffmpeg.
    frames: List/ndarray of shape [T, H, W] or [T, 1, H, W] or [T, 3, H, W]; values in [0,1] or [0,255].
    """
    # Convert to numpy array [T, H, W] or [T, 1, H, W] or [T, 3, H, W]
    if isinstance(frames, torch.Tensor):
        frames_np = frames.detach().cpu().numpy()
    else:
        frames_np = np.asarray(frames)

    # Normalize dtype to uint8 0..255
    if frames_np.dtype != np.uint8:
        # Assume 0..1 floats or other numeric; scale and clip
        frames_np = np.clip(frames_np, 0, 1) * 255.0
        frames_np = frames_np.astype(np.uint8)

    # Ensure shape [T, H, W, 3]
    if frames_np.ndim == 3:  # [T, H, W]
        frames_np = np.repeat(frames_np[:, :, :, None], 3, axis=-1)
    elif frames_np.ndim == 4:
        if frames_np.shape[1] in (1, 3):  # [T, C, H, W]
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))  # -> [T, H, W, C]
        # Now shape [T, H, W, C]
        C = frames_np.shape[-1]
        if C == 1:
            frames_np = np.repeat(frames_np, 3, axis=-1)
        elif C != 3:
            raise ValueError("Frames must have 1 or 3 channels")
    else:
        raise ValueError("Unexpected frames ndim; expected 3 or 4 dims")

    # Write using imageio (ffmpeg backend)
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=8) as writer:
        for f in frames_np:
            writer.append_data(f)


# ----------------------------- Example runner -----------------------------

### Test loading from files
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import os
#     import imageio

#     # Set up a directory to save visualizations
#     parent_dir = "dataset_examples"
#     os.makedirs(parent_dir, exist_ok=True)
#     out_dir = os.path.join(parent_dir, "mnist_world_dynamic_fromfiles_examples")
#     os.makedirs(out_dir, exist_ok=True)

#     # Example: change these paths as needed for your environment
#     root_dir = "./data/mnist_world/"
#     split = "dynamic_training"

#     # Instantiate the dataset
#     dataset = MNISTWorldDynamicDataset_FromFiles(
#         root_dir=root_dir,
#         split=split,
#         seq_len=50,
#         target_seq_len=20,
#         random_window=False,
#     )

#     print(f"Loaded MNISTWorldDynamicDataset_FromFiles with {len(dataset)} clips in split '{split}'.")

#     # Print some statistics
#     lengths = []
#     for i in range(min(100, len(dataset))):
#         mp4, pt, n = dataset.index[i]
#         lengths.append(n)
#     print(f"First 100 video lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")

#     # Visualize a few random examples
#     for k in range(3):
#         idx = np.random.randint(0, len(dataset))
#         item = dataset[idx]
#         input_seq = item["input_seq"]  # (L, 1, H, W)
#         target_seq = item["target_seq"]  # (Lt, 1, H, W)
#         input_actions = item["input_actions"]  # (L, 2)
#         target_actions = item["target_actions"]  # (Lt, 2)

#         # Print some info
#         print(f"\nExample {k+1}: idx={idx}")
#         print(f"  input_seq shape: {input_seq.shape}")
#         print(f"  target_seq shape: {target_seq.shape}")
#         print(f"  input_actions[0:5]:\n{input_actions[:5]}")
#         print(f"  target_actions[0:5]:\n{target_actions[:5]}")

#         # Visualize input and target sequences as a grid, with action printed on each frame
#         seq = torch.cat([input_seq, target_seq], dim=0)  # (T, 1, H, W)
#         actions = torch.cat([input_actions, target_actions], dim=0)  # (T, 2)
#         T = seq.shape[0]
#         ncols = 5
#         nrows = int(np.ceil(T / ncols))
#         fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
#         for t in range(T):
#             r, c = divmod(t, ncols)
#             ax = axes[r, c] if nrows > 1 else axes[c]
#             ax.imshow(seq[t,0].cpu().numpy(), cmap="gray")
#             action_str = f"({actions[t,0].item()},{actions[t,1].item()})"
#             ax.set_title(f"t={t}\naction={action_str}", fontsize=8)
#             ax.axis("off")
#         # Hide unused axes
#         for t in range(T, nrows*ncols):
#             r, c = divmod(t, ncols)
#             ax = axes[r, c] if nrows > 1 else axes[c]
#             ax.axis("off")
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, f"fromfiles_example_{k+1}.png"), dpi=120, bbox_inches="tight")
#         plt.close()

#         # Save as video (mp4)
#         video_path = os.path.join(out_dir, f"fromfiles_example_{k+1}.mp4")
#         frames = seq * 1.0  # (T, 1, H, W), float32 in [0,1]
#         _save_video_mp4(frames, video_path, fps=8)
#         print(f"  Saved grid to {os.path.join(out_dir, f'fromfiles_example_{k+1}.png')}")
#         print(f"  Saved video to {video_path}")

#     print("Done visualizing MNISTWorldDynamicDataset_FromFiles examples.")


### Test on-the-fly dynamic dataset generation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import imageio
    import argparse

    parent_dir = "dataset_examples"
    os.makedirs(parent_dir, exist_ok=True)
    out_dir = os.path.join(parent_dir, "mnist_world_dynamic_examples")
    os.makedirs(out_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--render_full_world", action="store_true", help="Save full-world videos with same frame count as camera view")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    dataset = MNISTWorldDynamicDataset(
        root="./data",
        train=True,
        world_size=80,
        num_digits=7,
        window_size=32,
        seq_len=50,
        target_seq_len=20,
        step_size=10,
        ensure_full_coverage=True,
        return_metadata=True,
        return_full_world=args.render_full_world,
        max_overlap_frac=None,
        unique_digits=True,
        # dynamics
        digit_vel_range=(-2, 2),
        moving_fraction=1.0,
        require_nonzero_velocity=True,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)


    # Optional prior visualization is skipped when rendering full world via dataset
    if not args.render_full_world:
        # Save a few random worlds at t=0 and t=10 just to visualize motion
        for k in range(3):
            patches, labels, init_pos = dataset._sample_and_place_initial_digits()
            moving_mask, velocities = dataset._assign_velocities()
            traj = dataset._compute_digit_trajectories(init_pos, velocities, moving_mask, total_T=11)

            world_t0 = dataset._render_world_from_positions(patches, traj[:, 0, :])
            world_t10 = dataset._render_world_from_positions(patches, traj[:, 10, :])

            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1); plt.imshow(world_t0[0], cmap="gray"); plt.title("t=0"); plt.axis("off")
            plt.subplot(1,2,2); plt.imshow(world_t10[0], cmap="gray"); plt.title("t=10"); plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"world_motion_{k+1}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # Save 4 MP4s of world evolutions only when not using dataset-returned full world
    if not args.render_full_world:
        total_T = dataset.seq_len + dataset.target_seq_len
        for idx in range(1, 5):
            patches, labels, init_pos = dataset._sample_and_place_initial_digits()
            moving_mask, velocities = dataset._assign_velocities()
            traj = dataset._compute_digit_trajectories(init_pos, velocities, moving_mask, total_T=total_T)
            world_frames = []  # [T, H, W]
            for t in range(total_T):
                world_t = dataset._render_world_from_positions(patches, traj[:, t, :])  # [1, H, W]
                world_frames.append(world_t[0].numpy())
            world_mp4_path = os.path.join(out_dir, f"world_evolution_{idx}.mp4")
            _save_video_mp4(np.stack(world_frames, axis=0), world_mp4_path, fps=args.fps)

    # Sample a few sequences and make contact sheets
    for i, sample in enumerate(loader):
        print("input_seq:", sample["input_seq"].shape, "target_seq:", sample["target_seq"].shape)

        # Use the first item in the batch for visualization
        inp = sample["input_seq"][0]      # [L, 1, 28, 28]
        tar = sample["target_seq"][0]     # [Lt, 1, 28, 28]
        ia = sample["input_actions"][0]   # [L, 2]
        ta = sample["target_actions"][0]  # [Lt, 2]

        L = inp.shape[0]
        Lt = tar.shape[0]
        idx_inp = np.linspace(0, L-1, min(10, L), dtype=int)
        idx_tar = np.linspace(0, Lt-1, min(10, Lt), dtype=int)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, len(idx_inp), figsize=(2*len(idx_inp), 4))
        fig.suptitle(f"Dynamic MNIST World Example {i+1}", fontsize=14)

        for j, t in enumerate(idx_inp):
            axes[0, j].imshow(inp[t, 0], cmap="gray")
            axes[0, j].set_title(f"t={t}\n({int(ia[t,0])},{int(ia[t,1])})")
            axes[0, j].axis("off")

        for j, t in enumerate(idx_tar):
            axes[1, j].imshow(tar[t, 0], cmap="gray")
            axes[1, j].set_title(f"t'={t}\n({int(ta[t,0])},{int(ta[t,1])})")
            axes[1, j].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"example_{i+1}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Save metadata preview
        md = {k: sample[k][0] for k in sample if k.startswith("digit_") or k.startswith("camera_")}
        if "digit_positions_over_time" in md:
            np.save(os.path.join(out_dir, f"digit_positions_over_time_{i+1}.npy"),
                    md["digit_positions_over_time"].numpy())

        if args.render_full_world and ("world_seq" in sample):
            # Save MP4 of the full-world view with same frame count as camera view
            world_frames = sample["world_seq"][0]  # [T, 1, H, W]
            world_mp4_path = os.path.join(out_dir, f"world_view_{i+1}.mp4")
            _save_video_mp4(world_frames, world_mp4_path, fps=args.fps)
        else:
            # Save MP4 of the camera (input + target concatenated)
            cam_frames = torch.cat([inp, tar], dim=0)  # [L+Lt, 1, 28, 28]
            cam_mp4_path = os.path.join(out_dir, f"camera_view_{i+1}.mp4")
            _save_video_mp4(cam_frames, cam_mp4_path, fps=args.fps)

        if i >= 3:
            break

    print(f"Saved figures & metadata to: {out_dir}")

