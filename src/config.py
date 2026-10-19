"""
config.py - Central configuration for the Predictive Maintenance project.

All paths, hyperparameters, and label mappings are defined here.
"""

from pathlib import Path
import json

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset A: Motor Rotor Bar Fault (banco de dados experimental)
DATASET_A_DIR = PROJECT_ROOT / "DatasetRaw" / "banco de dados experimental"

# Output directories
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "results"

# ============================================================================
# PREPROCESSING HYPERPARAMETERS
# ============================================================================
WINDOW_SIZE = 4096    # samples per window
STRIDE      = 2048    # 50% overlap
# Motor uses a larger stride (0% overlap): its signal is near-stationary so 50%-overlap
# windows are nearly interchangeable (empirical kurtosis≈0, peak|z| max 7.7). Doubling the
# stride halves the motor window count (~26k→~13k) → −½ RAM and −½ epoch time at ~zero
# accuracy risk.
MOTOR_STRIDE = 4096   # samples (Dataset A — motor only)
SEED        = 42      # random seed for reproducibility

# ============================================================================
# DATASET A: Motor Rotor Bar Fault Configuration
# ============================================================================
MOTOR_MAT_FILES = {
    "rs":  "struct_rs_R1.mat",    # Healthy rotor
    "r1b": "struct_r1b_R1.mat",   # 1 broken rotor bar
    "r2b": "struct_r2b_R1.mat",   # 2 broken rotor bars
    "r3b": "struct_r3b_R1.mat",   # 3 broken rotor bars
    "r4b": "struct_r4b_R1.mat",   # 4 broken rotor bars
}

# Vibration sensor channels to use
MOTOR_VIB_CHANNELS = [
    "Vib_acpe",    # Accelerometer - ponta de eixo (shaft end)
    "Vib_acpi",    # Accelerometer - ponta inversa (non-drive end)
    "Vib_axial",   # Axial vibration
    "Vib_base",    # Base vibration
    "Vib_carc",    # Carcass vibration
]

# Current channels (optional, for multi-modal fusion)
MOTOR_CURRENT_CHANNELS = ["Ia", "Ib", "Ic"]

# Voltage channels (optional)
MOTOR_VOLTAGE_CHANNELS = ["Va", "Vb", "Vc"]

# Torque levels available
MOTOR_TORQUE_LEVELS = [
    "torque05", "torque10", "torque15", "torque20",
    "torque25", "torque30", "torque35", "torque40",
]

# Label mapping for classification
MOTOR_LABEL_MAP = {
    "rs":  0,   # Healthy
    "r1b": 1,   # 1 broken bar
    "r2b": 2,   # 2 broken bars
    "r3b": 3,   # 3 broken bars
    "r4b": 4,   # 4 broken bars
}

MOTOR_LABEL_NAMES = {v: k for k, v in MOTOR_LABEL_MAP.items()}

# Binary classification mapping (healthy vs faulty)
MOTOR_BINARY_MAP = {
    "rs":  0,   # Healthy
    "r1b": 1,   # Faulty
    "r2b": 1,
    "r3b": 1,
    "r4b": 1,
}

# ============================================================================
# SIGNAL PROCESSING PARAMETERS
# ============================================================================

# Sampling frequencies
# Dataset A vib fs CORRECTED 2026-06-24: was 15990 (assumed 9.6 s duration → WRONG).
#   Source: IEEE DataPort dataset spec states vibration fs = 7.6 kHz (current = 50 kHz).
#   Empirical confirmation: dominant spectral peak lands at 28.6 Hz = rated 1715 rpm
#   rotational frequency ONLY when fs ≈ 7600 (153,504 samples / fs ≈ 20 s recording).
#   Impact: the slip-band bandpass cutoffs are normalised by fs, so the old value made
#   the 40–100 Hz band actually filter ~19–48 Hz (missing the 60 Hz supply sidebands).
MOTOR_VIB_FS = 7600         # Hz — vibration channels (IEEE DataPort spec; empirically verified)

# Slip-band envelope channel parameters (Dataset A — Motor, broken bar detection)
# Mechanism: broken bar produces twice-slip-frequency sidebands (1±2ks)·f around the
#   supply frequency (60 Hz, Brazilian grid; 4-pole, ~1715 rpm). Band 40–100 Hz captures
#   the supply fundamental + sidebands. NOTE: only correctly targeted after the fs fix
#   above — empirical broken/healthy energy ratio peaks at 60–70 Hz, supporting this band.
SLIP_BAND_LOW_HZ = 40.0
SLIP_BAND_HIGH_HZ = 100.0
SLIP_BAND_FILTER_ORDER = 4

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAIN_CONFIG = {
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "early_stopping_patience": 10,
    "label_smoothing": 0.1,
    "n_folds": 5,
    # Data augmentation settings (applied on-the-fly during training)
    "augmentation": {
        "window_warp": True,
        "window_warp_magnitude": 0.1,   # Warps time axis by ±10%
        "mag_scale": True,
        "mag_scale_magnitude": 0.2,     # Scales amplitude by ±20%
        "time_noise": True,
        "time_noise_magnitude": 0.01,   # Adds Gaussian noise σ=0.01
    },
}

def get_config_summary() -> str:
    """Print a summary of the current configuration."""
    summary = []
    summary.append("=" * 60)
    summary.append("PROJECT CONFIGURATION SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Window Size:    {WINDOW_SIZE} samples")
    summary.append(f"Stride:         {STRIDE} samples ({STRIDE/WINDOW_SIZE*100:.0f}% overlap)")
    summary.append(f"Random Seed:    {SEED}")
    summary.append(f"")
    summary.append(f"Dataset A (Motor):")
    summary.append(f"  Conditions:   {list(MOTOR_LABEL_MAP.keys())}")
    summary.append(f"  Vib Channels: {len(MOTOR_VIB_CHANNELS)}")
    summary.append(f"  Torque Lvls:  {len(MOTOR_TORQUE_LEVELS)}")
    summary.append(f"")
    summary.append("=" * 60)
    return "\n".join(summary)


if __name__ == "__main__":
    print(get_config_summary())
