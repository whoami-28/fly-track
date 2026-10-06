"""Configuration settings for Polytrack Drosophila Vision-Motor Loop."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

CALIBRATION_FILE = BASE_DIR / "calibration_config.json"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CONNECTOME_DIR = DATA_DIR / "connectome"
CONNECTOME_CACHE_FILE = CONNECTOME_DIR / "flywire_subgraph.npz"
CONNECTOME_METADATA_FILE = CONNECTOME_DIR / "metadata.json"

# Preferred Device
DEVICE = os.getenv("DEVICE", "cpu")

# Game Window Detection (Supports Desktop standalone app & Web browsers)
WINDOW_KEYWORD = os.getenv("WINDOW_KEYWORD", "Polytrack")
WINDOW_TITLE_KEYWORDS = [WINDOW_KEYWORD, "Polytrack", "Polytrack.exe", "itch.io", "Chrome", "Edge", "Firefox", "Brave"]
DEFAULT_MONITOR_INDEX = 1
DEFAULT_BBOX = {
    "top": 100,
    "left": 100,
    "width": 800,
    "height": 600,
}

# Target Performance & Latency Constraints
TARGET_FPS = int(os.getenv("TARGET_FPS", "60"))
MAX_ALLOWED_LATENCY_MS = float(os.getenv("MAX_ALLOWED_LATENCY_MS", "40.0"))

# Keyboard Controls
KEY_FORWARD = "w"
KEY_BRAKE = "s"
KEY_LEFT = "a"
KEY_RIGHT = "d"
KEY_RESET = "r"

# Fall & Game State Detection Parameters
# When the car falls off the track into the void, edge detail and contrast collapse dramatically.
FALL_CONTRAST_VAR_THRESHOLD = float(os.getenv("FALL_CONTRAST_THRESHOLD", "80.0"))
FALL_CONSECUTIVE_FRAMES = 5         # Number of consecutive frames to trigger fall event

# HUD / Timer Region (relative fractions of game window: [y_min, y_max, x_min, x_max])
HUD_TIMER_REL_ROI = (0.02, 0.12, 0.40, 0.60)  # Top center area where Polytrack timer is displayed

# Drosophila Retina & Vision Configuration (Module 2)
# Binocular ommatidia layout: 2 eyes x (16 x 16) = 512 total ommatidia, or single 16x32 panorama
RETINA_GRID_HEIGHT = 16
RETINA_GRID_WIDTH = 32          # Split: left 16 cols = Left Eye, right 16 cols = Right Eye
EYE_OMMATIDIA_COUNT = 16 * 16   # 256 per eye, 512 total
HRC_LOWPASS_ALPHA = 0.65        # Temporal decay for delayed arm in Hassenstein-Reichardt correlator
OPTIC_FLOW_GAIN = 10.0          # Gain scalar for motion-induced synaptic currents I_ext
ON_OFF_SEPARATION = True        # Separate ON (+delta I) and OFF (-delta I) processing pathways

# Drosophila SNN & Brain Model (Module 4)
RETINA_FEATURE_DIM = 2004       # 512 lum + 512 delta + 496 emd_x + 480 emd_y + 4 lptc
LPTC_NEURON_COUNT = 26          # 6 HS cells + 20 VS cells
SNN_SUBSTEPS_PER_FRAME = int(os.getenv("SNN_SUBSTEPS", "8"))
SNN_ALPHA_DECAY = float(os.getenv("SNN_ALPHA_DECAY", "0.85"))
SNN_V_THRESHOLD = 1.0           # Spike generation threshold
SNN_V_RESET = 0.0               # Membrane reset potential
SNN_SYNAPTIC_GAIN = float(os.getenv("SNN_SYNAPTIC_GAIN", "0.04"))

# Motor Decoding Thresholds (Descending Neurons -> WASD)
MOTOR_STEER_DIFF_THRESHOLD = float(os.getenv("MOTOR_STEER_THRESHOLD", "0.12"))
MOTOR_THROTTLE_RATE_THRESHOLD = float(os.getenv("MOTOR_THROTTLE_THRESHOLD", "0.15"))
MOTOR_BRAKE_RATE_THRESHOLD = float(os.getenv("MOTOR_BRAKE_THRESHOLD", "0.22"))


