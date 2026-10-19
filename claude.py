"""
Drone detection backend - Flask + OpenCV + Ultralytics YOLO11
Real-time auto-zoom, multi-scale detection and synchronised 2D / 3D-style views.

Live pipeline (one capture thread, ONE inference thread, bounded everywhere):

    Camera / RTSP / file
      -> capture thread            reads continuously, never waits for YOLO
      -> LatestFrameBuffer         one slot: a newer frame replaces an unprocessed one
      -> processing thread
           refine_frame()          adaptive CLAHE / sharpen (brightness for display only by default)
           detect_multiscale()     YOLO on the full frame
                                   -> generate_zoom_crops() picks a few promising regions
                                   -> YOLO on those crops (one batched call)
                                   -> map crop boxes back, class-aware IoU/IoS merge
           DetectionTracker        velocity-predicted matching (stable IDs for fast drones)
           drawing + HUD
      -> FramePacket               annotated frame, zoom view, detections, crops, timings
                                   published with ONE atomic reference swap
      -> /live_frame (dashboard)   one JSON snapshot = feed + zoom + map data of the SAME frame
         /camera_stream            MJPEG of the annotated feed (kept for old pages)
         /zoom_stream              MJPEG of the zoom view

Image (/detect) and video (/upload_video) use the same refine + multi-scale code.
"""

import os

# ------------------------------------------------------------------
# RTSP low-latency options. Must be set before OpenCV opens an FFmpeg
# capture. "nobuffer" + "low_delay" stop FFmpeg from building its own
# input buffer. TCP is the safer default on Wi-Fi; RTSP_TRANSPORT=udp is
# slightly lower latency on a clean wired network.
# ------------------------------------------------------------------
RTSP_TRANSPORT = os.environ.get('RTSP_TRANSPORT', 'tcp')
os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    f'rtsp_transport;{RTSP_TRANSPORT}|fflags;nobuffer|flags;low_delay'
)

from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from datetime import datetime
import json
import threading
import time
import math
import uuid
import atexit
import contextlib
from collections import deque

try:
    import torch
except ImportError:  # ultralytics always installs torch, but stay defensive
    torch = None

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.environ.get('YOLO_MODEL_PATH',
                            r"D:\Multicopter\runs\combined_yolo11l_1024\weights\best.pt")
STATIC_FOLDER = 'static'
CAPTURES_FOLDER = 'static/captures'
UPLOADS_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
SNAPSHOTS_FOLDER = 'static/snapshots'
CONFIDENCE_THRESHOLD = 0.25

# ---------------- Auto-zoom / multi-scale starting values ----------------
# These seed the runtime config (editable live via /set_zoom_options and the
# dashboard). They are starting points, not tuned optima - benchmark them.
ENABLE_AUTO_ZOOM = True
ZOOM_FACTORS = [2.0, 3.0, 4.0]
MAX_ZOOM_CROPS_PER_FRAME = 3
ENABLE_COARSE_SEARCH = True
COARSE_SEARCH_INTERVAL = 5          # frames between coarse tile scans when nothing is being tracked
ENABLE_FRAME_REFINEMENT = True
JPEG_QUALITY = 80

# ---------------- Performance configuration ----------------
# FP16 on CUDA is usually 1.5-2x faster with negligible accuracy change.
# Set YOLO_HALF=0 to force FP32.
USE_HALF_REQUESTED = os.environ.get('YOLO_HALF', '1') == '1'
# None = keep the model's own imgsz (your weights were trained at 1024 and
# Ultralytics keeps that from the checkpoint). Never lowered by default.
INFERENCE_IMGSZ = int(os.environ['YOLO_IMGSZ']) if os.environ.get('YOLO_IMGSZ') else None
# Video uploads get their own model instance so a long video job does not
# block the live camera on a shared lock. Costs one extra copy in VRAM.
SEPARATE_VIDEO_MODEL = os.environ.get('SEPARATE_VIDEO_MODEL', '1') == '1'
# Capture / reconnect
MAX_CONSECUTIVE_READ_FAILURES = 15
RECONNECT_MAX_DELAY_S = 5.0
CAPTURE_OPEN_TIMEOUT_MS = 5000
CAPTURE_READ_TIMEOUT_MS = 5000
CAMERA_BACKEND = os.environ.get('CAMERA_BACKEND', 'auto').lower()  # auto | dshow | msmf | v4l2

# Create folders
for _folder in (STATIC_FOLDER, CAPTURES_FOLDER, UPLOADS_FOLDER, PROCESSED_FOLDER,
                SNAPSHOTS_FOLDER, 'templates'):
    os.makedirs(_folder, exist_ok=True)

cv2.setUseOptimized(True)


# ============================================================
# MODEL LOADING + GPU CHECK
# ============================================================
def _detect_device():
    info = {'torch_version': None, 'cuda_available': False, 'device': 'cpu', 'gpu_name': None,
            'gpu_memory_gb': None}
    if torch is None:
        return info
    info['torch_version'] = torch.__version__
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['device'] = 'cuda:0'
        info['gpu_name'] = torch.cuda.get_device_name(0)
        try:
            info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
    return info


DEVICE_INFO = _detect_device()
INFER_DEVICE = 0 if DEVICE_INFO['cuda_available'] else 'cpu'
USE_HALF = USE_HALF_REQUESTED and DEVICE_INFO['cuda_available']

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print(f"Model loaded successfully! Classes: {model.names}")
print(f"[GPU] torch={DEVICE_INFO['torch_version']} cuda={DEVICE_INFO['cuda_available']} "
      f"gpu={DEVICE_INFO['gpu_name']} half={USE_HALF}")
if not DEVICE_INFO['cuda_available']:
    print("[GPU] WARNING: CUDA is NOT available - YOLO runs on CPU and will be slow.\n"
          "      If you have an NVIDIA GPU, install the CUDA build of PyTorch, e.g.:\n"
          "      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

# Ultralytics predictors are not safe to call from several threads at once.
# Every call to `model` goes through model_lock (the live pipeline has exactly
# one inference thread; /detect briefly shares it).
model_lock = threading.Lock()

_job_model = None
_job_model_lock = threading.Lock()
_job_model_init_lock = threading.Lock()


def _precision_kwargs():
    """Newer Ultralytics (8.4+) replaced `half=True` with `quantize=16`."""
    try:
        from ultralytics.cfg import get_cfg
        if 'quantize' in vars(get_cfg()):
            return {'quantize': 16}
    except Exception:
        pass
    return {'half': True}


PRECISION_KWARGS = _precision_kwargs()


def model_imgsz(m=None):
    m = m or model
    if INFERENCE_IMGSZ:
        return INFERENCE_IMGSZ
    try:
        size = m.overrides.get('imgsz') or 640
        return int(size[0] if isinstance(size, (list, tuple)) else size)
    except Exception:
        return 640


def _inference_ctx():
    if torch is not None:
        return torch.inference_mode()
    return contextlib.nullcontext()


def run_inference(image, conf, model_inst=None, lock=None):
    """Thread-safe YOLO call. `image` may be one frame or a LIST of frames
    (crops), which Ultralytics runs as a single batch. iou=0.45 as before."""
    m = model_inst if model_inst is not None else model
    lk = lock if lock is not None else model_lock
    kwargs = dict(conf=conf, iou=0.45, verbose=False, device=INFER_DEVICE)
    if USE_HALF:
        kwargs.update(PRECISION_KWARGS)
    if INFERENCE_IMGSZ:
        kwargs['imgsz'] = INFERENCE_IMGSZ
    with lk, _inference_ctx():
        return m.predict(image, **kwargs)


def warmup_model(m, lock):
    """Dummy inferences (single + batch of 3) so the first real frame doesn't pay setup cost."""
    try:
        size = model_imgsz(m)
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        run_inference(dummy, 0.5, m, lock)
        t1 = time.perf_counter()
        run_inference(dummy, 0.5, m, lock)
        t2 = time.perf_counter()
        crop = np.zeros((size // 2, size // 2, 3), dtype=np.uint8)
        run_inference([crop, crop, crop], 0.5, m, lock)
        t3 = time.perf_counter()
        print(f"[YOLO] warm-up: first={1000 * (t1 - t0):.1f} ms, single={1000 * (t2 - t1):.1f} ms, "
              f"batch3={1000 * (t3 - t2):.1f} ms, imgsz={size}")
    except Exception as e:
        print(f"[YOLO] warm-up failed: {e}")


def get_job_model():
    """Model instance + lock used by background video jobs."""
    global _job_model
    if not SEPARATE_VIDEO_MODEL:
        return model, model_lock
    with _job_model_init_lock:
        if _job_model is None:
            print("[YOLO] loading separate model instance for video jobs...")
            _job_model = YOLO(MODEL_PATH)
            warmup_model(_job_model, _job_model_lock)
    return _job_model, _job_model_lock


warmup_model(model, model_lock)

# Global variables
camera = None                    # kept for compatibility: points to the CameraPipeline
camera_lock = threading.Lock()   # serialises start/stop of the camera session
camera_active = False
current_frame = None             # latest ANNOTATED frame (used by /capture_frame)
detection_enabled = True
show_center = True
show_crosshair = True
confidence_threshold = 0.50
hide_edge_detections = False
range_focal_px = 0.0
drone_real_width_m = 0.0

# Display options (dashboard)
stream_base = 'raw'              # 'raw' = original camera image under the boxes | 'enhanced'
show_zoom_regions = True         # corner brackets where zoom crops ran
show_trails = True
overlay_scene_map = False        # burn draw_scene_map_overlay() into the live feed (the dashboard has its own panels)
jpeg_quality = JPEG_QUALITY
camera_hfov_deg = 60.0           # only used by the 3D view when focal length is not calibrated

detection_stats = {
    'total_detections': 0,
    'fps': 0,
    'last_detection_time': None,
    'detections': []
}
detection_stats_lock = threading.Lock()

camera_pipeline = None
stream_clients = 0
stream_clients_lock = threading.Lock()

# Video job tracking
video_jobs = {}
video_jobs_lock = threading.Lock()


# ============================================================
# FRAME REFINEMENT
# ============================================================
REFINE_DEFAULTS = {
    'enabled': ENABLE_FRAME_REFINEMENT,
    'mode': 'adaptive',            # 'adaptive' = only fix what the frame needs; 'always' = apply every enabled step
    'analysis_width': 160,         # frame statistics are measured on a tiny copy (cheap)
    # Global brightness / contrast (one LUT pass: levels stretch + gamma)
    'brightness_enabled': True,
    # 'display'  = brighten only the image the operator sees; YOLO gets the un-brightened frame.
    #              (default: in our A/B runs global brightening REDUCED detections - YOLO's training
    #              augmentation already covers exposure changes, and the gain amplifies sensor noise)
    # 'detector' = feed the brightened frame to YOLO too (enable only if YOUR benchmark shows a gain)
    'brightness_mode': 'display',
    'dark_threshold': 80.0,        # mean luma below this -> brighten
    'bright_threshold': 190.0,     # mean luma above this -> darken
    'target_mean': 118.0,
    'gamma_min': 0.45,
    'gamma_max': 1.8,
    'stretch_min_range': 150.0,    # p1..p99 range below this -> stretch levels
    'stretch_max_gain': 1.8,
    # CLAHE (local contrast) on the luma channel only, colours preserved
    'clahe_enabled': True,
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid': [8, 8],
    'low_contrast_std': 45.0,      # luma std below this -> CLAHE
    'clahe_colorspace': 'ycrcb',   # ycrcb (fast, ~6x cheaper conversion) | lab (slightly better colour fidelity)
    # Unsharp-mask sharpening, only for soft frames
    'sharpen_enabled': True,
    'sharpen_strength': 0.3,       # 0 = off, 0.3 conservative, >0.8 halos likely
    'sharpen_sigma': 1.0,
    'blur_threshold': 100.0,       # Laplacian variance of native-res centre patch below this -> sharpen
    # Noise reduction (off by default: it can erase tiny far-away drones)
    'denoise_enabled': False,
    'denoise_method': 'gaussian',  # gaussian | median | bilateral | nlmeans (nlmeans is NOT real-time)
    'noise_threshold': 6.0,
}
refine_config = json.loads(json.dumps(REFINE_DEFAULTS))
refine_config_lock = threading.Lock()
_thread_local = threading.local()
_refine_error_logged = False


def get_refine_config():
    with refine_config_lock:
        cfg = dict(refine_config)
        cfg['clahe_tile_grid'] = list(refine_config['clahe_tile_grid'])
        return cfg


def update_refine_config(changes):
    """Validate and apply partial updates. Returns (config, error)."""
    numeric = {k for k, v in REFINE_DEFAULTS.items() if isinstance(v, float) or
               (isinstance(v, int) and not isinstance(v, bool))}
    booleans = {k for k, v in REFINE_DEFAULTS.items() if isinstance(v, bool)}
    with refine_config_lock:
        new_cfg = dict(refine_config)
        for key, value in (changes or {}).items():
            if key not in REFINE_DEFAULTS:
                continue
            try:
                if key in booleans:
                    new_cfg[key] = value if isinstance(value, bool) else str(value).lower() in ('1', 'true', 'yes', 'on')
                elif key == 'clahe_tile_grid':
                    if isinstance(value, (int, float, str)):
                        value = [int(value), int(value)]
                    tx, ty = int(value[0]), int(value[1])
                    if not (1 <= tx <= 32 and 1 <= ty <= 32):
                        return None, 'clahe_tile_grid values must be 1..32'
                    new_cfg[key] = [tx, ty]
                elif key == 'mode':
                    if value not in ('adaptive', 'always'):
                        return None, "mode must be 'adaptive' or 'always'"
                    new_cfg[key] = value
                elif key == 'brightness_mode':
                    if value not in ('display', 'detector'):
                        return None, "brightness_mode must be 'display' or 'detector'"
                    new_cfg[key] = value
                elif key == 'clahe_colorspace':
                    if value not in ('ycrcb', 'lab'):
                        return None, "clahe_colorspace must be 'ycrcb' or 'lab'"
                    new_cfg[key] = value
                elif key == 'denoise_method':
                    if value not in ('gaussian', 'median', 'bilateral', 'nlmeans'):
                        return None, 'denoise_method must be gaussian|median|bilateral|nlmeans'
                    new_cfg[key] = value
                elif key in numeric:
                    new_cfg[key] = type(REFINE_DEFAULTS[key])(float(value))
            except (TypeError, ValueError, IndexError):
                return None, f'Invalid value for {key}'
        new_cfg['clahe_clip_limit'] = max(0.1, min(40.0, float(new_cfg['clahe_clip_limit'])))
        new_cfg['sharpen_strength'] = max(0.0, min(3.0, float(new_cfg['sharpen_strength'])))
        new_cfg['analysis_width'] = max(64, min(640, int(new_cfg['analysis_width'])))
        refine_config.clear()
        refine_config.update(new_cfg)
        return dict(refine_config), None


def _get_clahe(clip, tiles):
    """CLAHE objects are cached per thread (OpenCV CLAHE objects are not thread-safe)."""
    key = (round(float(clip), 3), int(tiles[0]), int(tiles[1]))
    cache = getattr(_thread_local, 'clahe', None)
    if cache is None:
        cache = _thread_local.clahe = {}
    clahe = cache.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=key[0], tileGridSize=(key[1], key[2]))
        cache[key] = clahe
    return clahe


def analyze_frame(frame, analysis_width=160, need_noise=False):
    """Cheap statistics measured on a tiny grayscale copy of the frame."""
    h, w = frame.shape[:2]
    aw = min(analysis_width, w)
    ah = max(1, int(round(h * aw / max(w, 1))))
    small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_NEAREST)  # stats only: sampling is enough
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cdf = np.cumsum(hist)
    total = cdf[-1] if cdf[-1] > 0 else 1
    p1 = int(np.searchsorted(cdf, 0.01 * total))
    p99 = int(np.searchsorted(cdf, 0.99 * total))
    # Sharpness is measured at NATIVE resolution on a centre patch: downscaling
    # hides blur, and a 256x256 patch keeps the cost well under 1 ms.
    ps = min(256, h, w)
    y0, x0 = (h - ps) // 2, (w - ps) // 2
    patch = cv2.cvtColor(frame[y0:y0 + ps, x0:x0 + ps], cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(patch, cv2.CV_32F).var())
    metrics = {
        'mean': float(mean[0][0]), 'std': float(std[0][0]),
        'p1': p1, 'p99': p99, 'blur': blur, 'noise': None,
    }
    if need_noise:
        residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
        metrics['noise'] = float(residual.mean()) * 1.25
    return metrics


def _build_levels_lut(lo, gain, gamma):
    x = np.arange(256, dtype=np.float32)
    x = np.clip((x - lo) * gain, 0, 255) / 255.0
    x = np.power(x, gamma) * 255.0
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def refine_frame(frame, config=None, return_info=False):
    """
    Lightweight, adaptive enhancement applied BEFORE YOLO. Returns the image YOLO
    should see (plus an info dict when return_info=True). See refine_frame_full().
    """
    detector_img, _, info = refine_frame_full(frame, config)
    return (detector_img, info) if return_info else detector_img


def display_frame(detector_img, display_lut):
    """Operator view: detector image + display-only brightness LUT (one cheap LUT pass)."""
    if display_lut is None or detector_img is None:
        return detector_img
    return cv2.LUT(detector_img, display_lut)


def refine_frame_full(frame, config=None):
    """
    Returns (detector_image, display_lut_or_None, info).

    Steps (each one only runs when the frame needs it in 'adaptive' mode):
      1. optional fast denoise        (noisy frames; disabled by default)
      2. levels stretch + gamma LUT   (dark / over-bright / flat frames; one LUT pass)
      3. CLAHE on the luma channel    (low local contrast, haze, dark scenes)
      4. unsharp-mask sharpening      (soft / slightly blurred frames)

    Output is always a uint8 BGR image with the same shape as the input.
    If refinement is disabled (or anything fails) the ORIGINAL array is
    returned without a copy.

    With brightness_mode='display' the levels/gamma LUT is NOT applied to the
    detector image; it is returned separately so the caller can brighten only
    what the operator sees (display_frame()).
    """
    global _refine_error_logged
    cfg = config if config is not None else get_refine_config()
    info = {'enabled': bool(cfg.get('enabled')), 'applied': [], 'metrics': None, 'ms': 0.0}
    display_lut = None
    if frame is None or not cfg.get('enabled'):
        return frame, None, info

    t0 = time.perf_counter()
    try:
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        always = cfg.get('mode') == 'always'
        m = analyze_frame(frame, int(cfg['analysis_width']), need_noise=bool(cfg['denoise_enabled']))
        info['metrics'] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()}
        out = frame

        # 1. Denoise first, so CLAHE / sharpening don't amplify noise.
        if cfg['denoise_enabled'] and (always or (m['noise'] or 0) > cfg['noise_threshold']):
            method = cfg['denoise_method']
            if method == 'median':
                out = cv2.medianBlur(out, 3)
            elif method == 'bilateral':
                out = cv2.bilateralFilter(out, 5, 30, 5)
            elif method == 'nlmeans':
                out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 5, 15)
            else:
                out = cv2.GaussianBlur(out, (3, 3), 0)
            info['applied'].append(f'denoise:{method}')

        # 2. Brightness / global contrast as a single 256-entry LUT.
        #    The stretch is decided first; gamma is then computed from the
        #    brightness the frame will have AFTER the stretch, so the two
        #    corrections never compound into a washed-out frame.
        est_mean, est_std = m['mean'], m['std']
        if cfg['brightness_enabled']:
            dyn_range = m['p99'] - m['p1']
            lo, gain = 0.0, 1.0
            if always or dyn_range < cfg['stretch_min_range']:
                gain = min(cfg['stretch_max_gain'], 255.0 / max(dyn_range, 1))
                lo = float(m['p1'])
                if gain <= 1.02:
                    lo, gain = 0.0, 1.0
            est_mean = min(254.0, max(1.0, (m['mean'] - lo) * gain))
            est_std = m['std'] * gain
            gamma = 1.0
            if est_mean < cfg['dark_threshold'] or est_mean > cfg['bright_threshold']:
                gamma = math.log(cfg['target_mean'] / 255.0) / math.log(est_mean / 255.0)
                gamma = max(cfg['gamma_min'], min(cfg['gamma_max'], gamma))
            if abs(gamma - 1.0) > 0.03 or gain > 1.0:
                lut = _build_levels_lut(lo, gain, gamma)
                if cfg.get('brightness_mode') == 'detector':
                    out = cv2.LUT(out, lut)
                    info['applied'].append(f'levels(g={gamma:.2f},x{gain:.2f})')
                else:
                    display_lut = lut
                    est_mean, est_std = m['mean'], m['std']   # detector frame is not brightened
                    info['applied'].append(f'display-levels(g={gamma:.2f},x{gain:.2f})')

        # 3. CLAHE on luma only -> colour hue unchanged, local detail boosted.
        #    Decided on the contrast the frame has after step 2.
        if cfg['clahe_enabled'] and (always or est_std < cfg['low_contrast_std']):
            clahe = _get_clahe(cfg['clahe_clip_limit'], cfg['clahe_tile_grid'])
            if cfg.get('clahe_colorspace') == 'lab':
                to_space, from_space = cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR
            else:
                to_space, from_space = cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR
            converted = cv2.cvtColor(out, to_space)
            luma = cv2.extractChannel(converted, 0)
            cv2.insertChannel(clahe.apply(luma), converted, 0)
            out = cv2.cvtColor(converted, from_space)
            info['applied'].append('clahe')

        # 4. Conservative unsharp mask for soft frames.
        strength = float(cfg['sharpen_strength'])
        # Dark frames look "blurry" to the Laplacian and are noisy -> never sharpen them adaptively.
        if cfg['sharpen_enabled'] and strength > 0 and (
                always or (m['blur'] < cfg['blur_threshold'] and m['mean'] >= cfg['dark_threshold'])):
            blurred = cv2.GaussianBlur(out, (0, 0), float(cfg['sharpen_sigma']))
            out = cv2.addWeighted(out, 1.0 + strength, blurred, -strength, 0)
            info['applied'].append(f'sharpen({strength:.2f})')

        if out.shape != frame.shape or out.dtype != np.uint8:
            raise ValueError(f'refined frame shape/dtype mismatch {out.shape} {out.dtype}')
        if not out.flags['C_CONTIGUOUS']:
            out = np.ascontiguousarray(out)
    except Exception as e:
        if not _refine_error_logged:
            print(f"[refine] error, falling back to original frame: {e}")
            _refine_error_logged = True
        out = frame
        display_lut = None
        info['applied'] = ['error']
    info['ms'] = round((time.perf_counter() - t0) * 1000.0, 2)
    return out, display_lut, info


# ============================================================
# AUTO-ZOOM / MULTI-SCALE DETECTION
# ============================================================
# Why zoom helps (and when it does not):
#   YOLO resizes every input to `imgsz` (1024 for your model). A 1920x1080
#   frame is shrunk ~1.9x, so a 20-px drone becomes ~11 px at the network
#   input. A 960x540 crop of the same frame is NOT shrunk, so the drone keeps
#   all 20 px (2x more input pixels per target). That is the whole benefit:
#   more of the captured pixels reach the network. Digital zoom never adds
#   detail the sensor did not capture - when the camera resolution is already
#   below imgsz, zooming mostly upsamples and the gain is small.
ZOOM_DEFAULTS = {
    'enabled': ENABLE_AUTO_ZOOM,
    'zoom_factors': list(ZOOM_FACTORS),     # crop side = frame side / factor (aspect preserved)
    'max_crops': MAX_ZOOM_CROPS_PER_FRAME,  # hard cap per live frame
    'coarse_search': ENABLE_COARSE_SEARCH,
    'coarse_interval': COARSE_SEARCH_INTERVAL,
    'coarse_zoom': 2.0,                     # tile grid used by the coarse search (2.0 -> 2x2 tiles + overlap)
    'coarse_overlap': 0.2,
    # Candidate selection
    'candidate_conf': 0.10,     # original-pass boxes between this and the live threshold are re-checked
                                # in a zoom crop. They are NEVER reported unless the zoomed pass itself
                                # reaches the normal confidence threshold.
    'small_target_frac': 0.06,  # box long side < 6% of frame long side -> "small / distant"
    'desired_target_px': 48,    # pick the smallest zoom that makes the target ~48 px at the YOLO input
    'min_context_ratio': 3.0,   # crop long side must be >= 3x target long side (keeps context + padding)
    'min_crop_px': 64,
    'max_upscale': 4.0,         # skip crops YOLO would have to upsample more than this (pure interpolation)
    'crop_overlap_max': 0.5,    # skip a new crop whose IoU with an already selected crop exceeds this
    'track_zoom_every': 2,      # small, stable (conf >= stable_conf) tracked targets: re-zoom every N frames
    'stable_conf': 0.6,
    'lost_track_frames': 6,     # re-acquire: zoom at the PREDICTED position of tracks missed <= N frames
    # Motion candidates (cheap frame differencing; ignored when the whole image moves, e.g. PTZ pan)
    'motion_candidates': True,
    'motion_width': 480,
    'motion_threshold': 18,
    'motion_min_area_px': 2,
    'motion_max_area_frac': 0.01,
    'motion_max_total_frac': 0.15,
    # Merging
    'merge_iou': 0.5,
    'merge_ios': 0.75,          # intersection-over-smaller: merges a crop-truncated box into the full box
    'cross_class_merge_iou': 0.85,  # near-identical boxes with DIFFERENT classes (e.g. drone/octocopter
                                    # flicker between scales) -> keep the more confident one. >1 disables.
    # Temporal confirmation of detections found ONLY in zoom crops (live / video).
    # 2 = must be seen in 2 frames before it is counted -> fewer zoom false positives, +1 frame delay.
    'crop_confirm_hits': 2,
    # Load control
    'adaptive_budget': True,
    'latency_budget_ms': 100.0, # if per-frame processing exceeds this, use fewer crops (min 1)
    'image_max_crops': 16,      # still images (/detect): no time pressure -> thorough scan
}

_zoom_config = json.loads(json.dumps(ZOOM_DEFAULTS))
_zoom_config_lock = threading.Lock()


def get_zoom_config():
    with _zoom_config_lock:
        cfg = dict(_zoom_config)
        cfg['zoom_factors'] = list(_zoom_config['zoom_factors'])
        return cfg


def update_zoom_config(changes):
    """Validate and apply partial updates. Returns (config, error)."""
    with _zoom_config_lock:
        new_cfg = dict(_zoom_config)
        for key, value in (changes or {}).items():
            if key not in ZOOM_DEFAULTS:
                continue
            default = ZOOM_DEFAULTS[key]
            try:
                if isinstance(default, bool):
                    new_cfg[key] = value if isinstance(value, bool) else \
                        str(value).lower() in ('1', 'true', 'yes', 'on')
                elif key == 'zoom_factors':
                    if isinstance(value, str):
                        value = [v for v in value.replace(';', ',').split(',') if v.strip()]
                    factors = sorted({round(float(v), 2) for v in value})
                    if not factors or any(f < 1.1 or f > 8.0 for f in factors):
                        return None, 'zoom_factors must be between 1.1 and 8.0'
                    new_cfg[key] = factors
                elif isinstance(default, int):
                    new_cfg[key] = int(float(value))
                else:
                    new_cfg[key] = float(value)
            except (TypeError, ValueError):
                return None, f'Invalid value for {key}'
        new_cfg['max_crops'] = max(0, min(12, new_cfg['max_crops']))
        new_cfg['image_max_crops'] = max(0, min(64, new_cfg['image_max_crops']))
        new_cfg['coarse_interval'] = max(1, new_cfg['coarse_interval'])
        new_cfg['crop_confirm_hits'] = max(1, min(5, new_cfg['crop_confirm_hits']))
        new_cfg['coarse_zoom'] = max(1.1, min(8.0, new_cfg['coarse_zoom']))
        new_cfg['coarse_overlap'] = max(0.0, min(0.5, new_cfg['coarse_overlap']))
        new_cfg['candidate_conf'] = max(0.01, min(0.95, new_cfg['candidate_conf']))
        _zoom_config.clear()
        _zoom_config.update(new_cfg)
        return dict(_zoom_config), None


# ---------------- geometry helpers ----------------
def _iou_ios(a, b):
    """IoU and intersection-over-smaller of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter), inter / min(area_a, area_b)


def _box_inside(inner, outer, margin=0.0):
    return (inner[0] >= outer[0] + margin and inner[1] >= outer[1] + margin
            and inner[2] <= outer[2] - margin and inner[3] <= outer[3] - margin)


def _boxes_from_result(result, offset=(0, 0), frame_size=None):
    """Ultralytics Result -> list of plain box dicts in frame coordinates."""
    out = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    ox, oy = offset
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = float(x1) + ox, float(y1) + oy, float(x2) + ox, float(y2) + oy
        if frame_size is not None:
            w, h = frame_size
            x1, x2 = max(0.0, min(w - 1.0, x1)), max(0.0, min(w - 1.0, x2))
            y1, y2 = max(0.0, min(h - 1.0, y1)), max(0.0, min(h - 1.0, y2))
        out.append({'xyxy': [x1, y1, x2, y2], 'conf': float(c), 'cls': int(k),
                    'source': 'original', 'zoom': 1.0, 'crop_index': None, 'truncated': False})
    return out


def merge_detections(boxes, iou_thr=0.5, ios_thr=0.75, cross_class_iou=0.85):
    """Class-aware greedy merge of original + crop boxes (plus a strict cross-class rule).
    Highest confidence wins; boxes cut by an inner crop edge are ranked lower.
    The surviving box records every scale that found it in 'sources'."""
    def rank(b):
        return b['conf'] * (0.85 if b['truncated'] else 1.0)

    kept = []
    for b in sorted(boxes, key=rank, reverse=True):
        tag = 'original' if b['source'] == 'original' else f"zoom{b['zoom']:.1f}x"
        duplicate = None
        for k in kept:
            iou, ios = _iou_ios(k['xyxy'], b['xyxy'])
            if k['cls'] != b['cls']:
                if iou >= cross_class_iou:
                    duplicate = k
                    break
                continue
            if iou >= iou_thr or ios >= ios_thr:
                duplicate = k
                break
        if duplicate is not None:
            if tag not in duplicate['sources']:
                duplicate['sources'].append(tag)
            continue
        b = dict(b)
        b['sources'] = [tag]
        kept.append(b)
    return kept


# ---------------- motion candidates ----------------
class MotionCandidateDetector:
    """Frame differencing on a small grey copy -> small moving blobs.
    Costs ~1 ms. Returns [] when too much of the image changes (camera pan,
    exposure jump) because then every pixel 'moves'."""

    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None

    def update(self, frame, cfg):
        h, w = frame.shape[:2]
        aw = int(min(cfg['motion_width'], w))
        ah = max(1, int(round(h * aw / w)))
        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        prev, self.prev = self.prev, gray
        if prev is None or prev.shape != gray.shape:
            return []
        diff = cv2.absdiff(gray, prev)
        _, mask = cv2.threshold(diff, int(cfg['motion_threshold']), 255, cv2.THRESH_BINARY)
        moving = cv2.countNonZero(mask)
        if moving == 0 or moving > cfg['motion_max_total_frac'] * mask.size:
            return []
        mask = cv2.dilate(mask, None, iterations=1)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        scale = w / float(aw)
        max_area = cfg['motion_max_area_frac'] * mask.size
        blobs = []
        for i in range(1, count):
            x, y, bw, bh, area = stats[i]
            if area < cfg['motion_min_area_px'] or area > max_area:
                continue
            blobs.append({'xyxy': [x * scale, y * scale, (x + bw) * scale, (y + bh) * scale],
                          'area': int(area)})
        blobs.sort(key=lambda b: b['area'], reverse=True)
        return blobs[:8]


class ZoomScheduler:
    """Per-stream state for auto-zoom: frame counter, coarse-scan cursor,
    per-track zoom cooldown, motion detector and adaptive crop budget."""

    def __init__(self):
        self.frame_index = 0
        self.last_coarse = -10 ** 9
        self.coarse_cursor = 0
        self.track_last_zoom = {}
        self.motion = MotionCandidateDetector()
        self.budget = None
        self.coarse_scans = 0

    def crop_budget(self, cfg):
        cap = int(cfg['max_crops'])
        if self.budget is None or not cfg['adaptive_budget']:
            self.budget = cap
        self.budget = max(0, min(cap, self.budget))
        return self.budget

    def feedback(self, processing_ms, cfg):
        """Simple step controller: one crop less when over budget, one more when well under."""
        if not cfg['adaptive_budget'] or self.budget is None:
            return
        cap = int(cfg['max_crops'])
        if processing_ms > cfg['latency_budget_ms'] and self.budget > 1:
            self.budget -= 1
        elif processing_ms < 0.7 * cfg['latency_budget_ms'] and self.budget < cap:
            self.budget += 1


def _crop_rect(cx, cy, crop_w, crop_h, frame_w, frame_h):
    """Crop of fixed size centred on (cx, cy), shifted (not shrunk) to stay inside the frame."""
    crop_w = min(frame_w, crop_w)
    crop_h = min(frame_h, crop_h)
    x1 = int(round(min(max(0.0, cx - crop_w / 2.0), frame_w - crop_w)))
    y1 = int(round(min(max(0.0, cy - crop_h / 2.0), frame_h - crop_h)))
    return [x1, y1, int(x1 + round(crop_w)), int(y1 + round(crop_h))]


def choose_zoom_for_box(box, frame_w, frame_h, cfg, imgsz):
    """Smallest configured zoom that makes the target ~desired_target_px at the
    YOLO input, limited so the crop still contains the target with context and
    YOLO does not have to upsample more than max_upscale. None = no zoom needed."""
    frame_long = float(max(frame_w, frame_h))
    target = max(2.0, box[2] - box[0], box[3] - box[1])
    chosen = None
    for z in sorted(cfg['zoom_factors']):
        crop_long = frame_long / z
        if (crop_long < cfg['min_crop_px'] or crop_long < target * cfg['min_context_ratio']
                or imgsz / crop_long > cfg['max_upscale']):
            break
        chosen = z
        if target * imgsz / crop_long >= cfg['desired_target_px']:
            break
    return chosen


def _coarse_tiles(frame_w, frame_h, cfg, imgsz):
    z = cfg['coarse_zoom']
    tw, th = frame_w / z, frame_h / z
    if imgsz / max(tw, th) > cfg['max_upscale'] or max(tw, th) < cfg['min_crop_px']:
        return []
    step_frac = 1.0 - cfg['coarse_overlap']
    nx = max(1, int(math.ceil((frame_w - tw) / (tw * step_frac) - 1e-6)) + 1)
    ny = max(1, int(math.ceil((frame_h - th) / (th * step_frac) - 1e-6)) + 1)
    xs = np.linspace(0, frame_w - tw, nx) if nx > 1 else [0.0]
    ys = np.linspace(0, frame_h - th, ny) if ny > 1 else [0.0]
    return [[int(round(x)), int(round(y)), int(round(x + tw)), int(round(y + th))] for y in ys for x in xs]


def generate_zoom_crops(frame, detections, config, state=None, candidates=None, predicted=None,
                        motion_boxes=None, budget=None, exhaustive=False, imgsz=None):
    """
    Decide WHERE to zoom for this frame. Returns at most `budget` crops:
        {'rect': [x1,y1,x2,y2], 'zoom': float, 'reason': str, 'priority': float,
         'target': {...} | None}

    Candidate priority (highest first):
      100 'reacquire'      tracked target missed this frame -> crop at its predicted position
                           (85 for tentative tracks, which need this crop to get confirmed)
       80 'low-confidence' original-pass box below the threshold -> verify at higher scale
       60 'small-target'   reported but small / distant -> refine box + confidence
                           (stable high-confidence targets only every `track_zoom_every` frames)
       40 'motion'         small moving blob without a detection
       10 'coarse-search'  nothing else to look at -> a few tiles of a grid, round-robin,
                           every `coarse_interval` frames (all tiles when exhaustive=True)
    Crops overlapping an already selected crop by more than crop_overlap_max are skipped.
    """
    cfg = config
    h, w = frame.shape[:2]
    imgsz = imgsz or model_imgsz()
    if budget is None:
        budget = cfg['image_max_crops'] if exhaustive else cfg['max_crops']
    if not cfg['enabled'] or budget <= 0:
        return []
    frame_index = state.frame_index if state is not None else 0

    wanted = []

    def add(box, priority, reason, target):
        wanted.append((priority, box, reason, target))

    for p in predicted or []:
        base = 100 if p.get('confirmed', True) else 85
        add(p['xyxy'], base - 5 * p.get('missed', 1), 'reacquire',
            {'track_id': p.get('track_id'), 'class': p.get('class'), 'confidence': p.get('confidence', 0.0)})
    for c in candidates or []:
        add(c['xyxy'], 80 + 10 * c['conf'], 'low-confidence',
            {'class_id': c['cls'], 'confidence': c['conf']})
    frame_long = float(max(w, h))
    for d in detections or []:
        size = max(d['xyxy'][2] - d['xyxy'][0], d['xyxy'][3] - d['xyxy'][1])
        if size >= cfg['small_target_frac'] * frame_long:
            continue
        tid = d.get('track_id')
        if (state is not None and tid is not None and d['conf'] >= cfg['stable_conf']
                and frame_index - state.track_last_zoom.get(tid, -10 ** 9) < cfg['track_zoom_every']):
            continue
        add(d['xyxy'], 60 + 10 * (1 - d['conf']), 'small-target',
            {'track_id': tid, 'class_id': d['cls'], 'confidence': d['conf']})
    for m in motion_boxes or []:
        covered = any(_iou_ios(m['xyxy'], d['xyxy'])[1] > 0.3 for d in (detections or []))
        if not covered:
            add(m['xyxy'], 40, 'motion', None)

    wanted.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for priority, box, reason, target in wanted:
        if len(selected) >= budget:
            break
        zoom = choose_zoom_for_box(box, w, h, cfg, imgsz)
        if zoom is None:
            continue
        # Target already well inside a chosen crop -> that crop covers it.
        if any(_box_inside(box, s['rect'], margin=4) for s in selected):
            continue
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        rect = _crop_rect(cx, cy, w / zoom, h / zoom, w, h)
        if any(_iou_ios(rect, s['rect'])[0] > cfg['crop_overlap_max'] for s in selected):
            continue
        selected.append({'rect': rect, 'zoom': round(frame_long / max(rect[2] - rect[0], rect[3] - rect[1]), 2),
                         'reason': reason, 'priority': round(priority, 1),
                         'target': dict(target, xyxy=[round(v, 1) for v in box]) if target else
                         {'xyxy': [round(v, 1) for v in box]}})

    run_coarse = cfg['coarse_search'] and len(selected) < budget and (
        exhaustive or (not selected and state is not None
                       and frame_index - state.last_coarse >= cfg['coarse_interval']))
    # Starvation guard: while targets keep the budget busy, still scan one search tile
    # every 3 coarse intervals so new distant drones can be discovered.
    if (cfg['coarse_search'] and not exhaustive and not run_coarse and state is not None and budget >= 2
            and selected and frame_index - state.last_coarse >= 3 * cfg['coarse_interval']):
        if len(selected) >= budget:
            selected.pop()                     # lowest priority goes
        run_coarse = True
    if run_coarse:
        tiles = _coarse_tiles(w, h, cfg, imgsz)
        if tiles:
            free = budget - len(selected)
            if exhaustive:
                order = list(range(len(tiles)))
            else:
                start = state.coarse_cursor % len(tiles)
                order = [(start + i) % len(tiles) for i in range(min(free, len(tiles)))]
                state.coarse_cursor = (start + len(order)) % len(tiles)
                state.last_coarse = frame_index
                state.coarse_scans += 1
            for index in order[:free]:
                rect = tiles[index]
                if any(_iou_ios(rect, s['rect'])[0] > cfg['crop_overlap_max'] for s in selected):
                    continue
                selected.append({'rect': rect,
                                 'zoom': round(frame_long / max(rect[2] - rect[0], rect[3] - rect[1]), 2),
                                 'reason': 'coarse-search', 'priority': 10.0, 'target': None,
                                 'tile': f'{index + 1}/{len(tiles)}'})

    if state is not None:
        for s in selected:
            tid = (s.get('target') or {}).get('track_id')
            if tid is not None:
                state.track_last_zoom[tid] = frame_index
    return selected


def detect_multiscale(frame, conf=None, config=None, scheduler=None, tracker=None, timestamp=None,
                      model_inst=None, lock=None, exhaustive=False, motion_frame=None):
    """
    Multi-scale detection on ONE frame:
      1. YOLO on the full frame (same imgsz/conf/iou as before). It runs at
         min(conf, candidate_conf) only to *find candidates*; boxes below `conf`
         are never reported from this pass.
      2. generate_zoom_crops() picks regions; YOLO runs on all crops in ONE batch.
         Ultralytics letterboxes each crop to imgsz and returns boxes in crop
         pixels, so mapping back is a pure offset: x_frame = crop_x + x_crop.
      3. Class-aware IoU / IoS merge.
    Returns dict(boxes, crops, stats, timings).
    """
    cfg = config if config is not None else get_zoom_config()
    conf = confidence_threshold if conf is None else conf
    m = model_inst if model_inst is not None else model
    lk = lock if lock is not None else model_lock
    h, w = frame.shape[:2]
    imgsz = model_imgsz(m)
    zoom_on = bool(cfg['enabled'])

    t0 = time.perf_counter()
    pass_conf = min(conf, cfg['candidate_conf']) if zoom_on else conf
    first = _boxes_from_result(run_inference(frame, pass_conf, m, lk)[0])
    original = [b for b in first if b['conf'] >= conf]
    low_conf = [b for b in first if b['conf'] < conf]
    t1 = time.perf_counter()

    crops, crop_boxes, motion = [], [], []
    if zoom_on:
        if scheduler is not None and cfg['motion_candidates']:
            motion = scheduler.motion.update(motion_frame if motion_frame is not None else frame, cfg)
        predicted = tracker.predict_lost(timestamp, cfg['lost_track_frames'], boxes=original) \
            if tracker is not None else []
        # attach track ids to original boxes (IoU with the tracker's predicted boxes) for the cooldown
        if tracker is not None:
            tracker.annotate_track_ids(original, timestamp)
        if exhaustive:
            budget = cfg['image_max_crops']
        elif scheduler is not None:
            budget = scheduler.crop_budget(cfg)
        else:
            budget = cfg['max_crops']
        crops = generate_zoom_crops(frame, original, cfg, state=scheduler, candidates=low_conf,
                                    predicted=predicted, motion_boxes=motion, budget=budget,
                                    exhaustive=exhaustive, imgsz=imgsz)
        if crops:
            images = [np.ascontiguousarray(frame[c['rect'][1]:c['rect'][3], c['rect'][0]:c['rect'][2]])
                      for c in crops]
            results = run_inference(images, conf, m, lk)
            for index, (crop, result) in enumerate(zip(crops, results)):
                x1, y1, x2, y2 = crop['rect']
                found = _boxes_from_result(result, offset=(x1, y1), frame_size=(w, h))
                for b in found:
                    b['source'] = 'zoom'
                    b['zoom'] = crop['zoom']
                    b['crop_index'] = index
                    bx = b['xyxy']
                    # touching an inner crop edge (not an image edge) -> probably cut off
                    b['truncated'] = ((bx[0] <= x1 + 2 and x1 > 0) or (bx[1] <= y1 + 2 and y1 > 0)
                                      or (bx[2] >= x2 - 3 and x2 < w) or (bx[3] >= y2 - 3 and y2 < h))
                crop['detections'] = len(found)
                crop_boxes.extend(found)
    t2 = time.perf_counter()

    merged = merge_detections(original + crop_boxes, cfg['merge_iou'], cfg['merge_ios'],
                              cfg['cross_class_merge_iou'])
    for b in merged:
        b['zoom_only'] = 'original' not in b['sources']
    t3 = time.perf_counter()
    if scheduler is not None:
        scheduler.frame_index += 1

    return {
        'boxes': merged,
        'crops': crops,
        'stats': {
            'original_detections': len(original),
            'low_conf_candidates': len(low_conf),
            'crops': len(crops),
            'crop_detections': len(crop_boxes),
            'zoom_only_detections': sum(1 for b in merged if b['zoom_only']),
            'merged_detections': len(merged),
            'motion_candidates': len(motion),
            'crop_budget': scheduler.budget if scheduler is not None else None,
        },
        'timings': {
            'original_ms': (t1 - t0) * 1000.0,
            'zoom_ms': (t2 - t1) * 1000.0,
            'merge_ms': (t3 - t2) * 1000.0,
        },
    }


# ============================================================
# PERFORMANCE HELPERS
# ============================================================
class RollingStat:
    """Last value, mean and p95 over a sliding window."""

    def __init__(self, size=120):
        self.values = deque(maxlen=size)
        self.last = 0.0

    def add(self, value):
        self.last = float(value)
        self.values.append(float(value))

    def reset(self):
        self.values.clear()
        self.last = 0.0

    def summary(self):
        if not self.values:
            return {'last': 0.0, 'avg': 0.0, 'p95': 0.0}
        ordered = sorted(self.values)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        return {'last': round(self.last, 2),
                'avg': round(sum(ordered) / len(ordered), 2),
                'p95': round(p95, 2)}


class RateMeter:
    """Events per second over a sliding time window; decays to 0 when events stop."""

    def __init__(self, window_s=2.0):
        self.window_s = window_s
        self.times = deque()

    def tick(self, t=None):
        t = time.perf_counter() if t is None else t
        self.times.append(t)
        self._prune(t)

    def _prune(self, now):
        while self.times and now - self.times[0] > self.window_s:
            self.times.popleft()

    def rate(self):
        self._prune(time.perf_counter())
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0

    def reset(self):
        self.times.clear()


class LatestFrameBuffer:
    """
    Single-slot, thread-safe buffer. put() overwrites an unprocessed frame
    (and counts it as dropped); get_latest() always returns the newest frame.
    No queue can ever build up.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._item = None
        self.dropped = 0

    def put(self, item):
        with self._cond:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._cond.notify()

    def get_latest(self, timeout=0.5):
        with self._cond:
            if self._item is None:
                self._cond.wait(timeout)
            item, self._item = self._item, None
            return item

    def wake(self):
        with self._cond:
            self._cond.notify_all()


class FramePacket:
    """Everything that belongs to ONE captured frame. Immutable after publish, so
    the feed, the zoom view, the 2D map and the 3D view can never mix frames."""
    __slots__ = ('frame_id', 'capture_wall', 'capture_perf', 'raw', 'refined', 'display', 'annotated',
                 'detections', 'tentative', 'predicted', 'crops', 'focus', 'ms_stats', 'refine_info',
                 'timings', 'published_perf', '_jpeg', '_zoom_jpeg', '_lock')

    def __init__(self, frame_id, capture_wall, capture_perf, raw, refined, display, annotated,
                 detections, refine_info, timings, jpeg=None, tentative=None, predicted=None,
                 crops=None, focus=None, ms_stats=None):
        self.frame_id = frame_id
        self.capture_wall = capture_wall
        self.capture_perf = capture_perf
        self.raw = raw
        self.refined = refined      # exactly what YOLO saw
        self.display = display      # operator view (refined + display-only brightening)
        self.annotated = annotated
        self.detections = detections
        self.tentative = tentative or []
        self.predicted = predicted or []
        self.crops = crops or []
        self.focus = focus
        self.ms_stats = ms_stats or {}
        self.refine_info = refine_info
        self.timings = timings
        self.published_perf = None
        self._jpeg = jpeg
        self._zoom_jpeg = None
        self._lock = threading.Lock()

    def get_jpeg(self):
        """Annotated JPEG, encoded at most once per packet (shared by all viewers)."""
        if self._jpeg is None:
            with self._lock:
                if self._jpeg is None:
                    self._jpeg = encode_jpeg(self.annotated, jpeg_quality) or b''
        return self._jpeg

    def get_zoom_jpeg(self):
        """Zoom view rendered lazily from THIS packet's own frame + detections."""
        if self._zoom_jpeg is None:
            with self._lock:
                if self._zoom_jpeg is None:
                    focus = self.focus or {'mode': 'idle', 'rect': [0, 0, self.display.shape[1], self.display.shape[0]],
                                           'zoom': 1.0, 'reason': 'waiting for target'}
                    view = render_zoom_view(self.display, focus, self.detections, self.tentative, self.frame_id)
                    self._zoom_jpeg = encode_jpeg(view, jpeg_quality) or b''
        return self._zoom_jpeg

    def meta(self):
        h, w = self.raw.shape[:2]
        crops = [{'rect': c['rect'], 'zoom': c['zoom'], 'reason': c['reason'],
                  'detections': c.get('detections', 0), 'tile': c.get('tile'),
                  'target': c.get('target')} for c in self.crops]
        return {
            'frame_id': self.frame_id,
            'capture_time': datetime.fromtimestamp(self.capture_wall).isoformat(timespec='milliseconds'),
            'frame_size': {'width': w, 'height': h},
            'detections': self.detections,
            'tentative_detections': self.tentative,
            'predicted_tracks': self.predicted,
            'zoom_crops': crops,
            'zoom_focus': self.focus,
            'multiscale': self.ms_stats,
            'timings_ms': {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.timings.items()},
            'refinement': {'enabled': self.refine_info.get('enabled'), 'applied': self.refine_info.get('applied')},
        }


def encode_jpeg(image, quality=None):
    q = int(jpeg_quality if quality is None else quality)
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else None


def get_stream_client_count():
    with stream_clients_lock:
        return stream_clients


class CameraPipeline:
    """Owns the VideoCapture, the capture thread and the single processing
    (= single inference) thread for one camera session."""

    def __init__(self, source, width=640, height=480, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.is_file = isinstance(source, str) and os.path.isfile(source)
        self.is_network = isinstance(source, str) and '://' in source
        self.cap = None
        self.buffer = LatestFrameBuffer()
        self.stop_event = threading.Event()
        self.packet_cond = threading.Condition()
        self.latest_packet = None
        self.last_detection_packet = None
        self.capture_thread = None
        self.process_thread = None
        self.frame_id = 0
        self.started_at = None
        self.last_error = None
        self.scheduler = ZoomScheduler()
        self.session_id = uuid.uuid4().hex[:8]
        self.stats_lock = threading.Lock()
        self._reset_counters()

    # ---------- statistics ----------
    def _reset_counters(self):
        self.frames_captured = 0
        self.frames_processed = 0
        self.capture_failures = 0
        self.reconnects = 0
        self.process_errors = 0
        self.dropped_offset = self.buffer.dropped
        self.capture_meter = RateMeter()
        self.detection_meter = RateMeter()
        self.display_meter = RateMeter()
        self.t_refine = RollingStat()
        self.t_infer = RollingStat()
        self.t_zoom = RollingStat()
        self.t_merge = RollingStat()
        self.t_annotate = RollingStat()
        self.t_encode = RollingStat()
        self.t_processing = RollingStat()
        self.t_queue_wait = RollingStat()
        self.t_frame_age = RollingStat()
        self.n_crops = RollingStat()
        self.n_original = RollingStat()
        self.n_crop_dets = RollingStat()
        self.n_zoom_only = RollingStat()
        self.n_merged = RollingStat()
        self.crops_total = 0
        self.zoom_only_total = 0
        self.q_frames = 0
        self.q_frames_with_targets = 0
        self.q_targets = 0
        self.q_conf_sum = 0.0
        self.last_refine_info = None
        self.last_ms_stats = {}
        self.last_processed_frame_id = 0

    def reset_stats(self):
        with self.stats_lock:
            self._reset_counters()

    def note_delivered(self):
        with self.stats_lock:
            self.display_meter.tick()

    def snapshot(self):
        with self.stats_lock:
            n = max(self.q_frames, 1)
            return {
                'source': str(self.source),
                'running': self.running,
                'capture_fps': round(self.capture_meter.rate(), 1),
                'detection_fps': round(self.detection_meter.rate(), 1),
                'displayed_fps': round(self.display_meter.rate(), 1),   # frames actually sent to viewers
                'refine_ms': self.t_refine.summary(),
                'inference_ms': self.t_infer.summary(),                 # YOLO on the original frame
                'zoom_inference_ms': self.t_zoom.summary(),             # YOLO on the zoom crops (batched)
                'merge_ms': self.t_merge.summary(),
                'annotate_ms': self.t_annotate.summary(),
                'encode_ms': self.t_encode.summary(),
                'processing_ms': self.t_processing.summary(),           # whole per-frame app work
                'queue_wait_ms': self.t_queue_wait.summary(),           # capture -> processing start
                'frame_age_ms': self.t_frame_age.summary(),             # capture -> published (app latency)
                'frames_captured': self.frames_captured,
                'frames_dropped': self.buffer.dropped - self.dropped_offset,
                'frames_processed': self.frames_processed,
                'current_frame_id': self.frame_id,
                'last_processed_frame_id': self.last_processed_frame_id,
                'capture_failures': self.capture_failures,
                'reconnects': self.reconnects,
                'process_errors': self.process_errors,
                'last_error': self.last_error,
                'enhancement_enabled': bool(refine_config.get('enabled')),
                'last_refinement': self.last_refine_info,
                'stream_clients': get_stream_client_count(),
                'zoom': {
                    'enabled': bool(_zoom_config.get('enabled')),
                    'crops_per_frame': self.n_crops.summary(),
                    'crops_total': self.crops_total,
                    'crop_budget': self.scheduler.budget,
                    'coarse_scans': self.scheduler.coarse_scans,
                    'original_detections': self.n_original.summary(),
                    'crop_detections': self.n_crop_dets.summary(),
                    'additional_zoom_detections': self.n_zoom_only.summary(),
                    'additional_zoom_detections_total': self.zoom_only_total,
                    'merged_detections': self.n_merged.summary(),
                    'last_frame': self.last_ms_stats,
                },
                'detection_quality': {
                    'frames': self.q_frames,
                    'frames_with_targets': self.q_frames_with_targets,
                    'hit_rate': round(self.q_frames_with_targets / n, 3),
                    'avg_targets_per_frame': round(self.q_targets / n, 3),
                    'avg_confidence': round(self.q_conf_sum / max(self.q_targets, 1), 3),
                },
            }

    # ---------- lifecycle ----------
    @property
    def running(self):
        return not self.stop_event.is_set()

    def _open(self):
        source = self.source
        params = []
        if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
            params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAPTURE_OPEN_TIMEOUT_MS]
        if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
            params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, CAPTURE_READ_TIMEOUT_MS]
        if isinstance(source, int):
            backend = {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF,
                       'v4l2': cv2.CAP_V4L2}.get(CAMERA_BACKEND, cv2.CAP_ANY)
            cap = cv2.VideoCapture(source, backend)
        elif self.is_network:
            try:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
            except Exception:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            return None
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # honoured by some backends only
        return cap

    def start(self):
        cap = self._open()
        if cap is None:
            self.stop_event.set()
            return False, f'Could not open camera source: {self.source}'
        self.cap = cap
        self.started_at = time.time()
        self.capture_thread = threading.Thread(target=self._capture_loop, name='capture', daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, name='process', daemon=True)
        self.capture_thread.start()
        self.process_thread.start()
        return True, 'Camera started successfully'

    def stop(self, timeout=4.0):
        """Signal both workers, wake any waiters and join. The capture thread
        releases the VideoCapture itself (never released mid-read from here)."""
        self.stop_event.set()
        self.buffer.wake()
        with self.packet_cond:
            self.packet_cond.notify_all()
        clean = True
        for t in (self.process_thread, self.capture_thread):
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout)
                clean = clean and not t.is_alive()
        if not clean:
            print("[camera] worker did not stop within timeout (blocked read); it will exit "
                  "and release the device when the read times out")
        return clean

    # ---------- capture thread ----------
    def _reconnect(self):
        delay = 0.5
        while not self.stop_event.is_set():
            print(f"[camera] reconnecting to {self.source} ...")
            cap = self._open()
            if cap is not None:
                with self.stats_lock:
                    self.reconnects += 1
                print("[camera] reconnected")
                return cap
            if self.stop_event.wait(delay):
                break
            delay = min(delay * 2, RECONNECT_MAX_DELAY_S)
        return None

    def _capture_loop(self):
        cap = self.cap
        consecutive_failures = 0
        file_period = 0.0
        if self.is_file:
            file_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            file_period = 1.0 / max(1.0, min(file_fps, 120.0))
        next_due = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                if cap is None:
                    cap = self._reconnect()
                    self.cap = cap
                    if cap is None:
                        break
                    consecutive_failures = 0
                    continue

                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    with self.stats_lock:
                        self.capture_failures += 1
                    if self.is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop test files
                        continue
                    if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                        self.last_error = 'stream lost - reconnecting'
                        cap.release()
                        cap = None
                        continue
                    self.stop_event.wait(0.01)
                    continue
                consecutive_failures = 0

                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                now = time.perf_counter()
                self.frame_id += 1
                # cap.read() allocates a fresh array each call -> no copy needed.
                self.buffer.put((self.frame_id, time.time(), now, frame))
                with self.stats_lock:
                    self.frames_captured += 1
                    self.capture_meter.tick(now)

                if file_period:   # play files at their native rate, like a live camera
                    next_due += file_period
                    sleep_for = next_due - time.perf_counter()
                    if sleep_for > 0:
                        self.stop_event.wait(sleep_for)
                    else:
                        next_due = time.perf_counter()
        except Exception as e:
            self.last_error = f'capture error: {e}'
            print(f"[camera] capture thread error: {e}")
        finally:
            if cap is not None:
                cap.release()
            self.cap = None
            print("[camera] capture thread stopped, VideoCapture released")

    # ---------- processing thread (the ONLY live inference worker) ----------
    def _process_loop(self):
        last_err_print = 0.0
        while not self.stop_event.is_set():
            item = self.buffer.get_latest(timeout=0.5)
            if item is None:
                continue
            try:
                packet = self._process(item)
            except Exception as e:
                with self.stats_lock:
                    self.process_errors += 1
                self.last_error = f'processing error: {e}'
                if time.time() - last_err_print > 2.0:
                    import traceback
                    traceback.print_exc()
                    last_err_print = time.time()
                continue
            self._publish(packet)
        print("[camera] processing thread stopped")

    def _process(self, item):
        frame_id, capture_wall, capture_perf, raw = item
        t_start = time.perf_counter()
        h, w = raw.shape[:2]

        # 1. refine the SAME frame that YOLO and the zoom crops will use
        refined, display_lut, refine_info = refine_frame_full(raw, get_refine_config())
        display = display_frame(refined, display_lut)
        t_refined = time.perf_counter()

        # 2. multi-scale detection + tracking
        zcfg = get_zoom_config()
        detections, tentative, predicted, crops = [], [], [], []
        ms_stats, ms_t = {}, {'original_ms': 0.0, 'zoom_ms': 0.0, 'merge_ms': 0.0}
        if detection_enabled:
            ms = detect_multiscale(refined, confidence_threshold, zcfg, self.scheduler, camera_tracker,
                                   capture_perf, motion_frame=raw)
            crops, ms_stats, ms_t = ms['crops'], ms['stats'], ms['timings']
            all_dets = boxes_to_detections(ms['boxes'], w, h, hide_edge_detections)
            camera_tracker.update(all_dets, timestamp=capture_perf,
                                  confirm_hits=zcfg['crop_confirm_hits'] if zcfg['enabled'] else 1)
            add_range_estimates(all_dets, range_focal_px, drone_real_width_m)
            detections = [d for d in all_dets if d.get('confirmed', True)]
            tentative = [d for d in all_dets if not d.get('confirmed', True)]
            predicted = camera_tracker.coasting(capture_perf)
        t_detected = time.perf_counter()

        # 3. annotate (boxes on the original or the enhanced image)
        base = raw if stream_base == 'raw' else display
        annotated = draw_detections(base, detections, show_center, show_crosshair, tentative=tentative)
        draw_track_overlay(annotated, detections, predicted, crops, show_trails, show_zoom_regions)
        det_fps = self.detection_meter.rate()
        k = ui_scale(w)
        t1, t2 = max(1, int(round(k))), max(2, int(round(2 * k)))
        cv2.putText(annotated, f"FPS: {int(round(det_fps))}",
                    (10, int(30 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 255, 0), t2)
        cv2.putText(annotated, f"Targets: {len(detections)}",
                    (10, int(60 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 255, 0), t2)
        status = "ACTIVE" if detection_enabled else "PAUSED"
        status_color = (0, 255, 0) if detection_enabled else (0, 0, 255)
        cv2.putText(annotated, f"Detection: {status}",
                    (10, int(90 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, status_color, t2)
        cv2.putText(annotated, f"YOLO11 {model_imgsz()}",
                    (w - int(140 * k), int(30 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (255, 255, 255), t1)
        cv2.putText(annotated, f"Conf: {confidence_threshold:.2f}",
                    (w - int(150 * k), int(60 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (255, 255, 255), t1)
        cv2.putText(annotated,
                    f"#{frame_id} | ENH {'ON' if refine_info['enabled'] else 'OFF'} | ZOOM {len(crops)} crop(s) | "
                    f"CAP {self.capture_meter.rate():.0f}fps | APP LAT {self.t_frame_age.last:.0f}ms",
                    (10, h - int(12 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (0, 255, 255), t1, cv2.LINE_AA)
        if overlay_scene_map:
            annotated = draw_scene_map_overlay(annotated, detections, inplace=True)
        focus = select_zoom_focus(detections, tentative, crops, w, h, zcfg)
        t_annotated = time.perf_counter()

        # 4. JPEG once, only if an MJPEG viewer is connected (dashboard requests encode lazily)
        jpeg = encode_jpeg(annotated) if get_stream_client_count() > 0 else None
        t_encoded = time.perf_counter()

        timings = {
            'refine_ms': (t_refined - t_start) * 1000.0,
            'inference_ms': ms_t['original_ms'],
            'zoom_inference_ms': ms_t['zoom_ms'],
            'merge_ms': ms_t['merge_ms'],
            'detect_total_ms': (t_detected - t_refined) * 1000.0,
            'annotate_ms': (t_annotated - t_detected) * 1000.0,
            'encode_ms': (t_encoded - t_annotated) * 1000.0,
            'processing_ms': (t_encoded - t_start) * 1000.0,
            'queue_wait_ms': (t_start - capture_perf) * 1000.0,
            'detection_ran': bool(detection_enabled),
            'encoded': jpeg is not None,
        }
        self.scheduler.feedback(timings['processing_ms'], zcfg)
        return FramePacket(frame_id, capture_wall, capture_perf, raw, refined, display, annotated,
                           detections, refine_info, timings, jpeg, tentative=tentative,
                           predicted=predicted, crops=crops, focus=focus, ms_stats=ms_stats)

    def _publish(self, packet):
        global current_frame
        now = time.perf_counter()
        packet.published_perf = now
        packet.timings['frame_age_ms'] = (now - packet.capture_perf) * 1000.0

        # Atomic hand-over: one reference swap under the condition lock.
        with self.packet_cond:
            self.latest_packet = packet
            if packet.detections:
                self.last_detection_packet = packet
            current_frame = packet.annotated
            self.packet_cond.notify_all()

        t = packet.timings
        s = packet.ms_stats
        with self.stats_lock:
            self.frames_processed += 1
            self.last_processed_frame_id = packet.frame_id
            self.detection_meter.tick(now)
            self.t_refine.add(t['refine_ms'])
            if t['detection_ran']:
                self.t_infer.add(t['inference_ms'])
                self.t_zoom.add(t['zoom_inference_ms'])
                self.t_merge.add(t['merge_ms'])
                self.n_crops.add(s.get('crops', 0))
                self.n_original.add(s.get('original_detections', 0))
                self.n_crop_dets.add(s.get('crop_detections', 0))
                self.n_zoom_only.add(s.get('zoom_only_detections', 0))
                self.n_merged.add(s.get('merged_detections', 0))
                self.crops_total += s.get('crops', 0)
                self.zoom_only_total += s.get('zoom_only_detections', 0)
                self.last_ms_stats = s
            self.t_annotate.add(t['annotate_ms'])
            if t['encoded']:
                self.t_encode.add(t['encode_ms'])
            self.t_processing.add(t['processing_ms'])
            self.t_queue_wait.add(t['queue_wait_ms'])
            self.t_frame_age.add(t['frame_age_ms'])
            self.last_refine_info = packet.refine_info
            if t['detection_ran']:
                self.q_frames += 1
                if packet.detections:
                    self.q_frames_with_targets += 1
                self.q_targets += len(packet.detections)
                self.q_conf_sum += sum(d['confidence'] for d in packet.detections)
            det_fps = self.detection_meter.rate()

        with detection_stats_lock:
            detection_stats['fps'] = int(round(det_fps))
            detection_stats['detections'] = packet.detections
            detection_stats['total_detections'] = len(packet.detections)
            detection_stats['frame_id'] = packet.frame_id
            detection_stats['tentative_detections'] = len(packet.tentative)
            if packet.detections:
                detection_stats['last_detection_time'] = datetime.now().strftime('%H:%M:%S')

    def wait_for_packet(self, last_id, timeout=1.0):
        deadline = time.perf_counter() + timeout
        with self.packet_cond:
            while not self.stop_event.is_set() and (
                    self.latest_packet is None or self.latest_packet.frame_id <= last_id):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self.packet_cond.wait(remaining)
            packet = self.latest_packet
        return packet if packet is not None and packet.frame_id > last_id else None

    def get_latest_packet(self, with_detections=False):
        with self.packet_cond:
            return self.last_detection_packet if with_detections else self.latest_packet


# ============================================================
# TARGET ZOOM (refined crops of detected drones)
# ============================================================
def build_target_crops(image, detections, max_targets=4, tile_size=224, pad_ratio=0.35):
    """Square, upscaled crops around the best detections (for on-screen inspection only;
    YOLO never sees these)."""
    crops = []
    if image is None or not detections:
        return crops
    h, w = image.shape[:2]
    ordered = sorted(detections, key=lambda d: d['confidence'], reverse=True)[:max_targets]
    for det in ordered:
        bw, bh = det['x2'] - det['x1'], det['y2'] - det['y1']
        side = max(32, int(max(bw, bh) * (1 + 2 * pad_ratio)))
        cx, cy = det['center']
        x1 = int(max(0, min(w - 1, cx - side // 2)))
        y1 = int(max(0, min(h - 1, cy - side // 2)))
        x2 = int(min(w, x1 + side))
        y2 = int(min(h, y1 + side))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        scale = tile_size / max(ch, cw)
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        tile = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=interp)
        canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        oy = (tile_size - tile.shape[0]) // 2
        ox = (tile_size - tile.shape[1]) // 2
        canvas[oy:oy + tile.shape[0], ox:ox + tile.shape[1]] = tile
        bx1 = int((det['x1'] - x1) * scale) + ox
        by1 = int((det['y1'] - y1) * scale) + oy
        bx2 = int((det['x2'] - x1) * scale) + ox
        by2 = int((det['y2'] - y1) * scale) + oy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 0), 1, cv2.LINE_AA)
        label = f"#{det.get('track_id', '-')} {det['class']} {det['confidence']:.2f} x{scale:.1f}"
        cv2.rectangle(canvas, (0, 0), (tile_size, 18), (0, 0, 0), -1)
        cv2.putText(canvas, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        if 'estimated_distance_m' in det:
            cv2.putText(canvas, f"{det['estimated_distance_m']:.1f} m", (4, tile_size - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        crops.append((canvas, det))
    return crops


def build_target_mosaic(image, detections, max_targets=4, tile_size=224):
    crops = build_target_crops(image, detections, max_targets, tile_size)
    if not crops:
        return None
    tiles = []
    for index, (canvas, _) in enumerate(crops):
        tiles.append(canvas)
        if index < len(crops) - 1:
            tiles.append(np.full((tile_size, 4, 3), 40, dtype=np.uint8))
    return np.hstack(tiles)


# ============================================================
# TRACKING (stable IDs, also for fast drones)
# ============================================================
class DetectionTracker:
    """Greedy tracker with stable IDs, a smoothed image center and a
    constant-velocity prediction.

    Changes vs. the original IoU-only tracker (same update(detections) API):
      * matching uses the track's PREDICTED box (position + velocity * dt), so a
        fast drone that moved more than its own width between frames keeps its ID;
      * if the predicted box does not overlap, a centre-distance gate
        (gate_scale x target size, at least min_gate_px) still allows the match;
      * the smoothed centre blends the prediction with the measurement
        (alpha-beta filter) instead of a plain moving average, so it no longer
        lags behind fast targets;
      * velocity (px/s), heading, hits, age and a short trail are exposed per target;
      * detections found only in zoom crops can be held as 'tentative' until they
        were seen `confirm_hits` times (update(..., confirm_hits=N));
      * cross_class=True lets a track continue when the class label flickers between
        frames (e.g. drone <-> octocopter at long range). Same-class matches always
        score higher; the track takes the latest class. False = original behaviour.
    """

    def __init__(self, max_missed=8, smoothing=0.65, gate_scale=2.5, min_gate_px=24.0,
                 velocity_gain=0.5, trail_length=40, cross_class=True, max_speed_px_s=1500.0):
        self.max_missed = max_missed
        self.smoothing = smoothing
        self.gate_scale = gate_scale
        self.min_gate_px = min_gate_px
        self.velocity_gain = velocity_gain
        self.trail_length = trail_length
        self.cross_class = cross_class
        self.max_speed_px_s = max_speed_px_s   # search radius for tracks whose velocity is not known yet
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _now(timestamp):
        return time.perf_counter() if timestamp is None else timestamp

    @staticmethod
    def _predicted_box(track, now):
        dt = max(0.0, min(1.0, now - track['time']))
        vx, vy = track['velocity']
        x1, y1, x2, y2 = track['box']
        return (x1 + vx * dt, y1 + vy * dt, x2 + vx * dt, y2 + vy * dt)

    def _gate(self, box_a, box_b, track=None, dt=0.0):
        size = max(box_a[2] - box_a[0], box_a[3] - box_a[1], box_b[2] - box_b[0], box_b[3] - box_b[1])
        gate = max(self.min_gate_px, self.gate_scale * size)
        if track is not None and dt > 0:
            if track['hits'] < 3:
                # velocity unknown yet: allow a fast start, but never a search radius of more than
                # 0.2 s of travel (otherwise a lost track could grab any detection in the frame)
                gate += self.max_speed_px_s * min(dt, 0.2)
            else:
                gate += 0.5 * math.hypot(*track['velocity']) * min(dt, 0.5)   # uncertainty grows with speed
        return gate

    def _match_score(self, track, pbox, box, dt=0.0):
        iou = _iou_ios(pbox, box)[0]
        if iou >= 0.2:
            return 1.0 + iou                      # overlap matches always win
        pcx, pcy = (pbox[0] + pbox[2]) / 2.0, (pbox[1] + pbox[3]) / 2.0
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        gate = self._gate(pbox, box, track, dt)
        dist = math.hypot(cx - pcx, cy - pcy)
        return 1.0 - dist / gate if dist < gate else None

    def update(self, detections, timestamp=None, confirm_hits=1):
        now = self._now(timestamp)
        candidates = []
        for track_id, track in self.tracks.items():
            pbox = self._predicted_box(track, now)
            dt = max(0.0, min(1.0, now - track['time']))
            for index, detection in enumerate(detections):
                same_class = track['class'] == detection['class']
                if not same_class and not self.cross_class:
                    continue
                box = (detection['x1'], detection['y1'], detection['x2'], detection['y2'])
                score = self._match_score(track, pbox, box, dt)
                if score is not None:
                    candidates.append((score if same_class else score * 0.5, track_id, index))

        candidates.sort(reverse=True)
        matched_tracks = set()
        matched_detections = set()
        for _, track_id, index in candidates:
            if track_id in matched_tracks or index in matched_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[index]
            dt = max(1e-3, now - track['time'])
            mx, my = detection['center']
            old_vx, old_vy = track['velocity']
            if dt <= 1.0:
                inst_vx = (mx - track['meas'][0]) / dt
                inst_vy = (my - track['meas'][1]) / dt
                gain = 1.0 if track['hits'] == 1 else self.velocity_gain
                vx = old_vx * (1 - gain) + inst_vx * gain
                vy = old_vy * (1 - gain) + inst_vy * gain
            else:
                vx = vy = 0.0
            # alpha-beta style smoothing: predict the smoothed centre, then blend with the measurement
            pred_x = track['center'][0] + old_vx * min(dt, 1.0)
            pred_y = track['center'][1] + old_vy * min(dt, 1.0)
            smooth_x = int(pred_x * self.smoothing + mx * (1 - self.smoothing))
            smooth_y = int(pred_y * self.smoothing + my * (1 - self.smoothing))
            track.update({
                'box': (detection['x1'], detection['y1'], detection['x2'], detection['y2']),
                'center': (smooth_x, smooth_y), 'meas': (mx, my), 'velocity': (vx, vy),
                'class': detection['class'], 'missed': 0, 'time': now,
                'hits': track['hits'] + 1, 'confidence': detection['confidence'],
            })
            track['trail'].append((smooth_x, smooth_y))
            if not detection.get('zoom_only') or track['hits'] >= confirm_hits:
                track['confirmed'] = True
            self._annotate(detection, track_id, track, now)
            matched_tracks.add(track_id)
            matched_detections.add(index)

        for track_id in list(self.tracks):
            if track_id not in matched_tracks:
                self.tracks[track_id]['missed'] += 1
                if self.tracks[track_id]['missed'] > self.max_missed:
                    del self.tracks[track_id]

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            center = tuple(detection['center'])
            self.tracks[track_id] = {
                'box': (detection['x1'], detection['y1'], detection['x2'], detection['y2']),
                'center': center, 'start_center': center, 'meas': center,
                'velocity': (0.0, 0.0), 'class': detection['class'], 'missed': 0,
                'time': now, 'born': now, 'hits': 1, 'confidence': detection['confidence'],
                'confirmed': (not detection.get('zoom_only')) or confirm_hits <= 1,
                'trail': deque([center], maxlen=self.trail_length),
            }
            self._annotate(detection, track_id, self.tracks[track_id], now)
        return detections

    @staticmethod
    def _annotate(detection, track_id, track, now):
        vx, vy = track['velocity']
        detection['track_id'] = track_id
        detection['smooth_center'] = [int(track['center'][0]), int(track['center'][1])]
        detection['track_start_center'] = [int(v) for v in track['start_center']]
        detection['velocity_px_s'] = [round(vx, 1), round(vy, 1)]
        detection['speed_px_s'] = round(math.hypot(vx, vy), 1)
        detection['heading_deg'] = round(math.degrees(math.atan2(-vy, vx)), 1)  # 0 = right, 90 = up
        detection['track_hits'] = track['hits']
        detection['track_age_s'] = round(now - track['born'], 2)
        detection['confirmed'] = bool(track['confirmed'])
        detection['trail'] = [[int(x), int(y)] for x, y in list(track['trail'])[-20:]]

    def annotate_track_ids(self, boxes, timestamp=None):
        """Attach the most likely existing track id to raw box dicts (used for zoom cooldowns)."""
        now = self._now(timestamp)
        for b in boxes:
            name = model.names[b['cls']]
            best, best_score = None, None
            for track_id, track in self.tracks.items():
                if track['class'] != name and not self.cross_class:
                    continue
                score = self._match_score(track, self._predicted_box(track, now), b['xyxy'],
                                          max(0.0, min(1.0, now - track['time'])))
                if score is not None and (best_score is None or score > best_score):
                    best, best_score = track_id, score
            if best is not None:
                b['track_id'] = best

    def predict_lost(self, timestamp=None, max_missed=6, boxes=None):
        """Predicted boxes of tracks that were missed recently or that no box of
        this frame's original pass explains -> 're-acquire' zoom crops."""
        now = self._now(timestamp)
        out = []
        for track_id, track in self.tracks.items():
            if track['missed'] > max_missed:
                continue
            pbox = self._predicted_box(track, now)
            explained = False
            if boxes:
                for b in boxes:
                    name = model.names[b['cls']]
                    if ((name == track['class'] or self.cross_class) and self._match_score(
                            track, pbox, b['xyxy'], max(0.0, min(1.0, now - track['time']))) is not None):
                        explained = True
                        break
            if explained:
                continue
            if track['missed'] == 0 and boxes is None:
                continue
            out.append({'xyxy': [float(v) for v in pbox], 'track_id': track_id, 'class': track['class'],
                        'confidence': track['confidence'], 'missed': max(1, track['missed']),
                        'confirmed': bool(track['confirmed'])})
        return out

    def coasting(self, timestamp=None, max_missed=None):
        """Tracks not seen this frame but still alive -> 'PRED' markers (display only, never counted)."""
        now = self._now(timestamp)
        limit = self.max_missed if max_missed is None else max_missed
        out = []
        for track_id, track in self.tracks.items():
            if 1 <= track['missed'] <= limit and track['confirmed']:
                pbox = self._predicted_box(track, now)
                out.append({'track_id': track_id, 'class': track['class'],
                            'center': [int((pbox[0] + pbox[2]) / 2), int((pbox[1] + pbox[3]) / 2)],
                            'box': [int(v) for v in pbox], 'missed': track['missed'],
                            'confidence': round(track['confidence'], 3),
                            'velocity_px_s': [round(track['velocity'][0], 1), round(track['velocity'][1], 1)],
                            'trail': [[int(x), int(y)] for x, y in list(track['trail'])[-20:]]})
        return out


def add_range_estimates(detections, focal_length_px, target_width_m):
    """Pinhole estimate: range = focal_px * known_width_m / observed_width_px.
    With a range, the image-plane speed is also converted to an approximate
    lateral (cross-track) speed; motion towards/away from the camera is not measured."""
    if focal_length_px <= 0 or target_width_m <= 0:
        return detections
    for detection in detections:
        if detection['width'] > 0:
            detection['estimated_distance_m'] = round(
                focal_length_px * target_width_m / detection['width'], 2)
            if 'speed_px_s' in detection:
                detection['approx_lateral_speed_m_s'] = round(
                    detection['speed_px_s'] * detection['estimated_distance_m'] / focal_length_px, 2)
    return detections


camera_tracker = DetectionTracker()


def get_direction(center_x, center_y, frame_width, frame_height, margin=0.1):
    """Calculate direction of target relative to screen center"""
    screen_center_x = frame_width // 2
    screen_center_y = frame_height // 2

    offset_x = float((center_x - screen_center_x) / (frame_width / 2))
    offset_y = float((center_y - screen_center_y) / (frame_height / 2))

    if abs(offset_x) < margin and abs(offset_y) < margin:
        position = "Inside"
        direction_text = "Target aligned"
    else:
        position = "Outside"
        direction_text = ""

        right = max(0, offset_x)
        left = max(0, -offset_x)
        down = max(0, offset_y)
        up = max(0, -offset_y)

        parts = []
        if right > 0: parts.append(f"Right: {right:.2f}")
        if left > 0:  parts.append(f"Left: {left:.2f}")
        if up > 0:    parts.append(f"Up: {up:.2f}")
        if down > 0:  parts.append(f"Down: {down:.2f}")
        direction_text = ", ".join(parts)

    return {
        'position': position,
        'right': max(0, offset_x),
        'left': max(0, -offset_x),
        'up': max(0, -offset_y),
        'down': max(0, offset_y),
        'text': direction_text
    }


def ui_scale(frame_w):
    """Text / line scale for overlays: 1.0 for frames up to ~960 px wide (original look),
    proportionally larger on HD/4K frames so labels stay readable."""
    return max(1.0, min(4.0, frame_w / 960.0))


def boxes_to_detections(boxes, frame_w, frame_h, hide_edges=False, edge_margin=0.06):
    """Merged box dicts -> the detection dicts used everywhere (same keys as before + scale info)."""
    detections = []
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b['xyxy']]
        if hide_edges and (
            x1 <= frame_w * edge_margin or y1 <= frame_h * edge_margin
            or x2 >= frame_w * (1 - edge_margin) or y2 >= frame_h * (1 - edge_margin)
        ):
            continue
        conf = float(b['conf'])
        cls = int(b['cls'])
        class_name = model.names[cls]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        zoom_only = bool(b.get('zoom_only', False))
        detections.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'center': [int(center_x), int(center_y)],
            'width': int(x2 - x1), 'height': int(y2 - y1),
            'confidence': conf, 'class': class_name, 'class_id': cls,
            'direction': get_direction(center_x, center_y, frame_w, frame_h),
            'source': 'zoom' if zoom_only else 'original',
            'sources': list(b.get('sources', ['original'])),
            'zoom_only': zoom_only,
            'zoom_factor': float(b.get('zoom', 1.0)),
        })
    return detections


def _class_color(class_name):
    return (0, 255, 0) if class_name.lower() == 'drone' else \
        (255, 165, 0) if class_name.lower() == 'octocopter' else (0, 255, 255)


TRACK_COLORS = [(255, 225, 48), (210, 95, 255), (85, 255, 115), (75, 195, 255),
                (190, 135, 255), (80, 160, 255), (255, 120, 120), (120, 255, 220)]


def _track_color(track_id):
    return TRACK_COLORS[(track_id or 0) % len(TRACK_COLORS)]


def draw_detections(image, detections, show_center=True, show_crosshair=True, tentative=None, copy=True):
    """Same visuals as the original draw_detection_with_center(): screen-centre marker,
    boxes, class + confidence labels, centre dot, crosshair, coordinates and the
    direction arrow. Adds the tracking ID to the label and a 'Z' tag for targets
    that only the zoom pass found. Tentative boxes are drawn thin with '?'."""
    annotated = image.copy() if copy else image
    h, w = annotated.shape[:2]
    k = ui_scale(w)                       # 1.0 up to ~960 px wide -> identical to the old drawing
    t1, t2 = max(1, int(round(k))), max(2, int(round(2 * k)))

    screen_center_x, screen_center_y = w // 2, h // 2
    cv2.circle(annotated, (screen_center_x, screen_center_y), int(10 * k), (0, 255, 255), t2)
    cv2.line(annotated, (screen_center_x - int(50 * k), screen_center_y),
             (screen_center_x + int(50 * k), screen_center_y), (0, 255, 255), t1)
    cv2.line(annotated, (screen_center_x, screen_center_y - int(50 * k)),
             (screen_center_x, screen_center_y + int(50 * k)), (0, 255, 255), t1)
    cv2.putText(annotated, "SCREEN CENTER",
                (screen_center_x - int(60 * k), screen_center_y + int(30 * k)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (0, 255, 255), t1)

    for det in tentative or []:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (170, 170, 170), t1)
        cv2.putText(annotated, f"? {det['class']} {det['confidence']:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k, (200, 200, 200), t1, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        center_x, center_y = det['center']
        class_name = det['class']
        color = _class_color(class_name)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, t2)

        prefix = f"#{det['track_id']} " if det.get('track_id') is not None else ''
        suffix = f" Z{det['zoom_factor']:.0f}x" if det.get('zoom_only') else ''
        label = f"{prefix}{class_name} {det['confidence']:.2f}{suffix}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, t2)[0]
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - int(10 * k)),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - int(5 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (0, 0, 0), t2)

        if show_center:
            cv2.circle(annotated, (center_x, center_y), int(8 * k), (0, 0, 255), -1)
            cv2.circle(annotated, (center_x, center_y), int(12 * k), (0, 0, 255), t2)

            if show_crosshair:
                crosshair_size = int(20 * k)
                cv2.line(annotated, (center_x - crosshair_size, center_y),
                         (center_x + crosshair_size, center_y), (255, 0, 0), t2)
                cv2.line(annotated, (center_x, center_y - crosshair_size),
                         (center_x, center_y + crosshair_size), (255, 0, 0), t2)

            coord_text = f"({center_x}, {center_y})"
            cv2.putText(annotated, coord_text, (center_x + int(15 * k), center_y - int(10 * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (0, 0, 255), t1)

        direction = det['direction']
        if direction['position'] != 'Inside':
            arrow_length = int(30 * k)
            dx = direction['right'] - direction['left']
            dy = direction['down'] - direction['up']

            if dx != 0 or dy != 0:
                angle = math.atan2(-dy, dx)
                end_x = int(screen_center_x + arrow_length * math.cos(angle))
                end_y = int(screen_center_y - arrow_length * math.sin(angle))
                cv2.arrowedLine(annotated, (screen_center_x, screen_center_y),
                                (end_x, end_y), (0, 255, 255), t2)
    return annotated


def draw_detection_with_center(image, results, show_center=True, show_crosshair=True,
                               hide_edges=False, edge_margin=0.06):
    """Draw bounding boxes with center points and crosshairs.
    `results` may be Ultralytics results (as before) or merged multi-scale box dicts."""
    h, w = image.shape[:2]
    if isinstance(results, list) and (not results or isinstance(results[0], dict)):
        boxes = results
    else:
        boxes = _boxes_from_result(results[0])
    detections = boxes_to_detections(boxes, w, h, hide_edges, edge_margin)
    annotated = draw_detections(image, detections, show_center, show_crosshair)
    return annotated, detections


ZOOM_REASON_COLORS = {
    'reacquire': (255, 80, 255), 'low-confidence': (0, 165, 255), 'small-target': (255, 255, 0),
    'motion': (255, 140, 60), 'coarse-search': (150, 150, 150),
}


def _corner_brackets(img, rect, color, length=14, thickness=1):
    x1, y1, x2, y2 = [int(v) for v in rect]
    length = max(4, min(length, (x2 - x1) // 3, (y2 - y1) // 3))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)


def draw_track_overlay(image, detections, predicted=None, crops=None, show_trails=True, show_regions=True):
    """In place: zoom regions (corner brackets), movement trails, velocity arrows and
    PRED markers for targets that are being coasted (not detected this frame)."""
    k = ui_scale(image.shape[1])
    t1, t2 = max(1, int(round(k))), max(2, int(round(2 * k)))
    if show_regions:
        for crop in crops or []:
            color = ZOOM_REASON_COLORS.get(crop['reason'], (200, 200, 200))
            _corner_brackets(image, crop['rect'], color, int(14 * k), t1)
            x1, y1 = crop['rect'][0], crop['rect'][1]
            cv2.putText(image, f"{crop['zoom']:.1f}x {crop['reason']}", (x1 + 4, y1 + int(14 * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k, color, t1, cv2.LINE_AA)
    for det in detections:
        color = _track_color(det.get('track_id'))
        trail = det.get('trail') or []
        if show_trails and len(trail) > 1:
            cv2.polylines(image, [np.array(trail, dtype=np.int32)], False, color, t2, cv2.LINE_AA)
        vx, vy = det.get('velocity_px_s', [0, 0])
        if math.hypot(vx, vy) > 15:
            cx, cy = det['center']
            cv2.arrowedLine(image, (cx, cy), (int(cx + vx * 0.3), int(cy + vy * 0.3)), color, t2,
                            cv2.LINE_AA, tipLength=0.25)
    for p in predicted or []:
        color = _track_color(p['track_id'])
        cx, cy = p['center']
        cv2.circle(image, (cx, cy), int(10 * k), color, t1, cv2.LINE_AA)
        cv2.line(image, (cx - int(4 * k), cy), (cx + int(4 * k), cy), color, t1)
        cv2.line(image, (cx, cy - int(4 * k)), (cx, cy + int(4 * k)), color, t1)
        cv2.putText(image, f"#{p['track_id']} PRED", (cx + int(12 * k), cy + int(4 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * k, color, t1, cv2.LINE_AA)
    return image


def draw_scene_map_overlay(image, detections, inplace=False):
    """Embed a measured 2D image-plane detection map into a live or processed frame.
    inplace=True skips the full-frame copy when the caller already owns `image`."""
    frame = image if inplace else image.copy()
    height, width = frame.shape[:2]
    panel_width = min(width - 20, max(220, int(width * 0.34)))
    panel_height = min(height - 20, int(panel_width * 0.72))
    if panel_width < 120 or panel_height < 100:
        return frame

    x0, y0 = width - panel_width - 12, 12
    panel = frame[y0:y0 + panel_height, x0:x0 + panel_width]
    tint = panel.copy()
    cv2.rectangle(tint, (0, 0), (panel_width - 1, panel_height - 1), (9, 12, 18), -1)
    cv2.addWeighted(tint, 0.91, panel, 0.09, 0, panel)

    def image_point(u, v):
        top_left = (int(panel_width * 0.22), int(panel_height * 0.08))
        top_right = (int(panel_width * 0.78), int(panel_height * 0.08))
        bottom_left = (int(panel_width * 0.03), int(panel_height * 0.96))
        bottom_right = (int(panel_width * 0.97), int(panel_height * 0.96))
        left_x = top_left[0] + (bottom_left[0] - top_left[0]) * v
        right_x = top_right[0] + (bottom_right[0] - top_right[0]) * v
        y = top_left[1] + (bottom_left[1] - top_left[1]) * v
        return int(left_x + (right_x - left_x) * u), int(y)

    grid_color = (74, 82, 91)
    for index in range(7):
        u = index / 6
        cv2.line(panel, image_point(u, 0), image_point(u, 1), grid_color, 1, cv2.LINE_AA)
        v = index / 6
        cv2.line(panel, image_point(0, v), image_point(1, v), grid_color, 1, cv2.LINE_AA)
    boundary = np.array([image_point(0, 0), image_point(1, 0),
                         image_point(1, 1), image_point(0, 1)], dtype=np.int32)
    cv2.polylines(panel, [boundary], True, (120, 133, 148), 1, cv2.LINE_AA)
    cv2.line(panel, image_point(0.5, 0), image_point(0.5, 1), (85, 94, 104), 1, cv2.LINE_AA)
    cv2.line(panel, image_point(0, 0.5), image_point(1, 0.5), (85, 94, 104), 1, cv2.LINE_AA)

    frame_height, frame_width = image.shape[:2]
    view_origin = image_point(0.5, 0.88)
    cv2.circle(panel, view_origin, 3, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.putText(panel, 'CAM', (view_origin[0] + 5, view_origin[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.22, (0, 220, 255), 1, cv2.LINE_AA)
    marker_colors = [(255, 225, 48), (210, 95, 255), (85, 255, 115),
                     (75, 195, 255), (190, 135, 255)]
    best_target = None
    for index, detection in enumerate(detections):
        center_x, center_y = detection['center']
        smooth_center = detection.get('smooth_center', detection['center'])
        u = max(0.0, min(1.0, smooth_center[0] / max(frame_width, 1)))
        v = max(0.0, min(1.0, smooth_center[1] / max(frame_height, 1)))
        point = image_point(u, v)
        color = marker_colors[index % len(marker_colors)]
        for step in range(0, 10, 2):
            t1, t2 = step / 10, (step + 1) / 10
            ray_start = (int(view_origin[0] + (point[0] - view_origin[0]) * t1),
                         int(view_origin[1] + (point[1] - view_origin[1]) * t1))
            ray_end = (int(view_origin[0] + (point[0] - view_origin[0]) * t2),
                       int(view_origin[1] + (point[1] - view_origin[1]) * t2))
            cv2.line(panel, ray_start, ray_end, (0, 220, 255), 1, cv2.LINE_AA)
        start_center = detection.get('track_start_center', detection['center'])
        start_point = image_point(
            max(0.0, min(1.0, start_center[0] / max(frame_width, 1))),
            max(0.0, min(1.0, start_center[1] / max(frame_height, 1)))
        )
        cv2.line(panel, start_point, point, (90, 205, 255), 1, cv2.LINE_AA)
        cv2.circle(panel, start_point, 4, (90, 205, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, 'START', (start_point[0] + 4, start_point[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, (140, 220, 255), 1, cv2.LINE_AA)
        if best_target is None or detection['confidence'] > best_target['confidence']:
            best_target = detection

        # Draw a tiny shaded quadcopter instead of a point marker.
        size_ratio = max(detection['width'], detection['height']) / max(frame_width, 1)
        size = max(4, min(12, int(3 + size_ratio * 80)))
        motors = [(-size, -int(size * 0.58)), (size, -int(size * 0.58)),
                  (-size, int(size * 0.58)), (size, int(size * 0.58))]
        for dx, dy in motors:
            motor = (point[0] + dx, point[1] + dy)
            cv2.line(panel, point, motor, (35, 55, 61), max(2, size // 3), cv2.LINE_AA)
            cv2.line(panel, point, motor, color, 1, cv2.LINE_AA)
            cv2.ellipse(panel, motor, (max(3, size // 2), 2), -20, 0, 360,
                        (205, 215, 220), 1, cv2.LINE_AA)
        body = np.array([
            (point[0], point[1] - size // 2),
            (point[0] + size // 2, point[1]),
            (point[0], point[1] + size // 2),
            (point[0] - size // 2, point[1])
        ], dtype=np.int32)
        side = body + np.array([0, max(2, size // 4)], dtype=np.int32)
        cv2.fillPoly(panel, [side], (34, 54, 59))
        cv2.fillPoly(panel, [body], color)
        cv2.polylines(panel, [body], True, (235, 245, 245), 1, cv2.LINE_AA)
        distance = (f"{detection['estimated_distance_m']:.1f}m"
                    if 'estimated_distance_m' in detection else
                    'FAR~' if size_ratio < 0.055 else 'MID~' if size_ratio < 0.14 else 'NEAR~')
        label = (f"#{detection.get('track_id', index + 1)} {detection['class']} "
                 f"{detection['confidence']:.2f} {distance} ({center_x},{center_y})")
        label_x = min(max(14, point[0] + size + 2), panel_width - 150)
        label_y = max(26, point[1] - size - 2)
        cv2.putText(panel, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, (245, 245, 245), 1, cv2.LINE_AA)

    best_text = 'NO TARGETS' if best_target is None else (
        f"BEST #{best_target.get('track_id', 1)} {best_target['class']} "
        f"{best_target['confidence']:.2f}"
    )
    cv2.putText(panel, '3D-STYLE TARGET MAP', (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 240, 246), 1, cv2.LINE_AA)
    cv2.putText(panel, best_text, (8, panel_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (190, 205, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, 'DEPTH APPROXIMATE UNLESS CALIBRATED', (8, panel_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (190, 205, 220), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_width, y0 + panel_height),
                  (210, 220, 230), 1)
    return frame


# ============================================================
# AUTO-ZOOM VIEW (Panel B)
# ============================================================
def select_zoom_focus(detections, tentative, crops, frame_w, frame_h, cfg):
    """Pick what Panel B shows for this frame (always from the same frame).
    Priority: a confirmed target that needed zoom (zoom-only or small) ->
    highest-confidence confirmed target -> tentative target -> the search crop
    that ran -> idle (whole frame)."""
    frame_long = float(max(frame_w, frame_h))

    def target_rect(det):
        box = (det['x1'], det['y1'], det['x2'], det['y2'])
        size = max(2, det['width'], det['height'])
        # reuse the crop the detector actually used for this target, if any
        for crop in crops or []:
            if _box_inside(box, crop['rect'], margin=2) and (det.get('zoom_only') or
                                                          (crop.get('target') and _iou_ios(box, crop['target']['xyxy'])[0] > 0.1)):
                return crop['rect'], crop['zoom'], f"detector crop ({crop['reason']})"
        # otherwise frame the target so it fills ~1/4 of the view
        zoom = None
        for z in sorted(cfg['zoom_factors'], reverse=True):
            if frame_long / z >= size * 4:
                zoom = z
                break
        if zoom is None:
            zoom = max(1.0, frame_long / (size * 1.6))
        rect = _crop_rect(det['center'][0], det['center'][1], frame_w / zoom, frame_h / zoom, frame_w, frame_h)
        return rect, round(frame_long / max(rect[2] - rect[0], rect[3] - rect[1]), 2), 'display framing'

    if detections:
        small_limit = cfg['small_target_frac'] * frame_long

        def relevance(d):
            small = max(d['width'], d['height']) < small_limit
            return (bool(d.get('zoom_only')) or small, d['confidence'])

        focus = max(detections, key=relevance)
        rect, zoom, how = target_rect(focus)
        why = 'found by zoom pass' if focus.get('zoom_only') else (
            'small / distant target' if max(focus['width'], focus['height']) < small_limit else 'highest confidence')
        return {'mode': 'target', 'rect': [int(v) for v in rect], 'zoom': zoom, 'framing': how,
                'reason': why, 'track_id': focus.get('track_id'), 'class': focus['class'],
                'confidence': round(focus['confidence'], 3), 'center': focus['center']}
    if tentative:
        focus = max(tentative, key=lambda d: d['confidence'])
        rect, zoom, how = target_rect(focus)
        return {'mode': 'tentative', 'rect': [int(v) for v in rect], 'zoom': zoom, 'framing': how,
                'reason': 'unconfirmed zoom detection', 'track_id': focus.get('track_id'),
                'class': focus['class'], 'confidence': round(focus['confidence'], 3), 'center': focus['center']}
    if crops:
        crop = crops[0]
        return {'mode': 'search', 'rect': [int(v) for v in crop['rect']], 'zoom': crop['zoom'],
                'framing': 'detector crop', 'reason': crop['reason'] + (f" {crop['tile']}" if crop.get('tile') else ''),
                'track_id': None, 'class': None, 'confidence': None, 'center': None}
    return {'mode': 'idle', 'rect': [0, 0, int(frame_w), int(frame_h)], 'zoom': 1.0, 'framing': 'full frame',
            'reason': 'waiting for target', 'track_id': None, 'class': None, 'confidence': None, 'center': None}


def render_zoom_view(image, focus, detections, tentative=None, frame_id=None, panel_w=640):
    """Upscaled view of focus['rect'] from `image` (the SAME frame the detections
    belong to), with boxes, IDs, confidences, zoom factor, crop coordinates and a
    minimap. Display interpolation only - it adds no information."""
    frame_h, frame_w = image.shape[:2]
    panel_h = int(max(240, min(720, round(panel_w * frame_h / max(frame_w, 1)))))
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    x1, y1, x2, y2 = focus['rect']
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        x1, y1, x2, y2 = 0, 0, frame_w, frame_h
    region = image[y1:y2, x1:x2]
    scale = min(panel_w / float(x2 - x1), (panel_h - 44) / float(y2 - y1))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    rw, rh = max(1, int((x2 - x1) * scale)), max(1, int((y2 - y1) * scale))
    resized = cv2.resize(region, (rw, rh), interpolation=interp)
    ox, oy = (panel_w - rw) // 2, 22 + (panel_h - 44 - rh) // 2
    canvas[oy:oy + rh, ox:ox + rw] = resized

    def tr(x, y):
        return int((x - x1) * scale + ox), int((y - y1) * scale + oy)

    for det in tentative or []:
        if det['x2'] < x1 or det['x1'] > x2 or det['y2'] < y1 or det['y1'] > y2:
            continue
        p1, p2 = tr(det['x1'], det['y1']), tr(det['x2'], det['y2'])
        cv2.rectangle(canvas, p1, p2, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"? {det['class']} {det['confidence']:.2f}", (p1[0], max(34, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    for det in detections:
        if det['x2'] < x1 or det['x1'] > x2 or det['y2'] < y1 or det['y1'] > y2:
            continue
        if focus.get('track_id') is not None:
            is_focus = det.get('track_id') == focus.get('track_id')
        else:
            is_focus = focus.get('center') is not None and det['center'] == focus.get('center')
        color = _class_color(det['class'])
        p1, p2 = tr(det['x1'], det['y1']), tr(det['x2'], det['y2'])
        cv2.rectangle(canvas, p1, p2, color, 3 if is_focus else 1, cv2.LINE_AA)
        if is_focus:
            pad = 8
            _corner_brackets(canvas, (p1[0] - pad, p1[1] - pad, p2[0] + pad, p2[1] + pad), (0, 0, 255), 18, 2)
        prefix = f"#{det['track_id']} " if det.get('track_id') is not None else ''
        label = f"{prefix}{det['class']} {det['confidence']:.2f}"
        if det.get('zoom_only'):
            label += ' [zoom]'
        if 'estimated_distance_m' in det:
            label += f" ~{det['estimated_distance_m']:.0f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(36, p1[1] - 6)
        lx = max(2, min(p1[0], panel_w - tw - 10))      # keep the label inside the view
        cv2.rectangle(canvas, (lx, ly - th - 6), (lx + tw + 6, ly + 2), (0, 0, 0), -1)
        cv2.putText(canvas, label, (lx + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        if is_focus:
            cv2.putText(canvas, 'TRIGGER', (max(2, min(p1[0], panel_w - 80)), min(panel_h - 26, p2[1] + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # header + footer bars
    cv2.rectangle(canvas, (0, 0), (panel_w, 21), (18, 22, 28), -1)
    cv2.rectangle(canvas, (0, panel_h - 22), (panel_w, panel_h), (18, 22, 28), -1)
    head = (f"AUTO-ZOOM {focus['zoom']:.1f}x  crop ({x1},{y1})-({x2},{y2})  view x{scale:.1f}"
            + (f"  frame #{frame_id}" if frame_id is not None else ''))
    cv2.putText(canvas, head, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 246), 1, cv2.LINE_AA)
    mode_colors = {'target': (0, 255, 0), 'tentative': (200, 200, 200), 'search': (0, 200, 255), 'idle': (150, 150, 150)}
    foot = {'target': f"TARGET  {focus['reason']}  ({focus.get('framing', '')})",
            'tentative': f"TENTATIVE  {focus['reason']}",
            'search': f"SEARCHING  {focus['reason']}",
            'idle': 'WAITING FOR TARGET'}[focus['mode']]
    cv2.putText(canvas, foot, (6, panel_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                mode_colors[focus['mode']], 1, cv2.LINE_AA)

    # minimap: where the crop sits in the full frame
    mm_w = 110
    mm_h = max(20, int(mm_w * frame_h / max(frame_w, 1)))
    mx, my = panel_w - mm_w - 8, panel_h - mm_h - 30
    cv2.rectangle(canvas, (mx, my), (mx + mm_w, my + mm_h), (40, 40, 40), -1)
    cv2.rectangle(canvas, (mx, my), (mx + mm_w, my + mm_h), (160, 160, 160), 1)
    sx = mm_w / float(frame_w)
    cv2.rectangle(canvas, (mx + int(x1 * sx), my + int(y1 * sx)), (mx + int(x2 * sx), my + int(y2 * sx)),
                  (0, 220, 255), 1)
    for det in detections:
        cv2.circle(canvas, (mx + int(det['center'][0] * sx), my + int(det['center'][1] * sx)), 2,
                   _class_color(det['class']), -1)
    return canvas


def _form_bool(name, default):
    value = request.form.get(name)
    if value is None:
        return default
    return value.lower() in ('1', 'true', 'yes', 'on')


def _b64(data):
    return base64.b64encode(data).decode('ascii') if data else None


def _camera_geometry(frame_w, frame_h):
    """Field of view for the 3D-style view. From the calibrated focal length when
    available, otherwise the configured nominal HFOV (then it is only illustrative)."""
    calibrated = range_focal_px > 0
    if calibrated:
        hfov = math.degrees(2 * math.atan(frame_w / (2.0 * range_focal_px)))
    else:
        hfov = camera_hfov_deg
    vfov = math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * frame_h / max(frame_w, 1)))
    return {'hfov_deg': round(hfov, 2), 'vfov_deg': round(vfov, 2), 'focal_calibrated': calibrated,
            'range_calibrated': range_focal_px > 0 and drone_real_width_m > 0}


# ============================================================
# PAGES
# ============================================================
@app.route('/')
def index():
    return render_template('index2.html', initial_tab='live')


@app.route('/templates/dashboard.css')
def dashboard_stylesheet():
    """Serve the dashboard stylesheet stored alongside the HTML templates."""
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'dashboard.css')

@app.route('/camera')
def camera_page():
    return render_template('index2.html', initial_tab='live')

@app.route('/video')
def video_page():
    return render_template('index2.html', initial_tab='video')


@app.route('/camera/legacy')
def camera_page_legacy():
    """Your previous camera.html (if it is still in templates/) keeps working here."""
    if os.path.exists(os.path.join(app.root_path, 'templates', 'camera.html')):
        return render_template('camera.html')
    return jsonify({'error': 'templates/camera.html not found'}), 404


@app.route('/video/legacy')
def video_page_legacy():
    if os.path.exists(os.path.join(app.root_path, 'templates', 'video.html')):
        return render_template('video.html')
    return jsonify({'error': 'templates/video.html not found'}), 404


# ============================================================
# IMAGE DETECTION
# ============================================================
@app.route('/detect', methods=['POST'])
def detect():
    try:
        confidence = float(request.form.get('confidence', 0.5))
        show_center_val = request.form.get('show_center', 'true') == 'true'
        show_crosshair_val = request.form.get('show_crosshair', 'true') == 'true'
        hide_edges = request.form.get('hide_edges', 'false').lower() == 'true'
        focal_length_px = max(0.0, float(request.form.get('focal_length_px', 0) or 0))
        target_width_m = max(0.0, float(request.form.get('target_width_m', 0) or 0))

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400

        h, w = image.shape[:2]
        rcfg = get_refine_config()
        rcfg['enabled'] = _form_bool('enhance', rcfg['enabled'])
        zcfg = get_zoom_config()
        zcfg['enabled'] = _form_bool('auto_zoom', zcfg['enabled'])

        t0 = time.perf_counter()
        refined, display_lut, refine_info = refine_frame_full(image, rcfg)
        display = display_frame(refined, display_lut)
        t1 = time.perf_counter()
        # Still image: no time pressure -> candidate crops + a full coarse tile scan.
        ms = detect_multiscale(refined, confidence, zcfg, exhaustive=True)
        t2 = time.perf_counter()
        detections = boxes_to_detections(ms['boxes'], w, h, hide_edges)
        add_range_estimates(detections, focal_length_px, target_width_m)
        annotated = draw_detections(display, detections, show_center_val, show_crosshair_val)
        draw_track_overlay(annotated, detections, None, ms['crops'], show_trails=False,
                           show_regions=_form_bool('show_zoom_regions', True))
        focus = select_zoom_focus(detections, [], ms['crops'], w, h, zcfg)
        zoom_view = render_zoom_view(display, focus, detections)
        t3 = time.perf_counter()

        _, buffer = cv2.imencode('.jpg', annotated)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        refined_base64 = None
        if refine_info['enabled']:
            _, rbuf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 90])
            refined_base64 = base64.b64encode(rbuf).decode('utf-8')

        target_crops = []
        for crop, det in build_target_crops(display, detections, max_targets=6, tile_size=256):
            _, cbuf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            target_crops.append({
                'image_base64': base64.b64encode(cbuf).decode('utf-8'),
                'class': det['class'], 'confidence': det['confidence'], 'center': det['center'],
                'source': det['source']
            })

        return jsonify({
            'success': True,
            'data': {
                'detections': detections,
                'total_detections': len(detections),
                'image_base64': image_base64,
                'image_size': {'width': image.shape[1], 'height': image.shape[0]},
                'refined_image_base64': refined_base64,
                'target_crops': target_crops,
                'zoom_view_base64': _b64(encode_jpeg(zoom_view, 90)),
                'zoom_focus': focus,
                'refinement': refine_info,
                'multiscale': {
                    'enabled': zcfg['enabled'],
                    'stats': ms['stats'],
                    'crops': [{'rect': c['rect'], 'zoom': c['zoom'], 'reason': c['reason'],
                               'detections': c.get('detections', 0), 'tile': c.get('tile')}
                              for c in ms['crops']],
                },
                'timing_ms': {
                    'refine': round((t1 - t0) * 1000, 2),
                    'inference': round(ms['timings']['original_ms'], 2),
                    'zoom_inference': round(ms['timings']['zoom_ms'], 2),
                    'detect_total': round((t2 - t1) * 1000, 2),
                    'annotate': round((t3 - t2) * 1000, 2),
                }
            },
            'metadata': {
                'model': f'YOLO11 ({os.path.basename(MODEL_PATH)})',
                'confidence_threshold': confidence,
                'enhancement_enabled': refine_info['enabled'],
                'auto_zoom_enabled': zcfg['enabled'],
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# VIDEO UPLOAD & PROCESSING
# ============================================================
MAX_VIDEO_SNAPSHOTS = 30


def process_video_job(job_id, input_path, output_path,
                      conf, show_center_flag, show_crosshair_flag,
                      skip_frames=1, hide_edges=False,
                      focal_length_px=0.0, target_width_m=0.0,
                      refine_cfg=None, zoom_cfg=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        with video_jobs_lock:
            video_jobs[job_id]['status'] = 'failed'
            video_jobs[job_id]['error'] = 'Cannot open video file'
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps / skip_frames if skip_frames > 1 else fps
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    if not writer.isOpened():
        cap.release()
        with video_jobs_lock:
            video_jobs[job_id]['status'] = 'failed'
            video_jobs[job_id]['error'] = 'Cannot open video writer'
        return

    job_model, job_lock = get_job_model()
    refine_cfg = refine_cfg if refine_cfg is not None else get_refine_config()
    zoom_cfg = dict(zoom_cfg if zoom_cfg is not None else get_zoom_config())
    zoom_cfg['adaptive_budget'] = False          # offline: no latency pressure, use the full crop budget

    frame_idx = 0
    total_detections = 0
    class_counts = {}
    last_annotated = None
    latest_detections = []
    tracker = DetectionTracker()
    scheduler = ZoomScheduler()
    t_refine = RollingStat(1000)
    t_infer = RollingStat(1000)
    t_zoom = RollingStat(1000)
    crops_total = 0
    zoom_only_total = 0
    snapshots = []
    last_snapshot_frame = -10 ** 9
    snapshot_every = max(1, int(round(fps)))   # at most one refined target snapshot per second

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                timestamp = frame_idx / fps
                t0 = time.perf_counter()
                refined, display_lut, _ = refine_frame_full(frame, refine_cfg)
                display = display_frame(refined, display_lut)
                t1 = time.perf_counter()
                ms = detect_multiscale(refined, conf, zoom_cfg, scheduler, tracker, timestamp,
                                       model_inst=job_model, lock=job_lock, motion_frame=frame)
                t_refine.add((t1 - t0) * 1000)
                t_infer.add(ms['timings']['original_ms'])
                t_zoom.add(ms['timings']['zoom_ms'])
                crops_total += ms['stats']['crops']
                all_dets = boxes_to_detections(ms['boxes'], width, height, hide_edges)
                tracker.update(all_dets, timestamp=timestamp,
                               confirm_hits=zoom_cfg['crop_confirm_hits'] if zoom_cfg['enabled'] else 1)
                add_range_estimates(all_dets, focal_length_px, target_width_m)
                detections = [d for d in all_dets if d.get('confirmed', True)]
                tentative = [d for d in all_dets if not d.get('confirmed', True)]
                zoom_only_total += sum(1 for d in detections if d.get('zoom_only'))
                annotated = draw_detections(display, detections, show_center_flag, show_crosshair_flag,
                                            tentative=tentative, copy=True)
                draw_track_overlay(annotated, detections, tracker.coasting(timestamp), ms['crops'],
                                   show_trails=True, show_regions=True)
                latest_detections = detections
                annotated = draw_scene_map_overlay(annotated, detections, inplace=True)
                total_detections += len(detections)
                for d in detections:
                    class_counts[d['class']] = class_counts.get(d['class'], 0) + 1
                last_annotated = annotated
                writer.write(annotated)

                if (detections and len(snapshots) < MAX_VIDEO_SNAPSHOTS
                        and frame_idx - last_snapshot_frame >= snapshot_every):
                    focus = select_zoom_focus(detections, tentative, ms['crops'], width, height, zoom_cfg)
                    view = render_zoom_view(display, focus, detections, tentative, frame_idx)
                    snap_name = f"{job_id}_{frame_idx:06d}.jpg"
                    cv2.imwrite(os.path.join(SNAPSHOTS_FOLDER, snap_name), view,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    snapshots.append({
                        'url': f'/static/snapshots/{snap_name}',
                        'frame_index': frame_idx,
                        'time_s': round(frame_idx / fps, 2),
                        'targets': len(detections),
                        'best_confidence': round(max(d['confidence'] for d in detections), 3),
                        'zoom': focus['zoom'],
                        'zoom_only_targets': sum(1 for d in detections if d.get('zoom_only')),
                    })
                    last_snapshot_frame = frame_idx
            else:
                writer.write(last_annotated if last_annotated is not None else frame)

            frame_idx += 1

            with video_jobs_lock:
                if total_frames > 0:
                    video_jobs[job_id]['progress'] = min(100, int(frame_idx / total_frames * 100))
                video_jobs[job_id]['frames_processed'] = frame_idx
                video_jobs[job_id]['detections_found'] = total_detections
                video_jobs[job_id]['latest_detections'] = latest_detections
                video_jobs[job_id]['frame_size'] = {'width': width, 'height': height}
                video_jobs[job_id]['range_calibrated'] = focal_length_px > 0 and target_width_m > 0
                video_jobs[job_id]['target_snapshots'] = list(snapshots)
                video_jobs[job_id]['zoom_stats'] = {'crops_total': crops_total,
                                                    'additional_zoom_detections': zoom_only_total}
                video_jobs[job_id]['timing_ms'] = {'refine': t_refine.summary(),
                                                   'inference': t_infer.summary(),
                                                   'zoom_inference': t_zoom.summary()}

    except Exception as e:
        with video_jobs_lock:
            video_jobs[job_id]['status'] = 'failed'
            video_jobs[job_id]['error'] = str(e)
        cap.release()
        writer.release()
        return

    cap.release()
    writer.release()

    with video_jobs_lock:
        video_jobs[job_id]['status'] = 'completed'
        video_jobs[job_id]['progress'] = 100
        video_jobs[job_id]['total_detections'] = total_detections
        video_jobs[job_id]['class_counts'] = class_counts
        video_jobs[job_id]['latest_detections'] = latest_detections
        video_jobs[job_id]['frame_size'] = {'width': width, 'height': height}
        video_jobs[job_id]['target_snapshots'] = list(snapshots)
        video_jobs[job_id]['zoom_stats'] = {'crops_total': crops_total,
                                            'additional_zoom_detections': zoom_only_total}
        video_jobs[job_id]['timing_ms'] = {'refine': t_refine.summary(),
                                           'inference': t_infer.summary(),
                                           'zoom_inference': t_zoom.summary()}
        video_jobs[job_id]['output_url'] = f'/static/processed/{os.path.basename(output_path)}'
        video_jobs[job_id]['finished_at'] = datetime.now().isoformat()


@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video provided'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        confidence = float(request.form.get('confidence', 0.5))
        show_center_flag = request.form.get('show_center', 'true') == 'true'
        show_crosshair_flag = request.form.get('show_crosshair', 'true') == 'true'
        skip_frames = max(1, int(request.form.get('skip_frames', 1)))
        hide_edges = request.form.get('hide_edges', 'false').lower() == 'true'
        focal_length_px = max(0.0, float(request.form.get('focal_length_px', 0) or 0))
        target_width_m = max(0.0, float(request.form.get('target_width_m', 0) or 0))
        refine_cfg = get_refine_config()
        refine_cfg['enabled'] = _form_bool('enhance', refine_cfg['enabled'])
        zoom_cfg = get_zoom_config()
        zoom_cfg['enabled'] = _form_bool('auto_zoom', zoom_cfg['enabled'])

        ext = os.path.splitext(file.filename)[1] or '.mp4'
        job_id = uuid.uuid4().hex[:12]
        input_name = f"input_{job_id}{ext}"
        input_path = os.path.join(UPLOADS_FOLDER, input_name)
        file.save(input_path)

        output_name = f"processed_{job_id}.mp4"
        output_path = os.path.join(PROCESSED_FOLDER, output_name)

        with video_jobs_lock:
            video_jobs[job_id] = {
                'job_id': job_id,
                'status': 'queued',
                'progress': 0,
                'frames_processed': 0,
                'detections_found': 0,
                'latest_detections': [],
                'frame_size': None,
                'range_calibrated': False,
                'total_detections': 0,
                'class_counts': {},
                'input_url': f'/static/uploads/{input_name}',
                'output_url': None,
                'created_at': datetime.now().isoformat(),
                'finished_at': None,
                'error': None,
                'enhancement_enabled': refine_cfg['enabled'],
                'auto_zoom_enabled': zoom_cfg['enabled'],
                'target_snapshots': [],
                'zoom_stats': None,
                'timing_ms': None,
            }

        t = threading.Thread(
            target=process_video_job,
            args=(job_id, input_path, output_path,
                  confidence, show_center_flag, show_crosshair_flag,
                  skip_frames, hide_edges, focal_length_px, target_width_m, refine_cfg, zoom_cfg),
            daemon=True
        )
        t.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Video uploaded and processing started'
        })

    except Exception as e:
        print(f"Upload video error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/video_status/<job_id>')
def video_status(job_id):
    with video_jobs_lock:
        job = video_jobs.get(job_id)
        job = dict(job) if job else None
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    return jsonify({'success': True, 'job': job})


@app.route('/video_jobs')
def list_video_jobs():
    with video_jobs_lock:
        jobs = [dict(j) for j in video_jobs.values()]
    jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify({'success': True, 'jobs': jobs})


# ============================================================
# LIVE CAMERA
# ============================================================
def _mjpeg_response(generator):
    resp = Response(generator, mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['X-Accel-Buffering'] = 'no'   # disable proxy buffering (nginx)
    return resp


@app.route('/camera_stream')
def camera_stream():
    return _mjpeg_response(generate_frames())


@app.route('/zoom_stream')
def zoom_stream():
    return _mjpeg_response(generate_frames(kind='zoom'))


def generate_frames(kind='feed'):
    """Streams the newest published packet. Never reads the camera, never runs YOLO:
    a slow viewer simply skips packets instead of slowing detection."""
    global stream_clients
    counted = kind == 'feed'
    if counted:
        with stream_clients_lock:
            stream_clients += 1
    try:
        last_id = -1
        pipeline = camera_pipeline
        while camera_active and pipeline is not None and pipeline is camera_pipeline and pipeline.running:
            packet = pipeline.wait_for_packet(last_id, timeout=1.0)
            if packet is None:
                continue
            last_id = packet.frame_id
            frame_bytes = packet.get_jpeg() if kind == 'feed' else packet.get_zoom_jpeg()
            if not frame_bytes:
                continue
            if counted:
                pipeline.note_delivered()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        if counted:
            with stream_clients_lock:
                stream_clients -= 1


@app.route('/live_frame')
def live_frame():
    """Dashboard endpoint: ONE response = annotated feed + zoom view + detections +
    crops + tracks + timings of the SAME frame. Long-polls until a frame newer
    than `after` exists, so a slow browser skips frames instead of queueing them."""
    pipeline = camera_pipeline
    if pipeline is None or not camera_active:
        return jsonify({'active': False, 'frame_id': None})
    try:
        after = int(request.args.get('after', -1))
        timeout = max(0.05, min(2.0, float(request.args.get('timeout', 1.0))))
    except ValueError:
        after, timeout = -1, 1.0
    if request.args.get('session') != pipeline.session_id:
        after = -1                      # camera restarted -> frame ids start again
    packet = pipeline.wait_for_packet(after, timeout=timeout)
    if packet is None:
        return jsonify({'active': camera_active, 'frame_id': None, 'session': pipeline.session_id})
    payload = packet.meta()
    h, w = packet.raw.shape[:2]
    payload.update({
        'active': True,
        'session': pipeline.session_id,
        'image': _b64(packet.get_jpeg()),
        'zoom_image': _b64(packet.get_zoom_jpeg()) if request.args.get('zoom', '1') != '0' else None,
        'camera': _camera_geometry(w, h),
        'fps': detection_stats.get('fps', 0),
        'total_detections': len(packet.detections),
        'performance': _performance_snapshot() if request.args.get('perf', '1') != '0' else None,
    })
    pipeline.note_delivered()
    return jsonify(payload)


def _stop_pipeline_locked():
    """Caller must hold camera_lock."""
    global camera, camera_active, camera_pipeline
    camera_active = False
    pipeline = camera_pipeline
    camera_pipeline = None
    camera = None
    if pipeline is not None:
        return pipeline.stop()
    return True


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, camera_active, camera_tracker, camera_pipeline, current_frame

    data = request.get_json(silent=True) or {}
    source = str(data.get('source', '0')).strip()

    if source.isdigit():
        source = int(source)

    try:
        width = int(data.get('width', 640))
        height = int(data.get('height', 480))
        fps = int(data.get('fps', 30))
    except (TypeError, ValueError):
        width, height, fps = 640, 480, 30

    with camera_lock:
        # Only one capture/processing pair per session.
        if camera_pipeline is not None and camera_pipeline.running:
            if camera_pipeline.source == source and not data.get('restart'):
                return jsonify({
                    'success': True,
                    'message': 'Camera already running',
                    'feed_url': '/camera_stream'
                })
            _stop_pipeline_locked()

        camera_tracker = DetectionTracker()
        current_frame = None
        with detection_stats_lock:
            detection_stats.update({'total_detections': 0, 'fps': 0, 'detections': []})

        pipeline = CameraPipeline(source, width, height, fps)
        pipeline.session_id = uuid.uuid4().hex[:8]
        ok, message = pipeline.start()
        if not ok:
            camera_active = False
            return jsonify({'success': False, 'message': message}), 400

        camera_pipeline = pipeline
        camera = pipeline
        camera_active = True

    return jsonify({
        'success': True,
        'message': 'Camera started successfully',
        'feed_url': '/camera_stream',
        'zoom_url': '/zoom_stream',
        'session': pipeline.session_id
    })


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    with camera_lock:
        clean = _stop_pipeline_locked()

    return jsonify({'success': True, 'message': 'Camera stopped', 'clean_shutdown': clean})


def _performance_snapshot():
    pipeline = camera_pipeline
    perf = pipeline.snapshot() if pipeline is not None else {'running': False}
    perf['enhancement_enabled'] = bool(get_refine_config()['enabled'])
    perf['auto_zoom_enabled'] = bool(get_zoom_config()['enabled'])
    perf['device'] = DEVICE_INFO['device']
    perf['gpu_name'] = DEVICE_INFO['gpu_name']
    perf['half'] = USE_HALF
    perf['imgsz'] = model_imgsz()
    perf['latency_note'] = ('frame_age_ms is application latency (frame decoded -> result published). '
                            'Camera exposure/encoding, network/RTSP buffering, decoding before cap.read() '
                            'and browser display are not included.')
    return perf


@app.route('/camera_status')
def camera_status():
    frame = current_frame
    frame_height, frame_width = frame.shape[:2] if frame is not None else (480, 640)
    with detection_stats_lock:
        stats = dict(detection_stats)
    return jsonify({
        'active': camera_active,
        'fps': stats.get('fps', 0),
        'total_detections': stats.get('total_detections', 0),
        'last_detection': stats.get('last_detection_time'),
        'detection_enabled': detection_enabled,
        'detections': stats.get('detections', []),
        'frame_size': {'width': frame_width, 'height': frame_height},
        'range_calibrated': range_focal_px > 0 and drone_real_width_m > 0,
        'frame_id': stats.get('frame_id'),
        'tentative_detections': stats.get('tentative_detections', 0),
        'enhancement_enabled': bool(get_refine_config()['enabled']),
        'auto_zoom_enabled': bool(get_zoom_config()['enabled']),
        'performance': _performance_snapshot(),
    })


@app.route('/set_detection_options', methods=['POST'])
def set_detection_options():
    global hide_edge_detections, range_focal_px, drone_real_width_m, detection_enabled
    data = request.get_json(silent=True) or {}
    hide_edge_detections = bool(data.get('hide_edges', hide_edge_detections))
    if 'detection_enabled' in data:
        detection_enabled = bool(data['detection_enabled'])
    try:
        range_focal_px = max(0.0, float(data.get('focal_length_px', range_focal_px) or 0))
        drone_real_width_m = max(0.0, float(data.get('target_width_m', drone_real_width_m) or 0))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Calibration values must be numbers'}), 400
    return jsonify({
        'success': True,
        'hide_edges': hide_edge_detections,
        'detection_enabled': detection_enabled,
        'range_calibrated': range_focal_px > 0 and drone_real_width_m > 0
    })


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global confidence_threshold

    data = request.get_json(silent=True)
    if data and 'confidence' in data:
        try:
            confidence_threshold = float(data['confidence'])
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Invalid request'}), 400
        return jsonify({'success': True, 'confidence': confidence_threshold})

    return jsonify({'success': False, 'error': 'Invalid request'}), 400


def _display_options():
    return {'show_center': show_center, 'show_crosshair': show_crosshair, 'base': stream_base,
            'show_zoom_regions': show_zoom_regions, 'show_trails': show_trails,
            'overlay_scene_map': overlay_scene_map, 'jpeg_quality': jpeg_quality,
            'camera_hfov_deg': camera_hfov_deg}


@app.route('/set_display_options', methods=['POST'])
def set_display_options():
    global show_center, show_crosshair, stream_base, show_zoom_regions, show_trails
    global overlay_scene_map, jpeg_quality, camera_hfov_deg

    data = request.get_json(silent=True)
    if data:
        if 'show_center' in data:
            show_center = data['show_center']
        if 'show_crosshair' in data:
            show_crosshair = data['show_crosshair']
        if data.get('base') in ('raw', 'enhanced'):
            stream_base = data['base']
        if 'show_zoom_regions' in data:
            show_zoom_regions = bool(data['show_zoom_regions'])
        if 'show_trails' in data:
            show_trails = bool(data['show_trails'])
        if 'overlay_scene_map' in data:
            overlay_scene_map = bool(data['overlay_scene_map'])
        try:
            if 'jpeg_quality' in data:
                jpeg_quality = max(30, min(95, int(data['jpeg_quality'])))
            if 'camera_hfov_deg' in data:
                camera_hfov_deg = max(5.0, min(170.0, float(data['camera_hfov_deg'])))
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Invalid number'}), 400
        return jsonify({'success': True, **_display_options()})

    return jsonify({'success': False, 'error': 'Invalid request'}), 400


@app.route('/display_options')
def display_options_get():
    return jsonify({'success': True, **_display_options(), 'confidence': confidence_threshold,
                    'detection_enabled': detection_enabled, 'hide_edges': hide_edge_detections,
                    'focal_length_px': range_focal_px, 'target_width_m': drone_real_width_m})


@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    pipeline = camera_pipeline
    packet = pipeline.get_latest_packet() if pipeline is not None else None
    frame = packet.annotated if packet is not None else current_frame

    if frame is None:
        return jsonify({'success': False, 'error': 'No frame available'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(CAPTURES_FOLDER, filename)
    cv2.imwrite(filepath, frame)
    zoom_url = None
    if packet is not None:
        zoom_name = f"capture_{timestamp}_zoom.jpg"
        with open(os.path.join(CAPTURES_FOLDER, zoom_name), 'wb') as fh:
            fh.write(packet.get_zoom_jpeg())
        zoom_url = f'/static/captures/{zoom_name}'

    return jsonify({
        'success': True,
        'message': 'Frame captured successfully',
        'filename': filename,
        'url': f'/static/captures/{filename}',
        'zoom_url': zoom_url,
        'detections': len(packet.detections) if packet is not None else detection_stats.get('total_detections', 0),
        'frame_id': packet.frame_id if packet is not None else None
    })


@app.route('/get_stats')
def get_stats():
    with detection_stats_lock:
        stats = dict(detection_stats)
    stats['performance'] = _performance_snapshot()
    return jsonify(stats)


@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    pipeline = camera_pipeline
    if pipeline is not None:
        pipeline.reset_stats()
    return jsonify({'success': True})


# ---------------- refinement + zoom control ----------------
@app.route('/refinement_config', methods=['GET'])
def refinement_config_get():
    return jsonify({'success': True, 'config': get_refine_config(), 'defaults': REFINE_DEFAULTS})


@app.route('/set_refinement', methods=['POST'])
def set_refinement():
    data = request.get_json(silent=True) or {}
    if data.get('reset'):
        data = dict(REFINE_DEFAULTS)
    cfg, error = update_refine_config(data)
    if error:
        return jsonify({'success': False, 'error': error}), 400
    return jsonify({'success': True, 'config': cfg})


@app.route('/zoom_config', methods=['GET'])
def zoom_config_get():
    return jsonify({'success': True, 'config': get_zoom_config(), 'defaults': ZOOM_DEFAULTS})


@app.route('/set_zoom_options', methods=['POST'])
def set_zoom_options():
    data = request.get_json(silent=True) or {}
    if data.get('reset'):
        data = json.loads(json.dumps(ZOOM_DEFAULTS))
    cfg, error = update_zoom_config(data)
    if error:
        return jsonify({'success': False, 'error': error}), 400
    pipeline = camera_pipeline
    if pipeline is not None:
        pipeline.scheduler.budget = None        # re-seed the adaptive budget from the new cap
    return jsonify({'success': True, 'config': cfg})


def _jpeg_response(image, packet=None, quality=85):
    data = encode_jpeg(image, quality)
    if data is None:
        return jsonify({'success': False, 'error': 'Encoding failed'}), 500
    resp = Response(data, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store'
    if packet is not None:
        resp.headers['X-Frame-Id'] = str(packet.frame_id)
        resp.headers['X-Frame-Age-Ms'] = f"{(time.perf_counter() - packet.capture_perf) * 1000:.0f}"
        resp.headers['Access-Control-Expose-Headers'] = 'X-Frame-Id, X-Frame-Age-Ms'
    return resp


@app.route('/frame_view/<kind>')
def frame_view(kind):
    """Latest frame as JPEG: raw | refined (YOLO input) | display | annotated | zoom - all the same frame_id."""
    pipeline = camera_pipeline
    packet = pipeline.get_latest_packet() if pipeline is not None else None
    if packet is None:
        return ('', 204)
    if kind == 'zoom':
        resp = Response(packet.get_zoom_jpeg(), mimetype='image/jpeg')
        resp.headers['Cache-Control'] = 'no-store'
        resp.headers['X-Frame-Id'] = str(packet.frame_id)
        return resp
    image = {'raw': packet.raw, 'refined': packet.refined, 'display': packet.display,
             'annotated': packet.annotated}.get(kind)
    if image is None:
        return jsonify({'success': False, 'error': 'kind must be raw|refined|display|annotated|zoom'}), 400
    return _jpeg_response(image, packet)


@app.route('/target_snapshot')
def target_snapshot():
    """Zoomed, refined crops of the targets from the most recent frame that had detections."""
    pipeline = camera_pipeline
    packet = pipeline.get_latest_packet(with_detections=True) if pipeline is not None else None
    if packet is None:
        return ('', 204)
    source = request.args.get('source', 'refined')
    image = {'raw': packet.raw, 'detector': packet.refined}.get(source, packet.display)
    mosaic = build_target_mosaic(image, packet.detections,
                                 max_targets=int(request.args.get('max', 4)),
                                 tile_size=int(request.args.get('size', 224)))
    if mosaic is None:
        return ('', 204)
    return _jpeg_response(mosaic, packet, quality=90)


@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'model_names': list(model.names.values()) if model else [],
        'model_path': MODEL_PATH,
        'camera_active': camera_active,
        'device': DEVICE_INFO['device'],
        'cuda_available': DEVICE_INFO['cuda_available'],
        'gpu_name': DEVICE_INFO['gpu_name'],
        'gpu_memory_gb': DEVICE_INFO['gpu_memory_gb'],
        'torch_version': DEVICE_INFO['torch_version'],
        'half': USE_HALF,
        'imgsz': model_imgsz(),
        'enhancement_enabled': bool(get_refine_config()['enabled']),
        'auto_zoom': get_zoom_config(),
        'separate_video_model': SEPARATE_VIDEO_MODEL,
    })


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@atexit.register
def _shutdown():
    try:
        with camera_lock:
            _stop_pipeline_locked()
    except Exception:
        pass


if __name__ == '__main__':
    # debug=True starts the reloader, which loads the model (and GPU memory) twice
    # and restarts threads on every save. Use FLASK_DEBUG=1 only while editing code.
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)),
            debug=debug, threaded=True, use_reloader=False)
