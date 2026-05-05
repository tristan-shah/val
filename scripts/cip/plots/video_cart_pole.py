"""
Renders rendering-only videos for cart pole seeds and combines them
into a 4x2 grid (1920x1240) matching the style of the other combined videos.

Usage (from repo root):
    python scripts/cip/plots/video_cart_pole.py

Seeds used  : 0-7  (8 seeds → 4 columns × 2 rows)
Path pattern: results/CIP/CART_POLE/ol/
              seed={s}-beta=0.0-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200
Output      : results/CIP/CART_POLE/ol/combined_seeds_white_1920x1240.mp4
"""

import os
# MUJOCO_GL is not forced here — set it in the shell if needed
# (on Linux/GPU: export MUJOCO_GL=egl; on macOS: leave unset or use glfw)

import subprocess
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from val import Dynamics

# ── configuration ────────────────────────────────────────────────────────────

ROOT        = Path('results/CIP/CART_POLE/ol')
PAT         = 'seed={s}-beta=0.0-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200'
SEEDS       = list(range(8))           # 0..7 → 4×2 grid
WHITE_XML   = 'xml/cart_pole_white.xml'
RENDER_VID  = 'rendering_vid.mp4'
OUT_VID     = ROOT / 'combined_seeds_white_1920x1240.mp4'

# render params — mirrors original cart_pole.py experiment
DT          = 0.01
SKIP        = 2
DISTANCE    = 4.0
LOOKAT      = jnp.array([0.0, 0.0, 1.0])   # default, frames cart level nicely

# ffmpeg grid params (matches humanoid / double pendulum combined videos)
TILE_W, TILE_H = 480, 620
CROP_MARGIN    = 60                    # px — applied after full-frame scan
VID_W, VID_H   = 1920, 1088

# ── step 1: render rendering_vid.mp4 for each seed ───────────────────────────

print('Loading white-background dynamics ...')
dyn = Dynamics(WHITE_XML, dt=DT)

for s in SEEDS:
    name    = PAT.format(s=s)
    path    = ROOT / name
    outfile = path / RENDER_VID

    if outfile.exists():
        print(f'  seed={s}: {RENDER_VID} already exists, skipping render.')
        continue

    traj_path = path / 'traj.npy'
    if not traj_path.exists():
        print(f'  seed={s}: traj.npy not found, skipping.')
        continue

    print(f'  seed={s}: rendering ...')
    X = jnp.load(traj_path)
    dyn.render(X, path=outfile, skip=SKIP, distance=DISTANCE, lookat=LOOKAT)
    print(f'  seed={s}: saved to {outfile}')

# ── step 2: detect crop bounds — scan EVERY frame via ffmpeg pipe ─────────────

print('\nDetecting crop bounds (scanning all frames) ...')

xmin_all, xmax_all = VID_W, 0
ymin_all, ymax_all = VID_H, 0
frame_bytes = VID_W * VID_H * 3

def scan_video(path):
    """Return (xmin, xmax, ymin, ymax) of non-white pixels across all frames."""
    proc = subprocess.Popen(
        ['ffmpeg', '-i', str(path),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    xmin, xmax, ymin, ymax = VID_W, 0, VID_H, 0
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(VID_H, VID_W, 3)
        mask  = np.any(frame < 240, axis=2)
        rows  = np.any(mask, axis=1)
        cols  = np.any(mask, axis=0)
        if rows.any():
            ymin = min(ymin, int(np.argmax(rows)))
            ymax = max(ymax, int(len(rows) - 1 - np.argmax(rows[::-1])))
            xmin = min(xmin, int(np.argmax(cols)))
            xmax = max(xmax, int(len(cols) - 1 - np.argmax(cols[::-1])))
    proc.wait()
    return xmin, xmax, ymin, ymax

for s in SEEDS:
    vid = ROOT / PAT.format(s=s) / RENDER_VID
    if not vid.exists():
        print(f'  seed={s}: missing {RENDER_VID}, skipping.')
        continue
    x0, x1, y0, y1 = scan_video(vid)
    xmin_all = min(xmin_all, x0); xmax_all = max(xmax_all, x1)
    ymin_all = min(ymin_all, y0); ymax_all = max(ymax_all, y1)
    print(f'  seed={s}: x=[{x0}, {x1}], y=[{y0}, {y1}]')

# apply margin and enforce even dimensions
cx0 = max(0,      xmin_all - CROP_MARGIN)
cx1 = min(VID_W,  xmax_all + CROP_MARGIN)
cy0 = max(0,      ymin_all - CROP_MARGIN)
cy1 = min(VID_H,  ymax_all + CROP_MARGIN)
cw  = cx1 - cx0;  cw += cw % 2
ch  = cy1 - cy0;  ch += ch % 2
CROP = f'crop={cw}:{ch}:{cx0}:{cy0}'
print(f'\n  Union bounds : x=[{xmin_all}, {xmax_all}], y=[{ymin_all}, {ymax_all}]')
print(f'  Crop filter  : {CROP}  (margin={CROP_MARGIN}px)')

# ── step 3: combine into 4×2 grid via ffmpeg ─────────────────────────────────

print('\nCombining into 4×2 grid ...')

VF = (f'{CROP},'
      f'scale={TILE_W}:{TILE_H}:force_original_aspect_ratio=decrease,'
      f'pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2:white')

inputs = []
for s in SEEDS:
    inputs += ['-i', str(ROOT / PAT.format(s=s) / RENDER_VID)]

n      = len(SEEDS)
chains = ''.join(f'[{i}:v]{VF}[v{i}];' for i in range(n))
layout = '|'.join(f'{(i % 4) * TILE_W}_{(i // 4) * TILE_H}' for i in range(n))
vstack = ''.join(f'[v{i}]' for i in range(n))
filter_complex = f'{chains}{vstack}xstack=inputs={n}:layout={layout}[out]'

cmd = (
    ['ffmpeg']
    + inputs
    + ['-filter_complex', filter_complex,
       '-map', '[out]',
       '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
       '-y', str(OUT_VID)]
)

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print(f'\nDone! Output: {OUT_VID}  ({TILE_W*4}x{TILE_H*2})')
else:
    print('ffmpeg error:')
    print(result.stderr[-2000:])
