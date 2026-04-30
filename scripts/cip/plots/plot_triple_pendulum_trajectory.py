"""
Freeze-frame trajectory figure for the triple pendulum environment.

Builds a single MuJoCo scene containing N copies of the pendulum model, each
placed at a different x-position and frozen at a selected trajectory pose.
One render → one image, no compositing seams.

Frames are sampled uniformly in cumulative pose-space arc-length so dynamic
regions are represented with fine granularity.

Usage (from repo root):
    conda run -n val python scripts/cip/plots/plot_triple_pendulum_trajectory.py
"""

from pathlib import Path
import copy
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAJ_PATH = (
    Path('results/CIP/TRIPLE_PENDULUM/ol')
    / 'retest-seed=0-gear=25.0-beta=2.5-h=128-shots=2048-iter=2-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200'
    / 'traj.npy'
)
XML_PATH = 'xml/triple_pendulum_white.xml'
OUTPUT   = Path('scripts/cip/plots/triple_pendulum_trajectory.pdf')

N_FRAMES  = 20     # number of poses
ARC_FRAC  = 0.80   # fraction of arc-length to draw from
X_SPACING = 0.55   # metres between successive pendulum origins

RENDER_W  = 3200
RENDER_H  = 700

AZIMUTH   = 90.0   # side-on view
ELEVATION = -8.0
LOOKAT_Z  = 1.50   # pivot height — symmetric between hanging (z≈0.15) and upright (z≈2.85)

# Orthographic-like camera: large-ish distance + small fovy.
ORTHO_DIST = 30.0

NQ = 3   # qpos dimension per pendulum (foot_joint, leg_joint, thigh_joint)


# ---------------------------------------------------------------------------
# Arc-length frame sampling
# ---------------------------------------------------------------------------

def arc_length_indices(X: np.ndarray, n: int, arc_frac: float) -> np.ndarray:
    qpos = X[:, :NQ]
    diffs = np.linalg.norm(np.diff(qpos, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(diffs)])
    target_end = cumlen[-1] * arc_frac
    targets = np.linspace(0.0, target_end, n)
    return np.array([np.searchsorted(cumlen, t, side='left') for t in targets])


# ---------------------------------------------------------------------------
# Multi-pose XML builder
# ---------------------------------------------------------------------------

def _rename_inplace(elem: ET.Element, prefix: str) -> None:
    for attr in ('name', 'joint', 'body1', 'body2', 'site', 'tendon'):
        if attr in elem.attrib:
            elem.attrib[attr] = prefix + elem.attrib[attr]
    for child in elem:
        _rename_inplace(child, prefix)


def build_scene_xml(xml_path: str, n_instances: int) -> str:
    """Return an XML string for a scene with n_instances pendulum copies."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    orig_worldbody = root.find('worldbody')
    orig_foot = orig_worldbody.find("body[@name='foot']")
    orig_foot_pos = [float(v) for v in orig_foot.get('pos').split()]

    new_root = ET.Element('mujoco', model='triple_pendulum_trajectory')
    for tag in ('compiler', 'option', 'default', 'asset'):
        elem = root.find(tag)
        if elem is not None:
            new_root.append(copy.deepcopy(elem))

    visual = ET.SubElement(new_root, 'visual')
    global_elem = ET.SubElement(visual, 'global')
    global_elem.set('offwidth',  str(RENDER_W))
    global_elem.set('offheight', str(RENDER_H))

    new_wb = ET.SubElement(new_root, 'worldbody')

    light = orig_worldbody.find('light')
    if light is not None:
        new_wb.append(copy.deepcopy(light))

    # Add a single floor plane
    floor = orig_worldbody.find("geom[@name='floor']")
    if floor is not None:
        new_wb.append(copy.deepcopy(floor))

    for i in range(n_instances):
        p = f'r{i}_'
        foot = copy.deepcopy(orig_foot)
        _rename_inplace(foot, p)
        foot.set('pos', (
            f'{orig_foot_pos[0] + i * X_SPACING} '
            f'{orig_foot_pos[1]} '
            f'{orig_foot_pos[2]}'
        ))
        new_wb.append(foot)

    return ET.tostring(new_root, encoding='unicode')


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_scene(xml_string: str, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    model = mujoco.MjModel.from_xml_string(xml_string)
    data  = mujoco.MjData(model)

    for i, frame_idx in enumerate(indices):
        offset = i * NQ
        data.qpos[offset: offset + NQ] = X[frame_idx, :NQ]

    mujoco.mj_forward(model, data)

    total_x = (len(indices) - 1) * X_SPACING
    mid_x   = total_x / 2.0
    half_w  = total_x / 2.0 * 1.14
    aspect  = RENDER_W / RENDER_H
    fovx_half = np.arctan(half_w / ORTHO_DIST)
    fovy_deg  = np.degrees(2.0 * fovx_half / aspect) * 1.41  # match humulum physical scale

    model.vis.global_.fovy = fovy_deg

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [mid_x, 0.0, LOOKAT_Z]
    camera.distance  = ORTHO_DIST
    camera.azimuth   = AZIMUTH
    camera.elevation = ELEVATION

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    renderer.update_scene(data, camera=camera)
    frame = renderer.render()
    renderer.close()
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading trajectory …')
    X = np.load(TRAJ_PATH)
    print(f'  shape: {X.shape}')

    indices = arc_length_indices(X, N_FRAMES, ARC_FRAC)
    print(f'  frame indices: {indices}')
    print(f'  times:         {np.round(indices * 0.01, 2)} s')

    print('Building multi-pose scene XML …')
    xml_string = build_scene_xml(XML_PATH, len(indices))

    print('Rendering …')
    frame = render_scene(xml_string, X, indices)

    img = Image.fromarray(frame)

    # Crop whitespace with small padding
    arr = np.array(img)
    mask = ~((arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245))
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    pad = 20
    r0 = max(np.argmax(rows) - pad, 0)
    r1 = min(len(rows) - np.argmax(rows[::-1]) + pad, arr.shape[0])
    c0 = max(np.argmax(cols) - pad, 0)
    c1 = min(len(cols) - np.argmax(cols[::-1]) + pad, arr.shape[1])
    img = img.crop((c0, r0, c1, r1))

    # Match humulum output size so images stack cleanly
    TARGET_SIZE = (2518, 627)
    img = img.resize(TARGET_SIZE, Image.LANCZOS)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(OUTPUT))
    png_path = OUTPUT.with_suffix('.png')
    img.save(str(png_path), dpi=(300, 300))
    print(f'Saved → {OUTPUT}')
    print(f'Saved → {png_path}')
