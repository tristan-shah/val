"""
Figure 1: freeze-frame trajectory of the gibbon (hanging humanoid) self-righting under CIP.

Builds a single MuJoCo scene containing N copies of the robot model, each
placed at a different x-position and frozen at a selected trajectory pose.
One render → one image, no compositing seams.

Frames are sampled uniformly in cumulative pose-space arc-length so the
dynamic swing-up phase is represented with fine granularity.

Writes figures/humulum_trajectory.{pdf,png}. Run from the repository root:
    python scripts/cip/plots/plot_humulum_trajectory.py
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

## runs behind the paper's figures (committed); point this at 'results' to plot new runs
RESULTS = Path('results/paper')

TRAJ_PATH = (
    RESULTS / 'CIP/HUMULUM'
    / 'seed=0-beta=9.0-h=512-shots=1024-iter=1-elite=0.2-smooth=0.1-rho=0.9-dt=0.01-steps=1200'
    / 'traj.npy'
)
XML_PATH = 'xml/humulum.xml'
OUTPUT   = Path('figures/humulum_trajectory.pdf')

N_FRAMES  = 20     # number of poses
ARC_FRAC  = 0.80   # fraction of arc-length to draw from
X_SPACING = 0.65   # metres between successive robot origins

RENDER_W  = 3200
RENDER_H  = 700

AZIMUTH   = 90.0   # side-on view
ELEVATION = -8.0
LOOKAT_Z  = 0.10   # vertical centre of interest (lower to keep hanging feet in frame)

# Orthographic-like camera: large-ish distance + small fovy.
# 30 m gives <2% size variation edge-to-centre while keeping the depth
# buffer well-conditioned (300 m caused z-fighting artifacts).
ORTHO_DIST = 30.0

NQ = 13   # qpos dimension per robot instance


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

def _rename_subtree(elem: ET.Element, prefix: str) -> ET.Element:
    """Recursively copy an XML subtree, prefixing every name/joint attribute."""
    elem = copy.deepcopy(elem)
    _rename_inplace(elem, prefix)
    return elem


def _rename_inplace(elem: ET.Element, prefix: str) -> None:
    for attr in ('name', 'joint', 'body1', 'body2', 'site', 'tendon'):
        if attr in elem.attrib:
            elem.attrib[attr] = prefix + elem.attrib[attr]
    for child in elem:
        _rename_inplace(child, prefix)


def build_scene_xml(xml_path: str, X: np.ndarray, indices: np.ndarray) -> str:
    """
    Return an XML string for a single scene with len(indices) robot copies,
    each at a different x-position and frozen at the corresponding trajectory pose.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    orig_worldbody = root.find('worldbody')

    # Identify the anchor bodies and the pelvis subtree in the original XML.
    # Layout in humulum.xml:  left_anchor, right_anchor, pelvis  (in worldbody)
    orig_l_anchor = orig_worldbody.find("body[@name='left_anchor']")
    orig_r_anchor = orig_worldbody.find("body[@name='right_anchor']")
    orig_pelvis   = orig_worldbody.find("body[@name='pelvis']")

    orig_equality = root.find('equality')

    # Build new XML preserving compiler / option / default / asset
    new_root = ET.Element('mujoco', model='humulum_trajectory')
    for tag in ('compiler', 'option', 'default', 'asset'):
        elem = root.find(tag)
        if elem is not None:
            new_root.append(copy.deepcopy(elem))

    # Ensure the offscreen framebuffer is wide enough for RENDER_W
    visual = ET.SubElement(new_root, 'visual')
    global_elem = ET.SubElement(visual, 'global')
    global_elem.set('offwidth',  str(RENDER_W))
    global_elem.set('offheight', str(RENDER_H))

    new_wb  = ET.SubElement(new_root, 'worldbody')
    new_eq  = ET.SubElement(new_root, 'equality')

    # Single directional light (same as original)
    light = orig_worldbody.find('light')
    if light is not None:
        new_wb.append(copy.deepcopy(light))

    for i, frame_idx in enumerate(indices):
        qpos = X[frame_idx, :NQ]
        # root_x and root_z are qpos[0], qpos[1]; remaining are joint angles
        x_root  = float(qpos[0]) + i * X_SPACING
        z_root  = float(qpos[1])
        joint_q = qpos[2:]   # waist, l_shoulder, l_elbow, r_shoulder, r_elbow,
                             # l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle

        p = f'r{i}_'   # prefix for all names in this instance

        # -- Anchor bodies (fixed in world, rendered invisible) --
        l_anc = _rename_subtree(orig_l_anchor, p)
        r_anc = _rename_subtree(orig_r_anchor, p)
        for anc in (l_anc, r_anc):
            for g in anc.iter('geom'):
                g.set('rgba', '0 0 0 0')
        # Shift anchors by the same x_root offset so weld constraints work
        orig_la_pos = [float(v) for v in orig_l_anchor.get('pos').split()]
        orig_ra_pos = [float(v) for v in orig_r_anchor.get('pos').split()]
        l_anc.set('pos', f'{orig_la_pos[0] + i * X_SPACING} {orig_la_pos[1]} {orig_la_pos[2]}')
        r_anc.set('pos', f'{orig_ra_pos[0] + i * X_SPACING} {orig_ra_pos[1]} {orig_ra_pos[2]}')
        new_wb.append(l_anc)
        new_wb.append(r_anc)

        # -- Pelvis subtree (remove slide joints, fix position) --
        pelvis = _rename_subtree(orig_pelvis, p)
        orig_ppos = [float(v) for v in orig_pelvis.get('pos').split()]
        pelvis.set('pos', f'{orig_ppos[0] + x_root} {orig_ppos[1]} {orig_ppos[2] + z_root}')
        # Remove the two root slide joints (root_x, root_z) so pelvis is fixed
        for jname in (p + 'root_x', p + 'root_z'):
            for j in pelvis.findall('joint'):
                if j.get('name') == jname:
                    pelvis.remove(j)
        new_wb.append(pelvis)

        # -- Equality weld constraints for this instance --
        for weld in orig_equality.findall('weld'):
            w = copy.deepcopy(weld)
            w.set('body1', p + weld.get('body1'))
            w.set('body2', p + weld.get('body2'))
            new_eq.append(w)

    return ET.tostring(new_root, encoding='unicode')


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_scene(xml_string: str, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    model = mujoco.MjModel.from_xml_string(xml_string)
    data  = mujoco.MjData(model)

    # Set joint angles for every instance.
    # After removing root_x/root_z each instance has NQ-2 = 11 dof.
    n_joints_per = NQ - 2
    for i, frame_idx in enumerate(indices):
        joint_q = X[frame_idx, 2:NQ]
        offset  = i * n_joints_per
        data.qpos[offset: offset + n_joints_per] = joint_q

    mujoco.mj_forward(model, data)

    # --- Orthographic-like camera ---
    # Compute fovy so the full row of robots fills the frame with margin.
    total_x   = (len(indices) - 1) * X_SPACING
    mid_x     = total_x / 2.0
    half_w    = total_x / 2.0 * 1.14   # 14% horizontal margin each side
    aspect    = RENDER_W / RENDER_H
    # tan(fovx/2) = half_w / dist;  fovy derived from fovx via aspect ratio
    fovx_half = np.arctan(half_w / ORTHO_DIST)
    fovy_deg  = np.degrees(2.0 * fovx_half / aspect) * 1.20  # 20% extra vertical

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
    xml_string = build_scene_xml(XML_PATH, X, indices)

    print('Rendering …')
    frame = render_scene(xml_string, X, indices)

    img = Image.fromarray(frame)

    # Crop whitespace: find bounding box of non-white pixels with small padding
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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(OUTPUT))
    png_path = OUTPUT.with_suffix('.png')
    img.save(str(png_path), dpi=(300, 300))
    print(f'Saved → {OUTPUT}')
    print(f'Saved → {png_path}')
