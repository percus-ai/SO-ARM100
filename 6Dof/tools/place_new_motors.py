#!/usr/bin/env python3
"""Compute the ideal poses of the J4/J5 STS3215 motors in their link frames and write
pre-transformed meshes for Fusion (build/meshes/sts3215_03a_v1__{lower_arm,forearm}.stl, metres).

The motors were placed by hand in Fusion; this keeps their position along the joint axis and
their rotation about it, but puts the output shaft exactly on the axis it drives:
  J4 motor -> forearm_roll axis (in lower_arm_link),  J5 motor -> wrist_flex axis through the
  centre plane of the J4-J5 cup (in forearm_link). Also reports the world shift the wrist chain
needs so that wrist_flex lies in that centre plane.
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshprops import read_stl, write_stl, transform
TOOLS = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(TOOLS, '..', '..'))
DUMP = json.load(open(os.path.join(TOOLS, 'build', 'so111_fusion_dump.json')))
SPEC = json.load(open(os.path.join(TOOLS, 'build', 'so101_assembly_spec.json')))
SHAFT_PT = np.array([0.0125, 0.0, 0.0187])   # horn face centre in the sts3215 mesh frame (m)
SHAFT_DIR = np.array([0.0, 0.0, 1.0])

def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], [-sp, cp*sr, cp*cr]])
def fusion_T(a):
    T = np.array(a, float).reshape(4, 4); T[:3, 3] /= 100.0; return T

T_world = {n: fusion_T(e['T_world']) for n, e in DUMP['links'].items()}
T_world['lower_arm_link'] = fusion_T([l for l in SPEC['links'] if l['name'] == 'lower_arm_link'][0]['T_world'])
pre = [p for l in SPEC['links'] if l['name'] == 'wrist_link' for p in l['parts'] if p['kind'] == 'mesh'][0]
T_pre = fusion_T(pre['T_local'])

def current_motor_pose(link):
    ch = [c for c in DUMP['links'][link]['children'] if c['component'].startswith('sts3215')][0]
    Tc = np.linalg.inv(T_world[link]) @ fusion_T(ch['T_world']) @ T_pre
    Rs = np.round(Tc[:3, :3]); assert np.abs(Rs - Tc[:3, :3]).max() < 0.02
    Tc[:3, :3] = Rs
    return Tc, ch['occ']

def snap_to_line(Tc, a, u):
    """Translate pose Tc so that its shaft line coincides with the line (a, u)."""
    u = u / np.linalg.norm(u)
    d = Tc[:3, :3] @ SHAFT_DIR
    assert abs(abs(np.dot(d, u)) - 1) < 1e-9, 'shaft not parallel to the target axis'
    p = Tc[:3, :3] @ SHAFT_PT + Tc[:3, 3]
    e = (p - a) - np.dot(p - a, u) * u
    Tc = Tc.copy(); Tc[:3, 3] -= e
    return Tc, np.linalg.norm(e)

# joint frames (child in parent) from the dump
def child_in_parent(parent, child):
    return np.linalg.inv(T_world[parent]) @ T_world[child]

out = {}
# J4 motor: shaft on the forearm_roll axis (forearm_link origin / z in lower_arm frame)
Tfr = child_in_parent('lower_arm_link', 'forearm_link')
Tc, occ4 = current_motor_pose('lower_arm_link')
T4, e4 = snap_to_line(Tc, Tfr[:3, 3], Tfr[:3, 2])
print('J4 motor: removed %.3f mm perpendicular offset; pose xyz=%s' % (e4 * 1000, np.round(T4[:3, 3], 6)))
# J5 motor: shaft on the wrist_flex axis moved into the cup centre plane (y = 0 in forearm frame)
Twf = child_in_parent('forearm_link', 'wrist_link')
b = Twf[:3, 3].copy(); wrist_dy = -b[1]; b[1] = 0.0
Tc, occ5 = current_motor_pose('forearm_link')
T5, e5 = snap_to_line(Tc, b, Twf[:3, 2])
print('J5 motor: removed %.3f mm perpendicular offset; pose xyz=%s' % (e5 * 1000, np.round(T5[:3, 3], 6)))
shift_world_cm = (T_world['forearm_link'][:3, :3] @ np.array([0.0, wrist_dy, 0.0])) * 100.0
print('wrist chain world shift (mm): %s' % np.round(shift_world_cm * 10, 4))

motor = read_stl(os.path.join(REPO, 'Simulation', 'SO101', 'assets', 'sts3215_03a_v1.stl'))
os.makedirs(os.path.join(TOOLS, 'build', 'meshes'), exist_ok=True)
for link, T, occ in (('lower_arm_link', T4, occ4), ('forearm_link', T5, occ5)):
    name = 'sts3215_03a_v1__' + link.replace('_link', '')
    path = os.path.join(TOOLS, 'build', 'meshes', name + '.stl')
    write_stl(path, transform(motor, T))
    out[link] = {'component': name, 'mesh_path': path, 'T_link_motor': T.tolist(), 'replaces_occ': occ}
out['wrist_chain_shift_world_cm'] = shift_world_cm.tolist()
json.dump(out, open(os.path.join(TOOLS, 'build', 'so111_motor_poses.json'), 'w'), indent=1)
print('wrote build/so111_motor_poses.json')
