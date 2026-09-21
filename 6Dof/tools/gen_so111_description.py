#!/usr/bin/env python3
"""Generate the SO111 (6-DoF SO101) URDF and MJCF from the Fusion assembly dump.

Inputs (all under 6Dof/tools/build):
  so111_fusion_dump.json   link/joint transforms exported from the Fusion document "SO111-Assembly"
  so101_assembly_spec.json part placements of the stock SO101 links (from gen_spec.py)
  ../../Simulation/SO111/assets/so111_*.stl  new printed parts exported from Fusion (mm, part frame)

Unchanged links (base, shoulder, upper_arm, wrist, gripper, moving_jaw) reuse the inertial and
mesh definitions of Simulation/SO101/so101_new_calib.urdf. The two new links (lower_arm_link,
forearm_link) get mesh placements from Fusion and inertials computed from the meshes with an
effective printed-part density and a lumped motor mass regressed from the SO101 URDF.
"""
import json, math, os, shutil, sys
import numpy as np
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshprops import read_stl, write_stl, transform, mass_properties

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TOOLS, '..', '..'))
SRC_SIM = os.path.join(REPO, 'Simulation', 'SO101')
SRC_URDF = os.path.join(SRC_SIM, 'so101_new_calib.urdf')
DST = os.path.join(REPO, '6Dof', 'Simulation', 'SO111')
DST_ASSETS = os.path.join(DST, 'assets')
DUMP = json.load(open(os.path.join(TOOLS, 'build', 'so111_fusion_dump.json')))
SPEC = json.load(open(os.path.join(TOOLS, 'build', 'so101_assembly_spec.json')))

MODEL = 'so111_new_calib'
RHO_PRINT = 496.7      # kg/m^3, effective density of printed parts (least squares on SO101 URDF masses)
M_MOTOR = 0.0577       # kg, lumped STS3215 mass (same regression)
LINK_ORDER = ['base_link', 'shoulder_link', 'upper_arm_link', 'lower_arm_link', 'forearm_link',
              'wrist_link', 'gripper_link', 'moving_jaw_so101_v1_link']
JOINT_ORDER = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'forearm_roll', 'wrist_flex', 'wrist_roll', 'gripper']
MJ_BODY = {l: l[:-5] if l.endswith('_link') else l for l in LINK_ORDER}
NEW_LINKS = {
    'lower_arm_link': [('SO111-J3-J4', 'so111_j3_j4_v1', '3d_printed'), ('sts3215_03a_v1__lower_arm', 'sts3215_03a_v1', 'sts3215')],
    'forearm_link':   [('SO111-J4-J5', 'so111_j4_j5_v1', '3d_printed'), ('sts3215_03a_v1__forearm', 'sts3215_03a_v1', 'sts3215')],
}
# poses of the J4/J5 motors in their link frames (place_new_motors.py); the Fusion components carry
# the mesh already in link coordinates, so their nested transforms are identity
MOTOR_POSES = json.load(open(os.path.join(TOOLS, 'build', 'so111_motor_poses.json')))
MATERIAL_RGBA = {'3d_printed': '1 0.82 0.12 1', 'sts3215': '0.1 0.1 0.1 1'}
PROVISIONAL = set()   # joints whose limits are still placeholders (forearm_roll +/-pi/2 was confirmed 2026-09-21)

# ---------------------------------------------------------------- math helpers
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])

def R_to_rpy(R):
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy > 1e-9:
        r, p, y = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    else:
        r, p, y = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0
    assert np.allclose(rpy_to_R(r, p, y), R, atol=1e-9), 'rpy extraction failed'
    return r, p, y

def R_to_quat(R):
    """Return (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z]); q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q

def T_of(xyz, rpy):
    T = np.eye(4); T[:3, :3] = rpy_to_R(*rpy); T[:3, 3] = xyz
    return T

def fusion_T(a):
    """Fusion row-major 4x4 (cm) -> numpy (m)."""
    T = np.array(a, dtype=float).reshape(4, 4); T[:3, 3] /= 100.0
    return T

def f(x):
    x = float(x)
    if abs(x) < 1e-12: x = 0.0
    s = '%.6g' % x
    return '0' if s == '-0' else s

def fv(v):
    return ' '.join(f(x) for x in v)

# ---------------------------------------------------------------- source URDF (stock links)
src = ET.parse(SRC_URDF).getroot()
src_links = {l.get('name'): l for l in src.findall('link')}

def parse_origin(el):
    xyz = [float(v) for v in (el.get('xyz') or '0 0 0').split()]
    rpy = [float(v) for v in (el.get('rpy') or '0 0 0').split()]
    return xyz, rpy

def stock_link(name):
    l = src_links[name]
    inert = l.find('inertial')
    o = inert.find('origin'); i = inert.find('inertia')
    link = {'name': name, 'inertial': {'xyz': parse_origin(o)[0], 'mass': float(inert.find('mass').get('value')),
            'I': [float(i.get(k)) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')]}, 'parts': []}
    coll = {os.path.basename(c.find('geometry/mesh').get('filename')) for c in l.findall('collision')}
    for vis in l.findall('visual'):
        mesh = os.path.splitext(os.path.basename(vis.find('geometry/mesh').get('filename')))[0]
        xyz, rpy = parse_origin(vis.find('origin'))
        link['parts'].append({'mesh': mesh, 'xyz': xyz, 'rpy': rpy, 'material': vis.find('material').get('name'),
                              'collision': (mesh + '.stl') in coll})
    return link

# ---------------------------------------------------------------- link frames and joints from Fusion
T_world = {n: fusion_T(e['T_world']) for n, e in DUMP['links'].items()}
# The elbow (upper arm + its motor) is unchanged from SO101, so lower_arm_link keeps the exact SO101
# lower_arm frame; the Fusion transform is that frame rounded to micrometres.
_T_lower_exact = fusion_T([l for l in SPEC['links'] if l['name'] == 'lower_arm_link'][0]['T_world'])
assert np.abs(_T_lower_exact - T_world['lower_arm_link']).max() < 2e-5
T_world['lower_arm_link'] = _T_lower_exact
joints = []
fj = {j['name']: j for j in DUMP['joints']}
for name in JOINT_ORDER:
    j = fj[name]
    Tpc = np.linalg.inv(T_world[j['parent']]) @ T_world[j['child']]
    # sanity: the Fusion joint origin/axis must be the child frame origin / +Z
    assert np.allclose(np.array(j['origin_cm']) / 100.0, T_world[j['child']][:3, 3], atol=2e-5)  # Fusion frames are rounded to micrometres
    assert np.allclose(np.array(j['axis']), T_world[j['child']][:3, 2], atol=1e-6)
    joints.append({'name': name, 'parent': j['parent'], 'child': j['child'], 'xyz': Tpc[:3, 3].tolist(),
                   'rpy': list(R_to_rpy(Tpc[:3, :3])), 'lower': j['lower'], 'upper': j['upper']})

# fixed gripper frame (from the stock URDF)
gf = [j for j in src.findall('joint') if j.get('name') == 'gripper_frame_joint'][0]
gf_xyz, gf_rpy = parse_origin(gf.find('origin'))

# ---------------------------------------------------------------- new links: meshes + inertials
os.makedirs(DST_ASSETS, exist_ok=True)
motor_tris_m = read_stl(os.path.join(SRC_SIM, 'assets', 'sts3215_03a_v1.stl'))          # meters
motor_vol = mass_properties(motor_tris_m, 1.0)[0]
RHO_MOTOR = M_MOTOR / motor_vol

def ensure_meters(path):
    tris = read_stl(path)
    if np.abs(tris).max() > 1.0:       # exported by Fusion in mm
        write_stl(path, tris / 1000.0)
        tris = tris / 1000.0
    return tris

def new_link(name):
    Tl = T_world[name]
    children = {c['component']: c for c in DUMP['links'][name]['children']}
    link = {'name': name, 'parts': []}
    props = []
    for comp, mesh, material in NEW_LINKS[name]:
        Tc = np.linalg.inv(Tl) @ fusion_T(children[comp]['T_world'])
        if material == 'sts3215':
            assert np.abs(Tc - np.eye(4)).max() < 2e-5, 'motor component must be at identity in its link'  # link frames are rounded to micrometres in Fusion
            Tc = np.array(MOTOR_POSES[name]['T_link_motor'])
            tris = motor_tris_m; rho = RHO_MOTOR
        else:
            tris = ensure_meters(os.path.join(DST_ASSETS, mesh + '.stl')); rho = RHO_PRINT
        link['parts'].append({'mesh': mesh, 'xyz': Tc[:3, 3].tolist(), 'rpy': list(R_to_rpy(Tc[:3, :3])),
                              'material': material, 'collision': True})
        props.append(mass_properties(transform(tris, Tc), rho))
    M = sum(m for m, _, _ in props)
    com = sum(m * c for m, c, _ in props) / M
    I = np.zeros((3, 3))
    for m, c, Ic in props:
        d = c - com
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    link['inertial'] = {'xyz': com.tolist(), 'mass': M, 'I': [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]}
    return link

links = [new_link(n) if n in NEW_LINKS else stock_link(n) for n in LINK_ORDER]

# ---------------------------------------------------------------- assets
used_meshes = []
for l in links:
    for p in l['parts']:
        if p['mesh'] not in used_meshes: used_meshes.append(p['mesh'])
for mesh in used_meshes:
    srcp = os.path.join(SRC_SIM, 'assets', mesh + '.stl')
    if os.path.exists(srcp):
        shutil.copy2(srcp, os.path.join(DST_ASSETS, mesh + '.stl'))
    else:
        assert os.path.exists(os.path.join(DST_ASSETS, mesh + '.stl')), mesh

# ---------------------------------------------------------------- URDF
HEADER = [
    '<!-- SO111: 6-DoF variant of the SO101 arm (forearm_roll added between elbow_flex and wrist_flex) -->',
    '<!-- Generated by 6Dof/tools/gen_so111_description.py from the Fusion assembly "SO111-Assembly" -->',
    '<!-- Zero pose: middle of range for the stock joints (as so101_new_calib); forearm_roll zero = forearm straight as assembled -->',
    '<!-- Masses of lower_arm_link/forearm_link are estimates: printed parts %.0f kg/m^3, STS3215 %.1f g (regressed from the SO101 URDF) -->' % (RHO_PRINT, M_MOTOR * 1000),
]
u = ['<?xml version="1.0" encoding="utf-8"?>'] + HEADER + ['<robot name="%s">' % MODEL, '', '  <!-- Materials -->']
for mname, rgba in MATERIAL_RGBA.items():
    u += ['  <material name="%s">' % mname, '    <color rgba="%s"/>' % rgba, '  </material>']
for l in links:
    u += ['', '  <!-- Link %s -->' % MJ_BODY[l['name']], '  <link name="%s">' % l['name'], '    <inertial>',
          '      <origin xyz="%s" rpy="0 0 0"/>' % fv(l['inertial']['xyz']),
          '      <mass value="%s"/>' % f(l['inertial']['mass']),
          '      <inertia ixx="%s" ixy="%s" ixz="%s" iyy="%s" iyz="%s" izz="%s"/>' % tuple(f(l['inertial']['I'][k]) for k in (0, 3, 4, 1, 5, 2)),
          '    </inertial>']
    for p in l['parts']:
        u += ['    <!-- Part %s -->' % p['mesh'], '    <visual>',
              '      <origin xyz="%s" rpy="%s"/>' % (fv(p['xyz']), fv(p['rpy'])),
              '      <geometry>', '        <mesh filename="assets/%s.stl"/>' % p['mesh'], '      </geometry>',
              '      <material name="%s"/>' % p['material'], '    </visual>']
        if p['collision']:
            u += ['    <collision>', '      <origin xyz="%s" rpy="%s"/>' % (fv(p['xyz']), fv(p['rpy'])),
                  '      <geometry>', '        <mesh filename="assets/%s.stl"/>' % p['mesh'], '      </geometry>', '    </collision>']
    u += ['  </link>']
u += ['', '  <!-- Gripper frame (dummy link + fixed joint) -->', '  <link name="gripper_frame_link">', '    <inertial>',
      '      <origin xyz="0 0 0" rpy="0 0 0"/>', '      <mass value="1e-9"/>',
      '      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>', '    </inertial>', '  </link>',
      '  <joint name="gripper_frame_joint" type="fixed">',
      '    <origin xyz="%s" rpy="%s"/>' % (fv(gf_xyz), fv(gf_rpy)),
      '    <parent link="gripper_link"/>', '    <child link="gripper_frame_link"/>', '  </joint>']
for i, j in enumerate(joints, 1):
    u += ['', '  <!-- Joint from %s to %s -->' % (MJ_BODY[j['parent']], MJ_BODY[j['child']])]
    if j['name'] in PROVISIONAL:
        u += ['  <!-- NOTE: joint limits are provisional (not yet measured on the arm) -->']
    u += ['  <joint name="%s" type="revolute">' % j['name'],
          '    <origin xyz="%s" rpy="%s"/>' % (fv(j['xyz']), fv(j['rpy'])),
          '    <parent link="%s"/>' % j['parent'], '    <child link="%s"/>' % j['child'],
          '    <axis xyz="0 0 1"/>',
          '    <limit effort="10" velocity="10" lower="%s" upper="%s"/>' % (f(j['lower']), f(j['upper'])),
          '  </joint>', '  <transmission name="%s_trans">' % j['name'],
          '    <type>transmission_interface/SimpleTransmission</type>', '    <joint name="%s">' % j['name'],
          '      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>', '    </joint>',
          '    <actuator name="motor%d">' % i,
          '      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>',
          '      <mechanicalReduction>1</mechanicalReduction>', '    </actuator>', '  </transmission>']
u += ['', '</robot>', '']
open(os.path.join(DST, MODEL + '.urdf'), 'w').write('\n'.join(u))

# ---------------------------------------------------------------- MJCF
def q(rpy):
    return fv(R_to_quat(rpy_to_R(*rpy)))

x = ['<?xml version="1.0" ?>'] + HEADER + ['<mujoco model="%s">' % MODEL,
     '  <compiler angle="radian" meshdir="assets" autolimits="true"/>',
     '  <default>', '    <default class="%s">' % MODEL,
     '      <joint damping="1" frictionloss="0.1" armature="0.005"/>', '      <position kp="50"/>',
     '      <default class="visual">', '        <geom type="mesh" contype="0" conaffinity="0" group="2"/>', '      </default>',
     '      <default class="collision">', '        <geom group="3"/>', '      </default>', '    </default>', '  </default>',
     '  <!-- Additional joints_properties.xml (same values as Simulation/SO101) -->', '  <default>',
     '    <default class="sts3215">', '      <geom contype="0" conaffinity="0"/>',
     '      <joint damping="0.60" frictionloss="0.052" armature="0.028"/>',
     '      <position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>', '    </default>',
     '    <default class="backlash">',
     '      <joint damping="0.01" frictionloss="0" armature="0.01" limited="true" range="-0.008726646259971648 0.008726646259971648"/>',
     '    </default>', '  </default>', '  <worldbody>']
child_joint = {j['child']: j for j in joints}
link_by = {l['name']: l for l in links}

def emit_body(name, depth):
    ind = '  ' * depth
    l = link_by[name]; j = child_joint.get(name)
    if j is None:
        x.append('%s<!-- Link %s -->' % (ind, MJ_BODY[name]))
        x.append('%s<body name="%s" pos="0 0 0" quat="1 0 0 0" childclass="%s">' % (ind, MJ_BODY[name], MODEL))
    else:
        x.append('%s<!-- Link %s -->' % (ind, MJ_BODY[name]))
        x.append('%s<body name="%s" pos="%s" quat="%s">' % (ind, MJ_BODY[name], fv(j['xyz']), q(j['rpy'])))
        x.append('%s  <!-- Joint from %s to %s -->' % (ind, MJ_BODY[j['parent']], MJ_BODY[name]))
        if j['name'] in PROVISIONAL:
            x.append('%s  <!-- NOTE: joint range is provisional (not yet measured on the arm) -->' % ind)
        x.append('%s  <joint axis="0 0 1" name="%s" type="hinge" range="%s %s" class="sts3215"/>' % (ind, j['name'], f(j['lower']), f(j['upper'])))
    I = l['inertial']['I']
    x.append('%s  <inertial pos="%s" mass="%s" fullinertia="%s"/>' % (ind, fv(l['inertial']['xyz']), f(l['inertial']['mass']), fv(I)))
    for p in l['parts']:
        x.append('%s  <!-- Part %s -->' % (ind, p['mesh']))
        for cls in (['visual', 'collision'] if (p['collision'] and name != 'base_link') else ['visual']):
            x.append('%s  <geom type="mesh" class="%s" pos="%s" quat="%s" mesh="%s" material="%s_material"/>' % (ind, cls, fv(p['xyz']), q(p['rpy']), p['mesh'], p['mesh']))
    if name == 'base_link':
        x.append('%s  <!-- Frame baseframe -->' % ind)
        x.append('%s  <site group="3" name="baseframe" pos="0 0 0" quat="1 0 0 0"/>' % ind)
    if name == 'gripper_link':
        x.append('%s  <!-- Frame gripperframe -->' % ind)
        x.append('%s  <site group="3" name="gripperframe" pos="%s" quat="%s"/>' % (ind, fv(gf_xyz), q(gf_rpy)))
    for jj in joints:
        if jj['parent'] == name:
            emit_body(jj['child'], depth + 1)
    x.append('%s</body>' % ind)

emit_body('base_link', 2)
x += ['  </worldbody>', '  <asset>']
for mesh in used_meshes:
    x.append('    <mesh file="%s.stl"/>' % mesh)
mat_of = {p['mesh']: p['material'] for l in links for p in l['parts']}
for mesh in used_meshes:
    x.append('    <material name="%s_material" rgba="%s"/>' % (mesh, MATERIAL_RGBA[mat_of[mesh]]))
x += ['  </asset>', '  <actuator>']
for j in joints:
    x.append('    <position class="sts3215" name="%s" joint="%s" forcerange="-3.35 3.35" ctrlrange="%s %s"/>' % (j['name'], j['name'], f(j['lower']), f(j['upper'])))
x += ['  </actuator>', '  <equality/>', '</mujoco>', '']
open(os.path.join(DST, MODEL + '.xml'), 'w').write('\n'.join(x))

scene = open(os.path.join(SRC_SIM, 'scene.xml')).read().replace('so101_new_calib.xml', MODEL + '.xml')
open(os.path.join(DST, 'scene.xml'), 'w').write(scene)
shutil.copy2(os.path.join(SRC_SIM, 'joints_properties.xml'), os.path.join(DST, 'joints_properties.xml'))

# ---------------------------------------------------------------- report
print('wrote', os.path.join(DST, MODEL + '.urdf'), 'and .xml')
for j in joints:
    print('  joint %-13s %-14s -> %-24s xyz=%s rpy=%s lim=(%s, %s)' % (j['name'], j['parent'], j['child'], fv(j['xyz']), fv(j['rpy']), f(j['lower']), f(j['upper'])))
for n in NEW_LINKS:
    l = link_by[n]
    print('  %s: mass=%.4f kg com=%s' % (n, l['inertial']['mass'], fv(l['inertial']['xyz'])))
    for p in l['parts']:
        print('     part %-18s xyz=%s rpy=%s' % (p['mesh'], fv(p['xyz']), fv(p['rpy'])))
print('  motor mesh volume %.2f cm3 -> rho_motor %.0f kg/m3' % (motor_vol * 1e6, RHO_MOTOR))
