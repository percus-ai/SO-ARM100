"""Generate a Fusion assembly spec (JSON) from the existing SO101 URDF.
All transforms are 4x4 row-major with translation in cm (Fusion internal units).
Motor meshes are pre-transformed into their link frame (meters) so that the Fusion
build script can insert them without any nested occurrence transform.
Usage: python3 gen_spec.py [out_dir]   (default: ./build next to this script)"""
import json, math, os, struct, sys
import numpy as np
import xml.etree.ElementTree as ET

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
URDF = f'{REPO}/Simulation/SO101/so101_new_calib.urdf'
STEP = f'{REPO}/STEP/SO101'
ASSETS = f'{REPO}/Simulation/SO101/assets'
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'build')
OUT = os.path.join(OUT_DIR, 'so101_assembly_spec.json')
MESH_DIR = os.path.join(OUT_DIR, 'meshes')

PART_SOURCE = {
    'base_motor_holder_so101_v1':        ('step', f'{STEP}/Base_motor_holder_SO101.step'),
    'base_so101_v2':                     ('step', f'{STEP}/Base_SO101.step'),
    'waveshare_mounting_plate_so101_v2': ('step', f'{STEP}/WaveShare_Mounting_Plate_SO101.step'),
    'motor_holder_so101_base_v1':        ('step', f'{STEP}/Motor_holder_SO101_Base.step'),
    'rotation_pitch_so101_v1':           ('step', f'{STEP}/Rotation_Pitch_SO101.step'),
    'upper_arm_so101_v1':                ('step', f'{STEP}/Upper_arm_SO101.step'),
    'under_arm_so101_v1':                ('step', f'{STEP}/Under_arm_SO101.step'),
    'motor_holder_so101_wrist_v1':       ('step', f'{STEP}/Motor_holder_SO101_Wrist.step'),
    'wrist_roll_pitch_so101_v2':         ('step', f'{STEP}/Wrist_Roll_Pitch_SO101.step'),
    'wrist_roll_follower_so101_v1':      ('step', f'{STEP}/Follower_Specific/Wrist_Roll_Follower_SO101.step'),
    'moving_jaw_so101_v1':               ('step', f'{STEP}/Follower_Specific/Moving_Jaw_SO101.step'),
    'sts3215_03a_v1':                    ('mesh', f'{ASSETS}/sts3215_03a_v1.stl'),
    'sts3215_03a_no_horn_v1':            ('mesh', f'{ASSETS}/sts3215_03a_no_horn_v1.stl'),
}

def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def snap_angle(a, tol=1e-4):
    k = round(a / (math.pi / 2))
    return k * math.pi / 2 if abs(a - k * math.pi / 2) < tol else a

def origin_T(el):
    xyz = [0,0,0]; rpy = [0,0,0]
    if el is not None:
        if el.get('xyz'): xyz = [float(v) for v in el.get('xyz').split()]
        if el.get('rpy'): rpy = [snap_angle(float(v)) for v in el.get('rpy').split()]
    T = np.eye(4); T[:3,:3] = rpy_to_R(*rpy); T[:3,3] = xyz
    return T

def to_fusion(T):
    """meters -> cm, flatten row-major, snap tiny values."""
    A = T.copy(); A[:3,3] *= 100.0
    A[:3,3][np.abs(A[:3,3]) < 1e-9] = 0.0
    return [float(v) for v in A.flatten()]

def stl_bbox_mm(path):
    with open(path,'rb') as f:
        f.read(80); data = f.read()
    n = struct.unpack('<I', data[:4])[0]
    dt = np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
    v = np.frombuffer(data[4:], dtype=dt, count=n)['v'].reshape(-1,3)
    return (v.min(0)*1000).round(3).tolist(), (v.max(0)*1000).round(3).tolist()

root = ET.parse(URDF).getroot()
links = {l.get('name'): l for l in root.findall('link')}
joints = root.findall('joint')
parent_of = {}   # child -> (joint el)
for j in joints:
    parent_of[j.find('child').get('link')] = j

# world transform of each link at zero pose
T_world = {}
def world(link):
    if link in T_world: return T_world[link]
    if link not in parent_of:
        T_world[link] = np.eye(4); return T_world[link]
    j = parent_of[link]
    T = world(j.find('parent').get('link')) @ origin_T(j.find('origin'))
    T_world[link] = T; return T

spec = {'links': [], 'joints': [], 'frames': []}
order = ['base_link','shoulder_link','upper_arm_link','lower_arm_link','wrist_link','gripper_link','moving_jaw_so101_v1_link']
for name in order:
    l = links[name]
    entry = {'name': name, 'T_world': to_fusion(world(name)), 'parts': []}
    inertial = l.find('inertial')
    if inertial is not None:
        entry['mass_kg'] = float(inertial.find('mass').get('value'))
        entry['com_m'] = [float(v) for v in inertial.find('origin').get('xyz').split()]
    for vis in l.findall('visual'):
        mesh = os.path.splitext(os.path.basename(vis.find('geometry/mesh').get('filename')))[0]
        kind, path = PART_SOURCE[mesh]
        assert os.path.exists(path), path
        mn, mx = stl_bbox_mm(f'{ASSETS}/{mesh}.stl')
        entry['parts'].append({'name': mesh, 'kind': kind, 'path': path,
                               'T_local': to_fusion(origin_T(vis.find('origin'))),
                               'stl_bbox_mm': [mn, mx]})
    spec['links'].append(entry)

for j in joints:
    parent = j.find('parent').get('link'); child = j.find('child').get('link')
    if j.get('type') == 'fixed':
        spec['frames'].append({'name': child, 'parent': parent, 'T_local': to_fusion(origin_T(j.find('origin')))})
        continue
    lim = j.find('limit')
    spec['joints'].append({'name': j.get('name'), 'parent': parent, 'child': child,
                           'axis': [float(v) for v in j.find('axis').get('xyz').split()],
                           'lower': float(lim.get('lower')), 'upper': float(lim.get('upper')),
                           'T_parent_child': to_fusion(origin_T(j.find('origin')))})

# keep chain order base->tip
jorder = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
spec['joints'].sort(key=lambda x: jorder.index(x['name']))

# pre-transform motor meshes into link frames
def read_stl(path):
    data = open(path,'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    dt = np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
    arr = np.frombuffer(data[84:], dtype=dt, count=n)
    return arr['n'].astype(np.float64), arr['v'].astype(np.float64)

def write_stl(path, normals, tris):
    dt = np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
    arr = np.zeros(len(tris), dtype=dt); arr['n'] = normals; arr['v'] = tris
    with open(path,'wb') as f:
        f.write(b'pre-transformed by gen_spec.py'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(tris))); f.write(arr.tobytes())

SHORT = {'base_link':'base','shoulder_link':'shoulder','upper_arm_link':'upper_arm','lower_arm_link':'lower_arm',
         'wrist_link':'wrist','gripper_link':'gripper','moving_jaw_so101_v1_link':'moving_jaw'}
os.makedirs(MESH_DIR, exist_ok=True)
for l in spec['links']:
    for p in l['parts']:
        if p['kind'] == 'mesh':
            T = np.array(p['T_local']).reshape(4,4); R = T[:3,:3]; t = T[:3,3] / 100.0
            nrm, tri = read_stl(p['path'])
            p['comp_name'] = f"{p['name']}__{SHORT[l['name']]}"
            p['path_transformed'] = os.path.join(MESH_DIR, p['comp_name'] + '.stl')
            write_stl(p['path_transformed'], nrm @ R.T, tri @ R.T + t)
        else:
            p['comp_name'] = p['name']

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(spec, open(OUT,'w'), indent=1)
print('wrote', OUT)
for l in spec['links']:
    t = l['T_world']
    print(f"{l['name']:26s} world_pos_mm=({t[3]*10:.2f},{t[7]*10:.2f},{t[11]*10:.2f}) parts={[p['name'] for p in l['parts']]}")
for j in spec['joints']:
    print(f"joint {j['name']:14s} {j['parent']} -> {j['child']} axis={j['axis']} lim=({j['lower']:.3f},{j['upper']:.3f})")
for f in spec['frames']:
    print('frame', f['name'], 'on', f['parent'])
