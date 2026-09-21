#!/usr/bin/env python3
"""Validate the generated SO111 description: parse, FK vs Fusion, MuJoCo compile + short sim."""
import json, os, sys
import numpy as np
TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TOOLS, '..', '..'))
DST = os.path.join(REPO, '6Dof', 'Simulation', 'SO111')
DUMP = json.load(open(os.path.join(TOOLS, 'build', 'so111_fusion_dump.json')))
ok = True

def T_fusion(a):
    T = np.array(a, float).reshape(4, 4); T[:3, 3] /= 100.0; return T

# ---- URDF
import yourdfpy
urdf = yourdfpy.URDF.load(os.path.join(DST, 'so111_new_calib.urdf'), build_scene_graph=True, load_meshes=True)
print('URDF: %d links, %d joints (%d actuated): %s' % (len(urdf.link_map), len(urdf.joint_map), len(urdf.actuated_joint_names), urdf.actuated_joint_names))
urdf.update_cfg(np.zeros(len(urdf.actuated_joint_names)))
worst = 0.0
for name, e in DUMP['links'].items():
    Tu = urdf.get_transform(name, 'base_link'); Tf = T_fusion(e['T_world'])
    d = np.abs(Tu - Tf).max(); worst = max(worst, d)
    print('  link %-26s URDF-vs-Fusion max diff %.2e' % (name, d))
Tg = urdf.get_transform('gripper_frame_link', 'base_link')
gf = np.array(DUMP['gripper_frame_world_cm']) / 100.0
print('  gripper_frame URDF pos (mm):', np.round(Tg[:3, 3] * 1000, 3), ' Fusion sketch:', np.round(gf * 1000, 3), ' diff %.3f mm' % (np.linalg.norm(Tg[:3, 3] - gf) * 1000))
ok &= worst < 1e-4 and np.linalg.norm(Tg[:3, 3] - gf) < 1e-4
# joint limits and mesh loading
for jn in urdf.actuated_joint_names:
    j = urdf.joint_map[jn]
    assert j.limit.lower < 0 < j.limit.upper or jn == 'gripper', jn
print('  scene meshes loaded:', len(urdf.scene.geometry))

# ---- MuJoCo
import mujoco
m = mujoco.MjModel.from_xml_path(os.path.join(DST, 'scene.xml'))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print('MJCF: nbody=%d njnt=%d nu=%d ngeom=%d, total mass %.4f kg' % (m.nbody, m.njnt, m.nu, m.ngeom, sum(m.body_mass[1:])))
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
print('  gripperframe site pos (mm):', np.round(d.site_xpos[sid] * 1000, 3), ' diff vs Fusion %.3f mm' % (np.linalg.norm(d.site_xpos[sid] - gf) * 1000))
ok &= np.linalg.norm(d.site_xpos[sid] - gf) < 1e-4
for i in range(m.njnt):
    print('  joint %-13s range=%s' % (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i), np.round(m.jnt_range[i], 4)))
# hold zero pose with the position actuators, then command mid-range targets
d.ctrl[:] = 0.0
for _ in range(2000): mujoco.mj_step(m, d)
print('  after 2000 steps holding zero: max |qpos| = %.4f rad, gripperframe drift %.2f mm' % (np.abs(d.qpos).max(), np.linalg.norm(d.site_xpos[sid] - gf) * 1000))
ok &= np.isfinite(d.qpos).all() and np.abs(d.qpos).max() < 0.05
targets = np.array([0.5, -0.4, 0.6, 0.8, -0.5, 0.7, 0.5])
d.ctrl[:] = targets
for _ in range(4000): mujoco.mj_step(m, d)
err = np.abs(d.qpos - targets)
print('  after commanding targets: max tracking error %.4f rad, warnings=%d' % (err.max(), int(d.warning.number.sum())))
ok &= np.isfinite(d.qpos).all() and err.max() < 0.05 and d.warning.number.sum() == 0
# offscreen render (best effort)
try:
    mujoco.mj_resetData(m, d); d.qpos[:] = [0.3, -0.5, 0.8, 0.6, -0.4, 0.5, 0.6]; mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 720, 960)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0.15, 0, 0.15]; cam.distance = 0.7; cam.azimuth = 150; cam.elevation = -20
    r.update_scene(d, cam); img = r.render()
    import PIL.Image
    out = os.path.join(TOOLS, 'build', 'so111_mujoco.png'); PIL.Image.fromarray(img).save(out); print('  rendered', out)
except Exception as e:
    print('  render skipped:', type(e).__name__, e)
print('VALIDATION', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
