# SO111 (6-DoF SO101) - URDF and MuJoCo Description

`SO111` is the SO101 arm with one extra axis: a forearm roll (`forearm_roll`) between
`elbow_flex` and `wrist_flex`. Two new printed parts carry it: `so111_j3_j4_v1` (replaces the
SO101 under-arm and holds the J4 motor) and `so111_j4_j5_v1` (cup on the J4 horn that holds the
J5 motor). Everything from the base to the upper arm and from the wrist to the gripper is the
stock SO101 hardware.

## Files

- `so111_new_calib.urdf` / `so111_new_calib.xml` - URDF and MJCF, meshes referenced relatively from `assets/`.
- `scene.xml` - MuJoCo scene (floor, light) including the MJCF.
- `joints_properties.xml` - STS3215 joint defaults (copied from `Simulation/SO101`).
- `assets/` - STL meshes in metres. Stock parts are copied from `Simulation/SO101/assets`;
  `so111_*.stl` are exported from the Fusion part designs.

## Kinematic chain

| # | joint | parent -> child | range (rad) |
|---|-------|-----------------|-------------|
| 1 | shoulder_pan | base_link -> shoulder_link | -1.92 .. 1.92 |
| 2 | shoulder_lift | shoulder_link -> upper_arm_link | -1.745 .. 1.745 |
| 3 | elbow_flex | upper_arm_link -> lower_arm_link | -1.69 .. 1.69 |
| 4 | forearm_roll | lower_arm_link -> forearm_link | -1.571 .. 1.571 |
| 5 | wrist_flex | forearm_link -> wrist_link | -1.658 .. 1.658 |
| 6 | wrist_roll | wrist_link -> gripper_link | -2.744 .. 2.841 |
| 7 | gripper | gripper_link -> moving_jaw_so101_v1_link | -0.175 .. 1.745 |

Every joint rotates about +Z of its child link frame; `gripper_frame_link` (URDF) and the
`gripperframe` site (MJCF) mark the tool frame as in SO101.

Zero pose: identical to `so101_new_calib` for the stock joints (middle of range, upper arm
vertical, forearm horizontal); `forearm_roll = 0` is the forearm as assembled (cup upright).

## How it was generated

1. `6Dof/tools/gen_spec.py` turns `Simulation/SO101/so101_new_calib.urdf` into a Fusion build
   spec; `fusion_build_assembly.py` builds the SO101 link assembly in Fusion (one component per
   link, as-built revolute joints named like the URDF joints).
2. The two new parts and their motors were placed in that assembly (Fusion document
   `SO111-Assembly`) and joined with `elbow_flex`, `forearm_roll`, `wrist_flex`.
3. `gen_so111_description.py` reads the link/joint transforms exported from Fusion
   (`tools/build/so111_fusion_dump.json`) and writes the URDF/MJCF. Unchanged links reuse the
   SO101 inertials and meshes. `lower_arm_link` and `forearm_link` inertials are computed from
   the meshes with an effective printed-part density of 497 kg/m^3 and a lumped STS3215 mass of
   57.7 g, both regressed from the SO101 URDF link masses (fit residual within +/-13 g).
4. `validate_so111.py` loads both files (yourdfpy, MuJoCo), compares the forward kinematics
   with the Fusion assembly and runs a short simulation.
