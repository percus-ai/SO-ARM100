"""Build the SO101 link assembly in Fusion from the spec produced by gen_spec.py.

Run through the Fusion MCP `script` execute (defines run(_context)). Set SPEC_PATH first.
Recipe (verified to survive snapshots / timeline replay / joint edits):
  * one top-level component per URDF link, moved with Occurrence.transform2 (world pose)
  * STEP bodies are moved INSIDE their part component with a MoveFeature (link frame);
    motor meshes are pre-transformed STLs. Nested occurrences therefore keep identity
    transforms -- nested transform overrides are silently dropped by Fusion on recompute.
  * as-built revolute joints created with (occurrenceOne=child, occurrenceTwo=parent) so
    that a positive joint value rotates the child about +Z of its link frame like URDF.
  * joint / link / sketch names follow the URDF (shoulder_pan, ..., <joint>_axis).
"""
import adsk.core, adsk.fusion, json, math

SPEC_PATH = '/path/to/build/so101_assembly_spec.json'
DOC_NAME = 'SO111-Assembly'
FOLDER_DOC_PREFIX = 'SO111-J'   # save next to an open document whose name starts with this


def mat(a):
    m = adsk.core.Matrix3D.create(); m.setWithArray(a); return m

def matmul(a, b):
    return [sum(a[i*4+k]*b[k*4+j] for k in range(4)) for i in range(4) for j in range(4)]

def apply(T, p):
    return [T[0]*p[0]+T[1]*p[1]+T[2]*p[2]+T[3], T[4]*p[0]+T[5]*p[1]+T[6]*p[2]+T[7], T[8]*p[0]+T[9]*p[1]+T[10]*p[2]+T[11]]

def rotz(t):
    c, s = math.cos(t), math.sin(t)
    return [c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1]

def maxdiff(a, b):
    return max(abs(x-y) for x, y in zip(a, b))

def last_child(locc):
    return locc.childOccurrences.item(locc.childOccurrences.count - 1)

def child_by_name(locc, name):
    for c in locc.childOccurrences:
        if c.component.name == name:
            return c
    raise KeyError(name)

def check_placement(spec, root):
    """Compare every part body's world bbox center with the URDF prediction (mm)."""
    occs = {o.component.name: o for o in root.occurrences}
    bad = []
    for link in spec['links']:
        locc = occs[link['name']]
        for part in link['parts']:
            c = child_by_name(locc, part['comp_name'])
            exp_T = matmul(link['T_world'], part['T_local'])
            smn, smx = part['stl_bbox_mm']
            exp_c = apply(exp_T, [(smn[k]+smx[k])/20.0 for k in range(3)])
            comp = c.component
            body = (comp.bRepBodies.item(0) if comp.bRepBodies.count else comp.meshBodies.item(0)).createForAssemblyContext(c)
            bb = body.boundingBox
            got_c = [(bb.minPoint.x+bb.maxPoint.x)/2, (bb.minPoint.y+bb.maxPoint.y)/2, (bb.minPoint.z+bb.maxPoint.z)/2]
            d = maxdiff(got_c, exp_c) * 10
            if d > 0.3:
                bad.append(f"{part['comp_name']}({d:.0f}mm)")
    return bad

def build_links(app, spec, root):
    im = app.importManager
    link_occs = {}
    for link in spec['links']:
        locc = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        lcomp = locc.component; lcomp.name = link['name']; link_occs[link['name']] = locc
        for part in link['parts']:
            if part['kind'] == 'step':
                opts = im.createSTEPImportOptions(part['path']); opts.isViewFit = False
                im.importToTarget(opts, lcomp)
                pcomp = last_child(locc).component; pcomp.name = part['comp_name']
                col = adsk.core.ObjectCollection.create(); col.add(pcomp.bRepBodies.item(0))
                inp = pcomp.features.moveFeatures.createInput2(col)
                inp.defineAsFreeMove(mat(part['T_local']))
                pcomp.features.moveFeatures.add(inp)
            else:
                lcomp.occurrences.addNewComponent(adsk.core.Matrix3D.create())
                pcomp = last_child(locc).component; pcomp.name = part['comp_name']
                bf = pcomp.features.baseFeatures.add(); bf.startEdit()
                pcomp.meshBodies.add(part['path_transformed'], adsk.fusion.MeshUnits.MeterMeshUnit, bf)
                bf.finishEdit()
    for link in spec['links']:
        link_occs[link['name']].transform2 = mat(link['T_world'])
    link_occs['base_link'].isGrounded = True
    return link_occs

def make_joint(root, parent_occ, child_occ, jspec):
    comp = child_occ.component
    sk = comp.sketches.add(comp.xYConstructionPlane); sk.name = jspec['name'] + '_axis'
    p0 = sk.modelToSketchSpace(adsk.core.Point3D.create(0, 0, 0))
    p1 = sk.modelToSketchSpace(adsk.core.Point3D.create(0, 0, 2.0))
    ln = sk.sketchCurves.sketchLines.addByTwoPoints(p0, p1)
    geo = adsk.fusion.JointGeometry.createByPoint(ln.startSketchPoint.createForAssemblyContext(child_occ))
    jin = root.asBuiltJoints.createInput(child_occ, parent_occ, geo)
    jin.setAsRevoluteJointMotion(adsk.fusion.JointDirections.CustomJointDirection, ln.createForAssemblyContext(child_occ))
    j = root.asBuiltJoints.add(jin); j.name = jspec['name']
    lim = adsk.fusion.RevoluteJointMotion.cast(j.jointMotion).rotationLimits
    lim.isMinimumValueEnabled = True; lim.minimumValue = jspec['lower']
    lim.isMaximumValueEnabled = True; lim.maximumValue = jspec['upper']
    lim.isRestValueEnabled = True; lim.restValue = 0.0
    return j

def check_joints(spec, root, occs):
    links = {l['name']: l for l in spec['links']}
    worst = 0.0
    for js in spec['joints']:
        j = root.asBuiltJoints.itemByName(js['name'])
        tw = links[js['child']]['T_world']
        rm = adsk.fusion.RevoluteJointMotion.cast(j.jointMotion)
        rm.rotationValue = 0.3
        got = occs[js['child']].transform2.asArray()
        rm.rotationValue = 0.0
        g = j.geometry
        worst = max(worst, maxdiff([g.origin.x, g.origin.y, g.origin.z], [tw[3], tw[7], tw[11]]),
                    maxdiff(got, matmul(tw, rotz(0.3))))
    return worst

def add_gripper_frame(spec, occs):
    fr = spec['frames'][0]
    gcomp = occs['gripper_link'].component
    sk = gcomp.sketches.add(gcomp.xYConstructionPlane); sk.name = fr['name'].replace('_link', '')
    T = fr['T_local']
    o = adsk.core.Point3D.create(T[3], T[7], T[11])
    px = adsk.core.Point3D.create(T[3]+T[0], T[7]+T[4], T[11]+T[8])
    pz = adsk.core.Point3D.create(T[3]+2*T[2], T[7]+2*T[6], T[11]+2*T[10])
    sk.sketchCurves.sketchLines.addByTwoPoints(sk.modelToSketchSpace(o), sk.modelToSketchSpace(px))
    sk.sketchCurves.sketchLines.addByTwoPoints(sk.modelToSketchSpace(o), sk.modelToSketchSpace(pz))

def run(_context: str):
    app = adsk.core.Application.get()
    spec = json.load(open(SPEC_PATH))
    doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    link_occs = build_links(app, spec, root)
    if design.snapshots.hasPendingSnapshot: design.snapshots.add()
    for lname, locc in link_occs.items():
        if locc.childOccurrences.count >= 2:
            col = adsk.core.ObjectCollection.create()
            for c in locc.childOccurrences: col.add(c)
            root.rigidGroups.add(col, True).name = lname + '_rigid'
    for js in spec['joints']:
        make_joint(root, link_occs[js['parent']], link_occs[js['child']], js)
    add_gripper_frame(spec, link_occs)
    if design.snapshots.hasPendingSnapshot: design.snapshots.add()
    bad = check_placement(spec, root)
    worst = check_joints(spec, root, link_occs)
    design.timeline.markerPosition = 0; design.timeline.moveToEnd()
    bad2 = check_placement(spec, root)
    print('placement:', bad or 'ok', '| after timeline roll:', bad2 or 'ok', '| joint worst deviation (cm):', '%.1e' % worst)
    if bad or bad2 or worst > 1e-2:
        print('NOT saved'); return
    folder = None
    for d in app.documents:
        if d.name.startswith(FOLDER_DOC_PREFIX) and d.isSaved:
            folder = d.dataFile.parentFolder; break
    if folder is None:
        folder = app.data.activeProject.rootFolder
    doc.activate()
    print('saved:', doc.saveAs(DOC_NAME, folder, 'generated by fusion_build_assembly.py', ''), '->', folder.name, '/', DOC_NAME)
