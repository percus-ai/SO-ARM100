"""Mesh utilities: binary STL IO and mass properties of closed triangle meshes."""
import struct, numpy as np

def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    if len(data) == 84 + n * 50:
        dt = np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
        return np.frombuffer(data[84:], dtype=dt, count=n)['v'].astype(np.float64)
    v = []
    for line in open(path, errors='ignore'):
        s = line.split()
        if s and s[0] == 'vertex': v.append([float(x) for x in s[1:4]])
    return np.array(v).reshape(-1, 3, 3)

def write_stl(path, tris):
    tris = np.asarray(tris, dtype=np.float64)
    n = np.cross(tris[:,1]-tris[:,0], tris[:,2]-tris[:,0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    dt = np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
    arr = np.zeros(len(tris), dtype=dt); arr['n'] = n; arr['v'] = tris
    with open(path, 'wb') as f:
        f.write(b'SO111 asset'.ljust(80, b'\0')); f.write(struct.pack('<I', len(tris))); f.write(arr.tobytes())

def transform(tris, T):
    """Apply 4x4 (meters) to triangles (N,3,3)."""
    R, t = T[:3,:3], T[:3,3]
    return tris @ R.T + t

def mass_properties(tris, density):
    """Volume, center of mass and inertia tensor (about the COM) of a closed mesh.
    Uses the signed-tetrahedron decomposition (Tonon 2004). Returns (mass, com, I_com)."""
    a, b, c = tris[:,0], tris[:,1], tris[:,2]
    det = np.einsum('ij,ij->i', a, np.cross(b, c))
    vol = det.sum() / 6.0
    com = ((a + b + c) * det[:, None]).sum(0) / (24.0 * vol)
    # second moments about origin
    def f(u, v, w):  # sum over tets of integral x_i x_j
        return det * ((u*u + v*v + w*w) + (u*v + v*w + w*u) * 0 )  # placeholder replaced below
    # canonical formulas
    xx = det * (a[:,0]**2 + b[:,0]**2 + c[:,0]**2 + a[:,0]*b[:,0] + b[:,0]*c[:,0] + c[:,0]*a[:,0]) / 60.0
    yy = det * (a[:,1]**2 + b[:,1]**2 + c[:,1]**2 + a[:,1]*b[:,1] + b[:,1]*c[:,1] + c[:,1]*a[:,1]) / 60.0
    zz = det * (a[:,2]**2 + b[:,2]**2 + c[:,2]**2 + a[:,2]*b[:,2] + b[:,2]*c[:,2] + c[:,2]*a[:,2]) / 60.0
    xy = det * (2*(a[:,0]*a[:,1] + b[:,0]*b[:,1] + c[:,0]*c[:,1]) + a[:,0]*b[:,1] + b[:,0]*a[:,1] + b[:,0]*c[:,1] + c[:,0]*b[:,1] + c[:,0]*a[:,1] + a[:,0]*c[:,1]) / 120.0
    yz = det * (2*(a[:,1]*a[:,2] + b[:,1]*b[:,2] + c[:,1]*c[:,2]) + a[:,1]*b[:,2] + b[:,1]*a[:,2] + b[:,1]*c[:,2] + c[:,1]*b[:,2] + c[:,1]*a[:,2] + a[:,1]*c[:,2]) / 120.0
    zx = det * (2*(a[:,2]*a[:,0] + b[:,2]*b[:,0] + c[:,2]*c[:,0]) + a[:,2]*b[:,0] + b[:,2]*a[:,0] + b[:,2]*c[:,0] + c[:,2]*b[:,0] + c[:,2]*a[:,0] + a[:,2]*c[:,0]) / 120.0
    Sxx, Syy, Szz, Sxy, Syz, Szx = xx.sum(), yy.sum(), zz.sum(), xy.sum(), yz.sum(), zx.sum()
    I0 = density * np.array([[Syy+Szz, -Sxy, -Szx], [-Sxy, Sxx+Szz, -Syz], [-Szx, -Syz, Sxx+Syy]])
    m = density * vol
    r = com
    I_com = I0 - m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return m, com, I_com

if __name__ == '__main__':
    # self-test: unit cube density 1 -> mass 1, com 0.5, I = 1/6
    c = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], float)
    faces = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    tris = c[np.array(faces)]
    m, com, I = mass_properties(tris, 1.0)
    print('cube self-test: mass', round(m,6), 'com', np.round(com,6), 'I diag', np.round(np.diag(I),6), 'offdiag max', np.abs(I - np.diag(np.diag(I))).max())
