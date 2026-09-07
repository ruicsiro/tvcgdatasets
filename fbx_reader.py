# -*- coding: utf-8 -*-
"""Minimal reader for binary FBX: extracts Vertices and PolygonVertexIndex.

Only what is needed to rebuild a triangle mesh -- no materials, no scene graph.
"""
import struct, zlib, numpy as np


def _read_props(buf, off, nprops):
    props = []
    for _ in range(nprops):
        t = chr(buf[off]); off += 1
        if t == 'Y':
            props.append(struct.unpack_from('<h', buf, off)[0]); off += 2
        elif t == 'C':
            props.append(bool(buf[off])); off += 1
        elif t == 'I':
            props.append(struct.unpack_from('<i', buf, off)[0]); off += 4
        elif t == 'F':
            props.append(struct.unpack_from('<f', buf, off)[0]); off += 4
        elif t == 'D':
            props.append(struct.unpack_from('<d', buf, off)[0]); off += 8
        elif t == 'L':
            props.append(struct.unpack_from('<q', buf, off)[0]); off += 8
        elif t in 'fdlib':
            n, enc, clen = struct.unpack_from('<III', buf, off); off += 12
            raw = buf[off:off + clen]; off += clen
            if enc == 1:
                raw = zlib.decompress(raw)
            fmt = {'f': 'f4', 'd': 'f8', 'l': 'i8', 'i': 'i4', 'b': 'i1'}[t]
            props.append(np.frombuffer(raw, dtype='<' + fmt, count=n))
        elif t in 'SR':
            ln = struct.unpack_from('<I', buf, off)[0]; off += 4
            props.append(buf[off:off + ln]); off += ln
        else:
            raise ValueError('unknown FBX property type %r' % t)
    return props, off


def parse(path):
    buf = open(path, 'rb').read()
    assert buf[:20] == b'Kaydara FBX Binary  ', 'not a binary FBX'
    version = struct.unpack_from('<I', buf, 23)[0]
    wide = version >= 7500
    hdr = '<QQQB' if wide else '<IIIB'
    hsz = 25 if wide else 13
    found = {}

    def walk(off, end):
        while off < end:
            if off + hsz > len(buf):
                return
            endoff, nprops, plen, namelen = struct.unpack_from(hdr, buf, off)
            if endoff == 0:
                return
            off2 = off + hsz
            name = buf[off2:off2 + namelen].decode('utf-8', 'replace')
            off2 += namelen
            props, after = _read_props(buf, off2, nprops)
            if name in ('Vertices', 'PolygonVertexIndex') and props:
                found.setdefault(name, []).append(props[0])
            if after < endoff:
                walk(after, endoff)
            off = endoff
    walk(27, len(buf))
    return version, found


def to_mesh(path):
    """Return (vertices Nx3 float64, faces Mx3 int) triangulating FBX polygons."""
    version, f = parse(path)
    if 'Vertices' not in f:
        raise ValueError('no Vertices in %s' % path)
    V = np.concatenate([np.asarray(v, dtype=np.float64) for v in f['Vertices']])
    V = V.reshape(-1, 3)
    faces = []
    if 'PolygonVertexIndex' in f:
        idx = np.concatenate([np.asarray(i) for i in f['PolygonVertexIndex']])
        poly = []
        for v in idx:
            if v < 0:
                poly.append(int(~v))
                for k in range(1, len(poly) - 1):      # fan triangulation
                    faces.append([poly[0], poly[k], poly[k + 1]])
                poly = []
            else:
                poly.append(int(v))
    return V, np.asarray(faces, dtype=np.int64)


if __name__ == '__main__':
    import sys, os
    base = r"./study_data"   # directory holding the study mesh assets
    tests = [os.path.join(base, 'rock files', 'xr', 'Rock_05_05_04_world.fbx'),
             os.path.join(base, 'OriginalSuperquadricMesh_1.fbx'),
             os.path.join(base, 'OriginalSuperquadricMesh_2.fbx')] + \
            [os.path.join(base, 'ValidateMesh_%d.fbx' % i) for i in range(1, 6)]
    for p in tests:
        try:
            ver, f = parse(p)
            V, F = to_mesh(p)
            print('%-34s v%d  verts=%-6d faces=%-6d  bbox=%s'
                  % (os.path.basename(p), ver, len(V), len(F),
                     np.round(V.max(0) - V.min(0), 3)))
        except Exception as e:
            print('%-34s FAILED: %s' % (os.path.basename(p), e))
