"""Sparse virtual edge queries. Formula documented in edge_penalty.py."""
import warp as wp


@wp.kernel(enable_backward=False)
def query_edge_depth(points: wp.array(dtype=wp.vec3), starts: wp.array(dtype=wp.vec3),
                     ends: wp.array(dtype=wp.vec3), offsets: wp.array(dtype=wp.int32),
                     ids: wp.array(dtype=wp.int32), ox: float, oy: float,
                     nx: int, ny: int, cell: float, radius: float,
                     result: wp.array(dtype=wp.float32)):
    i = wp.tid()
    p = points[i]
    x = int(wp.floor((p[0]-ox)/cell))
    y = int(wp.floor((p[1]-oy)/cell))
    deepest = float(0.)
    if x >= 0 and x < nx and y >= 0 and y < ny:
        c = x*ny+y
        for j in range(offsets[c], offsets[c+1]):
            e = ids[j]
            ab = ends[e]-starts[e]
            length = wp.length(ab)
            axis = ab/length
            rel = p-starts[e]
            along = wp.dot(rel,axis)
            if along >= 0. and along <= length:
                distance = wp.length(rel-along*axis)
                deepest = wp.max(deepest,radius-distance)
    result[i] = deepest
