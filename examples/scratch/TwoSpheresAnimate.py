import torch
import numpy as np
import math
from matplotlib import pyplot as plt

import kaolin as kal
from kaolin import _C
import kaolin.ops.spc as spc


device = 'cuda'


# In order to avoid 'holes', we start at level 6
# This may perform worse than starting at level 4 or 5
start_level = 6


points = spc.morton_to_points(torch.arange(pow(8, start_level)).to(device))

# t is the animation control parameter t \in [0,0.5] is a good range. 
# t = 0.25*(math.cos(time)+1.0) # should work well w.r.t acceleration/deceleration
# 
# t = 0.0
t = 0.3125 
# t = 0.3225
# t = 0.5

level = 11
for l in range(start_level, level+1):
    occupancies = _C.ops.spc.contains(points, l, t)
    insum = _C.ops.spc.inclusive_sum(occupancies)
    if l == level:
        points = _C.ops.spc.compactify(points, insum)
    else:
        points = _C.ops.spc.subdivide(points, insum)
    print(l, points.size(0))

# compute normals for surviving points
normals = _C.ops.spc.custom_normals(points, level, t)

# find octree
octree = kal.ops.spc.unbatched_points_to_octree(points, level)

# store in file if you want
# np.savez("/home/cloop/src/Work/Data/VolumeSparsify/quat", octree=octree.to('cpu'), normals=normals.to('cpu'))


# The rest is raytracing

lengths = torch.tensor([len(octree)], dtype=torch.int32)

level, pyramid, exsum = spc.scan_octrees(octree, lengths)
points = spc.generate_points(octree, pyramid, exsum)


pyramid = pyramid.squeeze() #undo batching

colors = points[pyramid[1,level]:].to('cpu')/pow(2,level)


w = 1024
h = 1024

eye = torch.tensor([0.0, 0.0, 8.0])
at = torch.tensor([-1.0, -1.0, -1.0])
up = torch.tensor([0.0, 1.0, 0.0])
fov = math.pi / 4.0
world = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

org, dir = _C.render.spc.generate_primary_rays(w, h, eye, at, up, fov, world)

o = _C.render.spc.raytrace(octree, points, pyramid, exsum, org, dir, level)
r = _C.render.spc.remove_duplicate_rays(o).cpu().numpy()


# display image using opencv
img = np.ones((h, w, 3))  
num = r.shape[0]
for i in range(num):
    ridx = r[i, 0]
    vidx = r[i, 1] - pyramid[1,level]

    y = ridx % w
    x = math.floor(ridx / w)

    clr = colors[vidx]
    # clr = np.array([0.0, 0.0, 0.0])
    img[x, y] = clr

    

plt.imshow(img)
plt.waitforbuttonpress()


print('done')




