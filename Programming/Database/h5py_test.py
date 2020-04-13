import time
import h5py
import numpy as np
import torch

def generate_numpy(I, d):
    x = np.random.random_sample([I, d])
    dx = np.random.random_sample([I, I, d, 3])
    rdx = np.random.random_sample([I, d, 6])
    return x, dx, rdx

def generate_torch(I1, I2, d):
    x = {"Si": torch.rand([I1, d]),
         "O": torch.rand([I2, d])}
    dx = {"Si": torch.rand([I1, I1+I2, d, 3]),
          "O": torch.rand([I2, I1+I2, d, 3])}
    rdx = {"Si": torch.rand([I1, d, 6]),
           "O": torch.rand([I2, d, 6])}
    return x, dx, rdx

f = h5py.File("test.h5", "w")
norm = f.create_group("norm")
x = norm.create_group("x")
dx = norm.create_group("dx")
rdx = norm.create_group("rdx")

arr = [63, 64]

t0 = time.time()
for i in range(219):
    np.random.shuffle(arr)
    d = generate_numpy(arr[0], 30)
    x.create_dataset(str(i), data=d[0])
    dx.create_dataset(str(i), data=d[1])
    rdx.create_dataset(str(i), data=d[2])
t1 = time.time()
print("Saving numpy time: ", t1-t0)

nnorm = f.create_group("nnorm")
_x = nnorm.create_group("x")
_dx = nnorm.create_group("dx")
_rdx = nnorm.create_group("rdx")

t0 = time.time()
for i in range(219):
    d = generate_torch(40, 20, 30)
    
    xnum = _x.create_group(str(i))
    dxnum = _dx.create_group(str(i))
    rdxnum = _rdx.create_group(str(i))

    for k, v in d[0].items():
        xnum.create_dataset(k, data=v)
    
    for k, v in d[1].items():
        dxnum.create_dataset(k, data=v)

    for k, v in d[2].items():
        rdxnum.create_dataset(k, data=v)
t1 = time.time()
print("Saving torch dict time: ", t1-t0)

f.close()

# Reading
f = h5py.File("test.h5", "r")
t0 = time.time()
for i in range(219):
    _x = f.get(f"nnorm/x/{i}")
    _dx = f.get(f"nnorm/dx/{i}")
    _rdx = f.get(f"nnorm/rdx/{i}")

    x, dx, rdx = {}, {}, {}
    for e in ["Si", "O"]:
        x[e] = torch.DoubleTensor(_x.get(e))
        dx[e] = torch.DoubleTensor(_dx.get(e))
        rdx[e] = torch.DoubleTensor(_rdx.get(e))

t1 = time.time()
print("Loading torch dict time : ", t1-t0)

f.close()
