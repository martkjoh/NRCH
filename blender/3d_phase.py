import numpy as np
from numpy import sqrt

N = 500
th = 1
k = sqrt(th)

def list_to_data(x, y, z):
    n = np.shape(x)[0]
    data = []
    for i in range(n):
        for j in range(n):
            data.append((x[i, j], y[i, j], z[i, j]))

    return data


def faces_from_data(data, n, mask):
    indices = np.arange(len(data)).reshape(n, n)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            if mask[i, j]:
                continue
            v1 = indices[i][j]
            v2 = indices[i][j+1]
            v3 = indices[i+1][j+1]
            v4 = indices[i+1][j]
            faces.append([v1, v2, v3, v4])

    return faces


def save(data, edges, faces, name):
    np.savetxt('data'+name+'.csv', data, delimiter=',')
    np.savetxt('edges'+name+'.csv', edges, delimiter=',', fmt='%i')
    np.savetxt('faces'+name+'.csv', faces, delimiter=',', fmt='%i')
    

def spin():
    name='spinodal'

    x0 = np.linspace(-k, k, N)
    r0 = np.linspace(-th, 0.001, N)
    x, r = np.meshgrid(x0, r0)

    a = np.sqrt(x**2**2 - (r + 2*x**2)**2 + 0j).real
    a[0:2]=0
    edge = np.zeros_like(a, dtype=bool)
    for i in (
        (0, 1), (0, -1), (1, 0), (-1, 0), 
        (1, 1), (1, -1), (-1, 1), (-1, -1), 
        (0, 2), (0, -2), (2, 0), (-2, 0)
        ):
        edge = edge | np.roll(a!=0, i, (0, 1))

    mask = edge==0
    mask[0:2] = True

    data = list_to_data(x, r, a)
    faces = faces_from_data(data, N, mask)

    save(data, [], faces, name)


def stab():
    name = 'stability'
    z = np.linspace(0, th/2 + 0.2, N)
    x = np.linspace(-k/sqrt(2), k/sqrt(2), N)
    z, x = np.meshgrid(z, x)
    y = - 2 * x**2

    mask = z<x**2

    data = list_to_data(x, y, z)
    faces = faces_from_data(data, N, mask)

    save(data, [], faces, name)


def exc():
    name = 'exceptional'
    x = np.linspace(-k/sqrt(2), k/sqrt(2), N)
    y = np.linspace(0, -th, N)
    x, y = np.meshgrid(x, y)
    z = x**2
    mask = np.zeros_like(x)

    data = list_to_data(x, y, z)
    faces = faces_from_data(data, N, mask)

    save(data, [], faces, name)

def crit_exc():
    name = 'CEL'
    t = np.linspace(-k/sqrt(2), k/sqrt(2), N)
    x, y, z = t, -2*t**2, t**2
    data = [(x[i], y[i], z[i]) for i in range(N)]
    edges = [(i, i+1) for i in range(N-1)]

    save(data, edges, [], name)


spin()
stab()
exc()
crit_exc()