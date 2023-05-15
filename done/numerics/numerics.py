import numpy as np

from numpy import pi, random
from numba import njit, objmode

fast = True
if fast: jit = njit
else: jit = lambda x : x

N = 100
M = 4_000_000

L = 100
dx = L / N
dt = .04 * (dx)**4
frames = 100
skip = M // frames

print('dt/dx^4 = %.3f'%(dt/dx**4))
print('T = %.3f'%(M*dt))

b = .02
b = b * (N/100)**2

param_names = ['N', 'M', 'L', 'dt', 'b', 'r', 'phibar', 'a']

# Initial conditions
A = .2
k = 1


i = np.arange(N)
D2 = np.zeros((N, N))
D2[i, i] = - 1 / dx**2
D2[i, (i+1)%N] = 1 / (2*dx**2)
D2[(i+1)%N, i] = 1 / (2*dx**2)


@jit
def mu(phi, param):
    N, M, L, dt, b, r, phibar, a = param
    m = ( r + (phi[: ,0]**2 + phi[:, 1]**2) )[:,None] * phi
    m += - D2@phi
    m[:, 0] += a *phi[:, 1]
    m[:, 1] -= a *phi[:, 0]

    u1 = np.random.random((N))
    u2 = np.random.random((N))
    z1 = np.sqrt(- 2 * np.log(u1)) * np.cos(u2)
    z2 = np.sqrt(- 2 * np.log(u1)) * np.sin(u2)
    m[:, 0] += z1 * b
    m[:, 0] += z2 * b

    return m

def F(phi, param):
    N, M, L, dt, b, r, phibar, a = param
    p2 = phi[: ,0]**2 + phi[:, 1]**2
    dp = D2@phi
    dp2 = dp[:, 0]**2 + dp[:, 1]**2
    return 1/2*r*p2 + 1/2*dp2 + p2**2 / 4

@jit
def f(phi, param):
    return D2 @ mu(phi, param)

@jit
def get_x_phi(param):
    N, M, L, dt, b, r, phibar, a = param
    phi = np.zeros((N, 2))
    x = np.linspace(0, L - dx, N)
    phi[:, 0] = A * np.sin(2*pi*x/L*k) + phibar
    phi[:, 1] = A * np.cos(2*pi*x/L*k)
    return x, phi

@jit
def loop(phit, param, phi):
    n1 = M//skip
    n2 = n1//10
    for i in range(1, M//skip):
        if ((i+1)//n2) - i//n2 == 1:
            with objmode(): print("|", end='', flush=True)
        for j in range(0, skip):
            phi = phi + f(phi, param) * dt
        phit[i] = phi
    print('')

def run_euler(param):
    x, phi = get_x_phi(param)

    phit = np.empty((M//skip, N, 2))
    phit[0] = phi

    loop(phit, param, phi)

    write_file(phit, param)

def filename_from_param(param):
    return ''.join(param_names[i] + '=' + str(param[i]) + '_' for i in range(len(param_names)))[:-1]

def write_file(phit, param):
    N, M, L, dt, b, r, phibar, a = param
    filename = filename_from_param(param)
    np.savetxt('data/'+filename+'.txt', phit.reshape((frames, 2*N)), comments='')





r, phibar, a = -1, -.8, .11
param = N, M, L, dt, b, r, phibar, a
run_euler(param)
# make_anim(param)
# print(D2)
# ps = [-.95, -.83, -0.73, -.55]
# aa = [0, .2, .5, .55]
# param = [[-1, p, a] for p in ps for a in aa]
# param.pop(15)
# param.pop(10)
# param.pop(7)
# param.pop(3)
# param.append([-1, -1/np.sqrt(2), .5])


# from multiprocessing import Pool
# with Pool(len(param)) as pool:
#     pool.starmap(make_anim, param)


plot_all()