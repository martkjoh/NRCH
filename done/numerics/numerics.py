import numpy as np
import matplotlib.pyplot as plt

from numpy import pi, random
from matplotlib import animation
from numba import njit, objmode

fast = True
if fast: jit = njit
else: jit = lambda x : x

N = 150
M = 10_000_000
dt = .000008

L = 10
dx = L / N
skip = 20_000
print('dt/dx^4 = %.3f'%(dt/dx**4))

A = .01
b = 1e4*dt * 1
k = 1
u = 1

i = np.arange(N)
D2 = np.zeros((N, N))
D2[i, i] = - 1 / dx**2
D2[i, (i+1)%N] = 1 / (2*dx**2)
D2[(i+1)%N, i] = 1 / (2*dx**2)


@jit
def mu(phi, param):
    r, phibar, a = param
    m = ( r + u * (phi[: ,0]**2 + phi[:, 1]**2) )[:,None] * phi
    m += - D2@phi
    m[:, 0] += a *phi[:, 1]
    m[:, 1] -= a *phi[:, 0]
    m += (random.random((N, 2))- 1/2) * b
    return m

@jit
def f(phi, param):
    return D2 @ mu(phi, param)

@jit
def get_x_phi(param):
    r, phibar, a = param
    phi = np.zeros((N, 2))
    x = np.linspace(0, L - dx, N)
    phi[:, 0] = A * np.sin(2*pi*x/L*k) + phibar
    phi[:, 1] = A * np.cos(2*pi*x/L*k)
    return x, phi

@jit
def run_euler(param):
    x, phi = get_x_phi(param)
    phit = np.empty((M//skip, N, 2))
    phit[0] = phi

    n1 = M//skip
    n2 = n1//10

    for i in range(1, M//skip):
        if ((i+1)//n2) - i//n2 == 1:
            with objmode(): print("|", end='', flush=True)
        for j in range(0, skip):
            phi = phi + f(phi, param) * dt
        phit[i] = phi
    print('')
    return x, phit


def make_anim(r, phibar, a):
    param = (r, phibar, a)
    name = f'r={r}_phibar={phibar}_a={a}'
    x, phit = run_euler(param)

    fig, ax = plt.subplots()
    pt = np.einsum('txi->ti',phit)
    dpt = (pt[1:] - pt[:-1])/dt
    t = np.linspace(0, M//skip*dt, M//skip-1)
    ax.plot(t, dpt[:,0], label="$\\frac{\\mathrm{d} \\bar \\varphi_1}{\\mathrm{d} t}$")
    ax.plot(t, dpt[:,1], label="$\\frac{\\mathrm{d} \\bar \\varphi_2}{\\mathrm{d} t}$")
    ax.legend()
    print(np.max(dpt))


    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    l1, = ax[0].plot([], [], 'r-')
    l2, = ax[0].plot([], [], 'k-')
    l3, = ax[0].plot([], [], 'g-.')
    l4, = ax[0].plot([], [], 'b')
    ax[0].plot([0, L], [phibar, phibar], 'r--')
    ax[0].plot([0, L], [0, 0], 'k--')
    ax[0].set_xlim(0, L)
    ax[0].set_ylim(-2, 2)

    m1, = ax[1].plot([], [], 'r-..')
    ax[1].plot(0, phibar, 'ro')
    t = np.linspace(0, 2*pi)
    ax[1].plot(np.cos(t), np.sin(t), 'k--')
    prange = 1.5
    ax[1].set_xlim(-prange, prange)
    ax[1].set_ylim(-prange, prange)
    l5 = ax[0].text(1, 1.8, 'progress:')

    def animate(m):

        n1 = M//skip
        n2 = n1//10
        txt = 'progress:' + (m+1)//n2*'|'
        l5.set_text(txt)


        p = phit[m].T
        l1.set_data(x, p[0])
        l2.set_data(x, p[1])
        p2 = np.sqrt( p[0]**2 + p[1]**2 )
        l3.set_data(x, p2)
        A = np.sqrt(-r - (2*np.pi/L*k)**2)
        l4.set_data([0, L], [np.sqrt(A)])

        m1.set_data([*p[1], p[1, 0]], [*p[0], p[0, 0]])

        return l1, l2, l3, l4, l5, m1

    time_per_step = skip * dt
    fpms = 10 * 1000


    anim = animation.FuncAnimation(fig, animate,  interval=100, frames=M//skip)
    FFwriter = animation.FFMpegWriter()
    plt.show()
    anim.save('done/fig/plot'+name+'.mp4', writer=FFwriter)


def test_D():
    x = np.linspace(0, (L-dx), N)
    y = np.sin(2*pi*x)

    plt.plot(x, y)
    D4 = D@D@D@D
    plt.plot(x, D@y)

    plt.show()


def test_D_phi():
    phi = np.zeros((N, 2))
    x = np.linspace(0, L - dx, N)
    phi[:, 0] = np.sin(2*pi*x)
    phi[:, 1] = np.cos(2*pi*x)

    plt.plot(x, phi[:, 0], 'k')
    plt.plot(x, D2@D2@D2@D2@phi[:, 0], 'r--')

    plt.show()


def test_eps():
    phi = np.zeros((N, 2))
    x = np.linspace(0, L - dx, N)
    phi[:, 0] = np.sin(x)
    phi[:, 1] = np.cos(x)

    plt.plot(x, phi[:, 0], 'k')
    plt.plot(x, phi[:, 1], 'k-k-')

    plt.plot(x, ((eps@phi.T).T)[:, 0], 'r')
    plt.plot(x, ((eps@phi.T).T)[:, 1], 'r--')

    plt.show()



make_anim(-1, -1 / np.sqrt(2) + .2, .51)


# aa = [0, .5, .8]
# pb = [0, -.2, -.4, -.6, -.8]

# for a in aa:
#     for p in pb:
#         make_anim(param=(-1, p, a))



ps = [-.9, -.8, -0.73, -.65, -1/np.sqrt(2)]
aa = [.55, .5]
param = [[-1, p, a] for p in ps for a in aa]

from multiprocessing import Pool

# with Pool(10) as pool:
#     pool.starmap(make_anim, param)


