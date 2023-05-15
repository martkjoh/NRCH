from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

import os

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)


param_names = ['N', 'M', 'L', 'dt', 'b', 'r', 'phibar', 'a']

def filename_from_param(param):
    return ''.join(param_names[i] + '=' + str(param[i]) + '_' for i in range(len(param_names)))[:-1]

def get_all_filenames_in_folder(folder_path):
    """
    Returns a list of all filenames of all files in a folder.
    """
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filenames.append(filename)
    return filenames

def param_dict_from_filename(s):
    """
    Extracts parameters from a string in the format 'value1=xxx_value2=yyy...' and
    returns a dictionary in the format {'value1': xxx, 'value2': yyy, ...}.
    """
    params = {}
    for param in s.split('_'):
        key, value = param.split('=')
        params[key] = float(value)
    return params

def param_from_dict(param_dict):
    return [param_dict[key] for key in param_names]

def param_from_filename(filename):
    N, M, L, dt, b, r, phibar, a =  param_from_dict(param_dict_from_filename(filename))
    N, M = int(N), int(M)
    param = (N, M, L, dt, b, r, phibar, a)
    return param



def load_file(filename):
    file = 'data/'+filename+'.txt'
    param = param_from_filename(filename)
    N, M, L, dt, b, r, phibar, a = param
    phit = np.loadtxt(file)
    frames = len(phit)
    phit = phit.reshape((frames, N, 2))
    x = np.linspace(0, L, N)
    return x, phit, param

def make_anim(filename):
    x, phit, param = load_file(filename)
    N, M, L, dt, b, r, phibar, a = param

    fig, ax = plt.subplots()
    pt = np.einsum('txi->ti',phit)
    dpt = (pt[1:] - pt[:-1])/dt
    frames = len(phit)
    t = np.linspace(0, frames*dt, frames-1)
    ax.plot(t, dpt[:,0], label="$\\frac{\\mathrm{d} \\bar \\varphi_1}{\\mathrm{d} t}$")
    ax.plot(t, dpt[:,1], label="$\\frac{\\mathrm{d} \\bar \\varphi_2}{\\mathrm{d} t}$")
    ax.legend()
    print(np.max(dpt))


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(filename_from_param(param))
    l1, = ax[0].plot([], [], 'r-', label='$\\varphi_1$')
    l2, = ax[0].plot([], [], 'k-', label='$\\varphi_2$')
    l3, = ax[0].plot([], [], 'g-.')
    l4, = ax[0].plot([], [], 'b')
    ax[0].plot([0, L], [phibar, phibar], 'r--')
    ax[0].plot([0, L], [0, 0], 'k--')
    ax[0].set_xlim(0, L)
    ax[0].set_ylim(-2, 2)
    ax[0].legend()
    

    tt = np.linspace(0, L, 1000)
    ax[0].plot(tt, (1 + phibar)*np.cos(2*tt/L*2*np.pi) + phibar)
    ax[0].plot(tt, 2*np.sqrt(-phibar-phibar**2)*np.cos(tt/L*2*np.pi))

    m1, = ax[1].plot([], [], 'r-..')
    ax[1].plot(0, phibar, 'ro')
    t = np.linspace(0, 2*pi)
    ax[1].plot(np.cos(t), np.sin(t), 'k--')
    prange = 1.5
    ax[1].set_xlim(-prange, prange)
    ax[1].set_ylim(-prange, prange)
    l5 = ax[0].text(1, 1.8, 'progress:')

    def animate(m):
        n2 = frames//10
        txt = 'progress:' + (m+1)//n2*'|'
        l5.set_text(txt)

        p = phit[m].T
        l1.set_data(x, p[0])
        l2.set_data(x, p[1])
        p2 = np.sqrt( p[0]**2 + p[1]**2 ) 
        # l3.set_data(x, F(phit[m], param))
        # l4.set_data(x, mu(phit[m], param)[:, 1])

        m1.set_data([*p[1], p[1, 0]], [*p[0], p[0, 0]])

        return l1, l2, l3, l4, l5, m1

    anim = animation.FuncAnimation(fig, animate,  interval=100, frames=frames)
    FFwriter = animation.FFMpegWriter()
    plt.show()
    # anim.save('done/fig/plot'+name+'.mp4', writer=FFwriter)


def plot_all():
    fnames = get_all_filenames_in_folder("data/")
    [make_anim(filename[:-4]) for filename in fnames ]




plot_all()