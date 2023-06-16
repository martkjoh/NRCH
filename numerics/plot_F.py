import numpy as np
import matplotlib.pyplot as plt

from loadfiles import *


folder = "data/sym/"
folder_vid = "numerics/vid/"


fnames = get_all_filenames_in_folder(folder)
filename = fnames[0]
filename = filename[:-4]

param = param_from_filename(filename)


phit, xit, param = load_file(folder, filename)

skip = 50
phit = phit[skip:]
xit = xit[skip:]
u, a, b, phibar, N, dt = param
M = len(phit)
T = M * dt
t = np.linspace(0, T + dt, M)
L = 10.
dx = L / N
D2 = lambda J : ( np.roll(J, 2, axis=-1) + np.roll(J, -2, axis=-1) - 2 * J ) / (2*dx)**2 
D = lambda J : (np.roll(J, 1, axis=-1) - np.roll(J, -1, axis=-1) ) / (2 * dx)


aeps = a* np.array([[0, 1], [-1, 0]])

p2 =  np.sum(phit**2, axis=1)
dF = u * (-1 + p2 )[:, None, :] * phit - D2(phit)
mut = np.einsum("ij,tjn->tin", aeps, phit)
mu = (dF + mut)[:-1] #+ D(xit) / 20_000
fdot = np.sum(D2(mu) * dF[:-1], axis=1)
Fdot = np.sum(fdot, axis=1) * dx

f1 = u * (- p2 / 2 + p2**2 / 4 ) - np.sum(phit * D2(phit), axis=1 )/ 2
F1 = np.sum(f1, axis=1) * dx

F2 = np.concatenate([[F1[0],], Fdot*dt])
F2 = np.cumsum(F2)


fig, ax = plt.subplots()

 
ax.plot([0, T], [0, 0], "k--")

dFdt = np.diff(F1) / np.diff(t)
ax.plot((t[:-1] + t[1:])/2, dFdt, ls="--")
ax.plot(t[:-1], Fdot, ls="--", alpha=.2)


ax2 = plt.twinx(ax)
ax2.plot(t, F1, label="F actual")
ax2.plot(t, F2, label="F integrated")
# ax.set_xlim(-.1, 10)

plt.legend()
plt.show()
