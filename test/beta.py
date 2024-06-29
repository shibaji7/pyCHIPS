import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import beta

np.random.seed(0)

x = np.linspace(0, 1, 100)
fig = plt.figure(dpi=300, figsize=(8, 3))

ax = fig.add_subplot(121)
samples = np.random.beta(5, 3, size=1000)
ax.hist(samples, bins=20, density=True, histtype="step", color="r", label=r"$x_\tau$")
ax.set_ylabel(r"$B(x_\tau|\alpha,\beta)$")
ax.set_xlabel(r"$x_\tau$")
ax.plot(x, beta.pdf(x, 4.9, 3.1),
       "b-", lw=1, alpha=0.6, label="Fitted PDF")
ax.set_xlim(0,1)
ax.set_ylim(0,4)
ax.axvline(0.5, color="k", ls="--", lw=0.5)
p = np.trapz(beta.pdf(x[x>0.5], 4.9, 3.1), x[x>0.5])
ax.fill_between(x[x>0.5], 0, beta.pdf(x[x>0.5], 4.9, 3.1), color="b", alpha=0.2)
ax.text(0.1, 0.9, rf"(a) $\theta$={'%.3f'%p}", ha="left", va="center", transform=ax.transAxes)
ax.legend(loc=1)

ax = fig.add_subplot(122)
samples = np.random.beta(4, 8, size=1000)
ax.hist(samples, bins=20, density=True, histtype="step", color="r")
ax.set_xlabel(r"$x_\tau$")
ax.set_xlim(0,1)
ax.set_ylim(0,4)
ax.axvline(0.5, color="k", ls="--", lw=0.5)
ax.plot(x, beta.pdf(x, 4.05, 8.03),
       "b-", lw=1, alpha=0.6, label="Fitted PDF")
ax.fill_between(x[x>0.5], 0, beta.pdf(x[x>0.5], 4.05, 8.03), color="b", alpha=0.2)
p = np.trapz(beta.pdf(x[x>0.5], 4.05, 8.03), x[x>0.5])
ax.text(0.1, 0.9, rf"(b) $\theta$={'%.3f'%p}", ha="left", va="center", transform=ax.transAxes)
fig.savefig("tmp/CHIPS.png", bbox_inches="tight")
