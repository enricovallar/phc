# %%
E = [
    2.2,
    2.3,
    2.4,
    2.5,
    2.6,
    2.7,
    2.8,
    2.9,
    3.0,
    3.1,
]

alpha = [
    3.12e1,
    7.79e3,
    2.72e4,
    6.43e4,
    1.44e5,
    7.39e5,
    3.35e6,
    5.38e6,
    6.81e6,
    8.64e6,
]

import matplotlib.pyplot as plt
plt.plot(E, alpha)
plt.xlabel('E [eV]')
plt.ylabel(r'$\alpha$')
plt.show()
