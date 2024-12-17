import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from amuse.units import units
from amuse.io import read_set_from_file

# We define some properties for the figures
import matplotlib as mpl
SMALL_SIZE = 10 * 2 
MEDIUM_SIZE = 12 * 2
BIGGER_SIZE = 14 * 2

# plt.rc('text', usetex=True)
plt.rc('axes', titlesize=SMALL_SIZE)                     # fontsize of the axes title\n",
plt.rc('axes', labelsize=MEDIUM_SIZE)                    # fontsize of the x and y labels\n",
plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')   # fontsize of the tick labels\n",
plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')   # fontsize of the tick labels\n",
plt.rc('legend', fontsize=SMALL_SIZE)                    # legend fontsize\n",
mpl.rcParams['axes.titlesize'] = BIGGER_SIZE
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXgeneral'

mpl.rcParams['figure.dpi'] = 100

mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.size'] = 4

mpl.rcParams['xtick.major.width'] = 1.25
mpl.rcParams['ytick.major.width'] = 1.25
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1


def plot_inc_ecc(ax, time, inc, ecc):
    ecc_ax = ax
    inc_ax = ax.twinx()

    ecc_ax.plot(time.value_in(units.yr), ecc, color='red', label='Eccentricity')
    plt.plot(time.value_in(units.yr), inc.value_in(units.deg), color='blue', label='Inclination')

    ecc_ax.set_xlabel('Time [yr]')
    ecc_ax.set_ylabel('Eccentricity', color='red')
    ecc_ax.tick_params(axis='y', labelcolor='red')
    ecc_ax.set_ylim(0,1)

    inc_ax.set_ylabel('Inclination [deg]', color='blue')
    inc_ax.tick_params(axis='y', labelcolor='blue')
    inc_ax.set_ylim(0, 180)
    return


def plot_smbh_and_stars(axes, particle_set, lim=10):
    smbh = particle_set[particle_set.name == 'SMBH']
    stars = particle_set[ np.logical_or(particle_set.name == 'primary', particle_set.name == 'secondary') ]

    ax_xy, ax_yz, ax_xz = axes

    for ax in axes.flatten():
        ax.scatter(0, 0, marker='o', s=75, color='black', label='Sgr A*')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')

    ax_xy.scatter(stars.position.x.value_in(1e3 * units.AU) - smbh.position.x.value_in(1e3 * units.AU), 
                  stars.position.y.value_in(1e3 * units.AU) - smbh.position.y.value_in(1e3 * units.AU),
                  marker='o', color='red', label='Star(s)')
    ax_xy.set_xlabel(r'$x \, \left[10^3 \, \mathrm{AU} \right]$')
    ax_xy.set_ylabel(r'$y \, \left[10^3 \, \mathrm{AU} \right]$')
    ax_yz.scatter(stars.position.y.value_in(1e3 * units.AU) - smbh.position.y.value_in(1e3 * units.AU), 
                  stars.position.z.value_in(1e3 * units.AU) - smbh.position.z.value_in(1e3 * units.AU),
                  marker='o', color='red')
    ax_yz.set_xlabel(r'$y \, \left[10^3 \, \mathrm{AU} \right]$')
    ax_yz.set_ylabel(r'$z \, \left[10^3 \, \mathrm{AU} \right]$')
    ax_xz.scatter(stars.position.x.value_in(1e3 * units.AU) - smbh.position.x.value_in(1e3 * units.AU), 
                  stars.position.z.value_in(1e3 * units.AU) - smbh.position.z.value_in(1e3 * units.AU),
                  marker='o', color='red')
    ax_xz.set_xlabel(r'$x \, \left[10^3 \, \mathrm{AU} \right]$')
    ax_xz.set_ylabel(r'$z \, \left[10^3 \, \mathrm{AU} \right]$')
    
    ax_xy.legend(loc='upper left')    


def plot_stars(axes, particle_set, lim=3):
    primary = particle_set[particle_set.name == 'primary']
    secondary = particle_set[particle_set.name == 'secondary']

    ax_xy, ax_yz, ax_xz = axes

    for ax in axes.flatten():
        ax.scatter(0, 0, marker='o', color='red')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')

    ax_xy.scatter(secondary.position.x.value_in(units.AU) - primary.position.x.value_in(units.AU), 
                  secondary.position.y.value_in(units.AU) - primary.position.y.value_in(units.AU),
                  marker='o', color='red')
    ax_xy.set_xlabel(r'$x \, \left[\mathrm{AU} \right]$')
    ax_xy.set_ylabel(r'$y \, \left[\mathrm{AU} \right]$')
    ax_yz.scatter(secondary.position.y.value_in(units.AU) - primary.position.y.value_in(units.AU), 
                  secondary.position.z.value_in(units.AU) - primary.position.z.value_in(units.AU),
                  marker='o', color='red')
    ax_yz.set_xlabel(r'$y \, \left[\mathrm{AU} \right]$')
    ax_yz.set_ylabel(r'$z \, \left[\mathrm{AU} \right]$')
    ax_xz.scatter(secondary.position.x.value_in(units.AU) - primary.position.x.value_in(units.AU), 
                  secondary.position.z.value_in(units.AU) - primary.position.z.value_in(units.AU),
                  marker='o', color='red')
    ax_xz.set_xlabel(r'$x \, \left[\mathrm{AU} \right]$')
    ax_xz.set_ylabel(r'$z \, \left[\mathrm{AU} \right]$')

    return


if __name__ == '__main__':
    data = read_set_from_file('/data2/pouw/amuse-project/smbh_binary_disk.hdf5')
    print(len(data))

    time = 1 | units.yr

    gs = gridspec.GridSpec(4, 4)
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[:2, 2:])
    ax3 = fig.add_subplot(gs[2:4, 1:3])
    plot_smbh_and_stars(np.array([ax1, ax2, ax3]), data)
    fig.suptitle(f'Time = {time.value_in(units.yr):.0f} year', fontsize=BIGGER_SIZE)
    plt.tight_layout()
    fig.show()
    

    time = 1 | units.yr
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 9.5))
    plot_stars(np.array([ax1, ax2, ax3]), data, lim=1)
    plot_inc_ecc(ax4, np.linspace(0, 1, 100) | units.yr, np.linspace(0, 100, 100) | units.deg, np.linspace(0, 1, 100))
    fig.suptitle(f'Time = {time.value_in(units.yr):.0f} year', fontsize=BIGGER_SIZE)
    plt.tight_layout()
    # ax4.set_aspect('equal')
    fig.show()
