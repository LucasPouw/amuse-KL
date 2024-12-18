import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from amuse.units import units, constants
from amuse.io import read_set_from_file
import argparse
import os
import glob
from amuse.ext.orbital_elements import get_orbital_elements_from_binaries
from amuse.lab import Particles


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


# def get_com(orbiter):
#     if len(orbiter) == 2:
#         com = (orbiter[0].position * orbiter[0].mass + orbiter[1].position * orbiter[1].mass) / (orbiter[0].mass + orbiter[1].mass)
#     else: #single body
#         com = orbiter.copy().position
#     return com


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

    """
    Requires axes to be a shape-(3,) array

    Plot is centered on the SMBH    
    """

    smbh = particle_set[particle_set.name == 'SMBH']
    stars = particle_set[ np.logical_or(particle_set.name == 'primary_star', particle_set.name == 'secondary_star') ]
    # stars = particle_set[particle_set.name == 'disk']

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

    return


def plot_stars(axes, particle_set, lim=3):

    "Requires axes to be a shape-(3,) array"

    primary = particle_set[particle_set.name == 'primary_star']
    secondary = particle_set[particle_set.name == 'secondary_star']

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


def plot_binary_disk_arrow(axes, particle_set, lim=30):

    "Requires axes to be a shape-(3,) array"

    ax_xy, ax_yz, ax_xz = axes

    particle_set.position -= particle_set[particle_set.name == 'primary_star'].position  # Center on primary

    disk = particle_set[particle_set.name == 'disk']
    ax_xy.scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1)
    ax_yz.scatter(disk.y.value_in(units.AU), disk.z.value_in(units.AU), s=1)
    ax_xz.scatter(disk.x.value_in(units.AU), disk.z.value_in(units.AU), s=1)
    plot_stars(axes, particle_set, lim)

    smbh = particle_set[particle_set.name == 'SMBH']
    arrow_to_smbh = (smbh.position / smbh.position.length())[0]  # Normalize arrow to length 1

    slice_dict = {0:[0,1], 1:[1,2], 2:[0,2]} # Able to extract xy, yz and xz plane
    for i, axis in enumerate(axes):

        ymin, ymax = axis.get_ylim()
        yrange = ymax - ymin

        # com_plot = data_to_axis(com.value_in(units.AU)[slice_dict[i]]) #get the relevant com coordinates
        com_plot = (0,0)

        # This sets the scale of the arrowhead, and it is now directly depedent on the yscale of the axis
        # which may mess up in the future so if it does, look here
        if i != 0:
            arrowhead = arrow_to_smbh[slice_dict[i]] * 0.25 * yrange
        else:
            arrowhead = arrow_to_smbh[slice_dict[i]] * 0.5 * yrange


        axis.annotate("", xy=arrowhead, xytext=com_plot,
                      arrowprops=dict(arrowstyle="->"),xycoords='data',textcoords='data'
                      )
        
        ##Uncomment if you want to plot the com and arrowhead as seperate scatters
        # axis_to_data = axis.transLimits.inverted().transform
        # print(axis_to_data(com_plot))
        # print(axis_to_data((0.5,0.5)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make relevant plots from HDF5 files')
    parser.add_argument('--file_dir',type=str,default='./snapshots-test/',help='Directory where HDF5 files are stored.')
    parser.add_argument('--image_dir',type=str,default='./images-test/',help='Directory where plots will be stored.')
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        os.mkdir(args.image_dir)
    else:  # Empty out image directory
        if len(os.listdir(args.image_dir)) != 0: 
            print(f'Found existing image(s) in {args.image_dir}, removing them...')
            files = glob.glob(args.image_dir + '/*.png')
            for f in files:
                os.remove(f)


    unsorted_datafiles = glob.glob(args.file_dir + '/*.hdf5')
    file_numbers = [filename.split('_')[-1].split('.')[0] for filename in unsorted_datafiles]
    file_numbers = list(map(int, file_numbers))
    times, datafiles = zip(*sorted(zip(file_numbers, unsorted_datafiles)))
    times = np.array(times) / 365  # In years
    times = times.astype(int)  

    # datafiles=[args.file_dir + 'snapshot_0.hdf5']
    median_incs = []
    median_eccs = []
    plot_time = []
    for time, datafile in zip(times, datafiles):
        data = read_set_from_file(datafile)  # Full particle set at single timestep

        stars = data[ np.logical_or(data.name == 'primary_star', data.name == 'secondary_star') ]
        com = stars[0]  # THIS IS JUST BECAUSE I WAS TOO LAZY TO GET THE ACTUAL COM AS A PARTICLE!!!!! CHANGE IT ASAP!!!
        disk = data[data.name == 'disk']

        _, _, _, eccs, _, incs, _, _ = get_orbital_elements_from_binaries(com, disk, G=constants.G)

        median_incs.append(np.mean(incs.value_in(units.deg)))
        median_eccs.append(np.mean(eccs))
        plot_time.append(time)

        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
        plot_binary_disk_arrow(np.array([ax1, ax2, ax3]), data)
        plot_inc_ecc(ax4, 
                     plot_time | units.yr, 
                     median_incs | units.deg,
                     median_eccs)
        fig.suptitle(f'Time = {time} year', fontsize=BIGGER_SIZE)
        plt.tight_layout()
        # ax4.set_aspect('equal')

        fig.savefig(f'{args.image_dir}binary-with-arrow-{datafile.split('_')[-1].split('.')[0]}.png', bbox_inches='tight')
        

        # gs = gridspec.GridSpec(4, 4)
        # fig = plt.figure(figsize=(10,10))
        # ax1 = fig.add_subplot(gs[:2, :2])
        # ax2 = fig.add_subplot(gs[:2, 2:])
        # ax3 = fig.add_subplot(gs[2:4, 1:3])
        # plot_smbh_and_stars(np.array([ax1, ax2, ax3]), data)
        # fig.suptitle(f'Time = {time.value_in(units.yr):.0f} year', fontsize=BIGGER_SIZE)
        # plt.tight_layout()

        # fig.savefig(f'{args.image_dir}smbh-and-stars-{datafile.split('_')[-1].split('.')[0]}.png', bbox_inches='tight')


        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 9.5))
        # plot_stars(np.array([ax1, ax2, ax3]), data, lim=1)
        # plot_inc_ecc(ax4, np.linspace(0, 1, 100) | units.yr, np.linspace(0, 100, 100) | units.deg, np.linspace(0, 1, 100))
        # fig.suptitle(f'Time = {time.value_in(units.yr):.0f} year', fontsize=BIGGER_SIZE)
        # plt.tight_layout()
        # # ax4.set_aspect('equal')

        # fig.savefig(f'{args.image_dir}binary-{datafile.split('_')[-1].split('.')[0]}.png', bbox_inches='tight')
