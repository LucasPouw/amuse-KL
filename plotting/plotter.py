## Made by Lucas Pouw, Yannick Badoux and Tim van der Vuurst for the 
## course "Simulations and Modeling in Astrophysics" '24-'25. 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from amuse.units import units, constants
from amuse.io import read_set_from_file
import argparse
import os
import glob
from amuse.ext.orbital_elements import get_orbital_elements_from_binaries
from amuse.lab import Particles, Particle
from amuse.units.quantities import Quantity, ScalarQuantity, VectorQuantity
from tqdm import tqdm   
import matplotlib as mpl

# We define some properties for the figures
SMALL_SIZE = 10 * 2 
MEDIUM_SIZE = 12 * 2
BIGGER_SIZE = 14 * 2

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


def get_com(orbiter):
    """Helper function to find the center of mass (com) position of the (binary) orbiter.

    Args:
        orbiter (Particles): Particle set holding the oribiter body or bodies.

    Returns:
        VectorQuantity: Position of the center of mass 
    """
    if len(orbiter) == 2:
        com = (orbiter[0].position * orbiter[0].mass + orbiter[1].position * orbiter[1].mass) / (orbiter[0].mass + orbiter[1].mass)
    else: #single body
        com = orbiter.copy().position.flatten()
    return com


def get_com_vel(orbiter):
    """Helper function to find the center of mass (com) velocity of the (binary) orbiter.

    Args:
        orbiter (Particles): Particle set holding the oribiter body or bodies.

    Returns:
        VectorQuantity: Velocity of the center of mass 
    """
    if len(orbiter) == 2:
        com = (orbiter[0].velocity * orbiter[0].mass + orbiter[1].velocity * orbiter[1].mass) / (orbiter[0].mass + orbiter[1].mass)
    else: #single body
        com = orbiter.copy().velocity.flatten()
    return com


def plot_ecc_cos_ang(ax: mpl.axes.Axes, ang: VectorQuantity, ecc: np.ndarray, **kwargs) -> None:
    """Create plot of eccentricity versus cos(angle), where the angle can be any of the three orbital angles."""
    plt.plot(np.cos(ang.value_in(units.rad)), ecc, **kwargs)
    ax.set_ylabel(r'$e$')
    ax.set_ylim(0,1)
    ax.set_xlim(-1, 1)
    return


def plot_inc_ecc(ax: mpl.axes.Axes, time: VectorQuantity, inc: VectorQuantity, ecc: np.ndarray) -> None:
    """Create plot of inclination and eccentricity over time given an axis to plot on.

    Args:
        ax (mpl.axes.Axes): Matplotlib axis to create figure on.
        time (VectorQuantity): VectorQuantity of the time to put on the x-axis, will be plotted in units of years.
        inc (VectorQuantity): VectorQuantity of the inclination to be put on one of the y-axes. Will be plotted in units of degrees.
        ecc (np.ndarray): Array of eccentricities to put on one of the y-axes.
    """
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


def plot_ang_ecc(ax: mpl.axes.Axes, time: VectorQuantity, inc: VectorQuantity, asc: VectorQuantity,
                  peri: VectorQuantity, ecc: np.ndarray, legend_kwargs=None) -> None:
    """Creates plot of eccentricity, inclination, longitude of ascending node and argument of periapsis over time. 

    Args:
        ax (mpl.axes.Axes): Matplotlib axis to create figure on.
        time (VectorQuantity): VectorQuantity of the time to put on the x-axis, will be plotted in units of years.
        inc (VectorQuantity): VectorQuantity of the inclination to be put on one of the y-axes. Will be plotted in units of degrees.
        asc (VectorQuantity): VectorQuantity of longitude of ascending node. Will be plotted in units of degrees.
        peri (VectorQuantity): VectorQuantity of argument of periapsis. Will be plotted in units of degrees.
        ecc (np.ndarray): Array of eccentricities to put on one of the y-axes.
    """
    ecc_ax = ax
    ang_ax = ax.twinx()

    ecc_ax.plot(time.value_in(units.yr), ecc, color='red', label=r'$e$')
    plt.plot(time.value_in(units.yr), inc.value_in(units.deg), color='blue', linestyle='solid', label=r'$\iota$')
    plt.plot(time.value_in(units.yr), asc.value_in(units.deg), color='blue', linestyle='dotted', label=r'$\Omega$')
    plt.plot(time.value_in(units.yr), peri.value_in(units.deg), color='blue', linestyle='dashed', label=r'$\omega$')

    ecc_ax.set_xlabel('Time [yr]')
    ecc_ax.set_ylabel('Eccentricity', color='red')
    ecc_ax.tick_params(axis='y', labelcolor='red')
    ecc_ax.set_ylim(0,1)

    ang_ax.set_ylabel('Angle [deg]', color='blue')
    ang_ax.tick_params(axis='y', labelcolor='blue')
    ang_ax.set_ylim(-180, 180)

    if legend_kwargs:
        plt.legend(**legend_kwargs)
    return


def plot_smbh_and_stars(axes:mpl.axes.Axes, particle_set: Particles, lim: int = 10) -> None:
    """    
    Creates plot SMBH and the (binary) system orbiting it.
    Since this is in 3D space, plot 3 subplots of every 2D planar projection. Requires axes to be an array of shape (3,). 

    Plot is centered on the SMBH.

    Args:
        ax (mpl.axes.Axes): Matplotlib axis to create figure on.
        particle_set (Particles): Particle set of the SMBH and the orbiter. 
        lim (int, optional): Sets the xlim and ylim of the figure. Defaults to 10.
    """
    # Isolate SMBHand star(s)
    smbh = particle_set[particle_set.name == 'SMBH'] 
    stars = particle_set[ np.logical_or(particle_set.name == 'primary_star', particle_set.name == 'secondary_star') ]

    ax_xy, ax_yz, ax_xz = axes

    # Plot SMBH in every planar projection
    for ax in axes.flatten():
        ax.scatter(0, 0, marker='o', s=75, color='black', label='Sgr A*')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')

    # Plot star(s) in every planar projection
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


def plot_stars(axes:mpl.axes.Axes, particle_set: Particles, lim: int = 3) -> None:
    """ Creates plot of the binary star system. Therefore, this can only be called when dealing with a binary system.
        Since this is in 3D space, plot 3 subplots of every 2D planar projection. 
        Requires axes to be an array of shape (3,). Plots are centered on the primary.

    Args:
        ax (mpl.axes.Axes): Matplotlib axis to create figure on.
        particle_set (Particles): Particle set of the binary system. 
        lim (int, optional): Sets the xlim and ylim of the figure. Defaults to 3.
    """

    # Isolate primary and secondary
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


def plot_binary_disk_arrow(axes:mpl.axes.Axes, particle_set: Particles, lim: int = 20) -> None:
    """ Creates plot of the binary star system with hydrodynamical disk around it. Moreover, an arrow is pointing towards
        the SMBH which is itself not plotted due to scaling reasons. The disk is colored based on the Euclidian coordinates of its 
        particles.
        Since this is in 3D space, plot 3 subplots of every 2D planar projection. 
        Requires axes to be an array of shape (3,). Plots are centered on the primary.

    Args:
        ax (mpl.axes.Axes): Matplotlib axis to create figure on.
        particle_set (Particles): Particle set of the entire system, i.e. the binary, SMBH and disk. 
        lim (int, optional): Sets the xlim and ylim of the figure. Defaults to 20.
    """

    ax_xy, ax_yz, ax_xz = axes

    particle_set.position -= particle_set[particle_set.name == 'primary_star'].position  # Center on primary

    disk = particle_set[particle_set.name == 'disk']
    sph_pos = (disk.position.value_in(units.AU) / lim) / 2 + 0.5  # Get rgb = xyz color on disk normalized on furthest bound particle
    sph_pos[np.any(sph_pos > 1, axis=1)] = 0
    sph_pos[np.any(sph_pos < 0, axis=1)] = 0

    #Plot disk and stars
    ax_xy.scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1, c=sph_pos)
    ax_yz.scatter(disk.y.value_in(units.AU), disk.z.value_in(units.AU), s=1, c=sph_pos)
    ax_xz.scatter(disk.x.value_in(units.AU), disk.z.value_in(units.AU), s=1, c=sph_pos)
    plot_stars(axes, particle_set, lim)

    # Isolate SMBH from set and initialize arrow
    smbh = particle_set[particle_set.name == 'SMBH']
    arrow_to_smbh = (smbh.position / smbh.position.length())[0]  # Normalize arrow to length 1

    slice_dict = {0:[0,1], 1:[1,2], 2:[0,2]} # Able to extract xy, yz and xz plane
    for i, axis in enumerate(axes):

        ymin, ymax = axis.get_ylim()
        yrange = ymax - ymin

        com_plot = (0,0)

        # This sets the scale of the arrowhead, which is directly depedent on the yscale of the axis
        if i != 0:
            arrowhead = arrow_to_smbh[slice_dict[i]] * 0.25 * yrange
        else:
            arrowhead = arrow_to_smbh[slice_dict[i]] * 0.5 * yrange

        # Plot arrow
        axis.annotate("", xy=arrowhead, xytext=com_plot,
                      arrowprops=dict(arrowstyle="->"),xycoords='data',textcoords='data'
                      )
  

if __name__ == '__main__':
    # Initialize parser for the plotter functionality. 
    parser = argparse.ArgumentParser(description='Make relevant plots from HDF5 files')
    parser.add_argument('--snapshot_dir',type=str,default=os.getcwd(),help='Full path to directory where HDF5 snapshots are stored.')
    parser.add_argument('--step_size',type=int,default=100,help='Generate a plot every step_size snapshot.')
    parser.add_argument('--no_disk', type = bool, default = False, help = 'Boolean to specify if simulation is with or without disk.')
    args = parser.parse_args()

    # Move one directory up from the given snapshot_dir so as not to save plots where snapshots are stored
    os.chdir(args.snapshot_dir) 
    os.chdir('..')    
    print(f'Moving to {os.getcwd()}')

    # Create name of (new) image dir
    image_tail = args.snapshot_dir.split('/')[-2].split('-')[1:]
    image_dir = 'images-' + image_tail[0] + '-' + image_tail[1] 
    if args.no_disk:
        image_dir += '-' + 'no_disk'

    print(f'Saving images in {os.path.join(os.getcwd(),image_dir)}')
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
    else:  # Empty out image directory if it already exists
        if len(os.listdir(image_dir)) != 0: 
            print(f'Found existing image(s) in {image_dir}, removing them...')
            files = glob.glob(image_dir + '/*.png')
            for f in files:
                os.remove(f)

    # Loading in data
    unsorted_datafiles = glob.glob(args.snapshot_dir + '/*.hdf5')
    file_numbers = [filename.split('_')[-1].split('.')[0] for filename in unsorted_datafiles]
    file_numbers = list(map(int, file_numbers))
    _, datafiles = zip(*sorted(zip(file_numbers, unsorted_datafiles)))

    try:  # TODO: exception always raised
        time_glob = glob.glob(os.getcwd()+f'/times-year-{image_tail[0][4:]}*-{image_tail[1][4:]}*.npy')[0]  # Requires the run to have succeeded such that a time array exists
        times = np.load(time_glob, allow_pickle=True)
        try:  # Check if there are units to remove
            times = [int(time.value_in(units.yr)) for time in times]  
        except:
            pass
    except:
        print('\nWARNING: EXCEPTION RAISED IN IMPORTING TIMES\n')
        times = np.arange(len(datafiles))

    median_incs = []
    median_ascs = []
    median_peris = []
    median_eccs = []
    plot_time = []
    n_bound_list = []

    print(f'Processing {len(datafiles)} snapshots, making a plot of every {args.step_size}...')
    for time, datafile in tqdm(zip(times[::args.step_size], datafiles[::args.step_size])): #every args.step_size instance is looped over
        data = read_set_from_file(datafile)  # Full particle set at single timestep

        # Get orbital parameters in case of a simulation with a disk
        if not args.no_disk:
            disk = data[data.name == 'disk']
            stars = data[ np.logical_or(data.name == 'primary_star', data.name == 'secondary_star') ]

            if len(stars) == 1:
                com = stars
            else:
                com = Particle()
                com.position = get_com(stars)
                com.velocity = get_com_vel(stars)
                com.mass = stars[0].mass + stars[1].mass

            _, _, _, eccs, _, incs, ascs, peris = get_orbital_elements_from_binaries(com, disk, G=constants.G)


        ## Get orbital parameters in case of a simulation without a disk (i.e., gravity only)
        else:
            primary = data[1]
            primary.name = 'primary_star'
            secondary = data[2]
            secondary.name = 'secondary_star'

            _, _, _, eccs, _, incs, _, _ = get_orbital_elements_from_binaries(primary, secondary, G=constants.G) 

        # Find the bound particles in the disk
        bound = eccs < 1
        bound_eccs = eccs[bound]
        bound_incs = incs[bound]
        bound_ascs = ascs[bound]
        bound_peris = peris[bound]
        n_bound = np.sum(bound)

        median_incs.append(np.median(bound_incs.value_in(units.deg)))
        median_ascs.append(np.median(bound_ascs.value_in(units.deg)))
        median_peris.append(np.median(bound_peris.value_in(units.deg)))
    
        median_eccs.append(np.median(bound_eccs))
        plot_time.append(time)
        n_bound_list.append(n_bound)

        ### MAKE PLOT WITH ONLY THE BINARY ###
        if args.no_disk:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 9.5))
            plot_stars(np.array([ax1, ax2, ax3]), data, lim=5)
            plot_inc_ecc(ax4, 
                        plot_time | units.yr, 
                        median_incs | units.deg,
                        median_eccs)
            fig.suptitle(f'Time = {time:.0f} year', fontsize=BIGGER_SIZE)
            plt.tight_layout()

            fig.savefig(f"{image_dir}/binary-{datafile.split('_')[-1].split('.')[0]}.png", bbox_inches='tight')
            plt.close()

        #########################################

        ### MAKE PLOT WITH BINARY + DISK + ARROW ###
        if not args.no_disk:

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
            plot_binary_disk_arrow(np.array([ax1, ax2, ax3]), data)
            plot_ang_ecc(ax4, 
                        plot_time | units.yr, 
                        median_incs | units.deg,
                        median_ascs | units.deg,
                        median_peris | units.deg,
                        median_eccs)
            fig.suptitle(f'Time = {time} year', fontsize=BIGGER_SIZE)
            plt.tight_layout()

            fig.savefig(f"{image_dir}/binary-with-arrow-{datafile.split('_')[-1].split('.')[0]}.png", bbox_inches='tight')
            plt.close()

        #########################################
        
