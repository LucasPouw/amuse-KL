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


def plot_binary_with_arrow(particle_set): 
    particle_set.position -= particle_set[particle_set.name=='primary'].position #center on primary
    orbiter = particle_set[np.logical_or(particle_set.name == 'primary',particle_set.name == 'secondary')]
    disk = particle_set[particle_set.name == 'disk']
    smbh = particle_set[particle_set.name == 'SMBH']
    
    arrow_to_smbh = smbh.position 
    arrow_to_smbh = (arrow_to_smbh / arrow_to_smbh.length())[0] #normalize arrow to length 1

    fig, ax = plt.subplots(2, 2, figsize=(10,10))

    ax[0,0].scatter(orbiter.x.value_in(units.AU), orbiter.y.value_in(units.AU), zorder=100)
    ax[0,0].scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1)
    # ax[0,0].set_ylim(com.y.value_in(units.AU) - 25, com.y.value_in(units.AU) + 25)
    # ax[0,0].set_xlim(com.x.value_in(units.AU) - 25, com.x.value_in(units.AU) + 25)
    ax[0,0].set_xlabel('x [AU]')
    ax[0,0].set_ylabel('y [AU]')

    ax[1,0].scatter(orbiter.x.value_in(units.AU), orbiter.z.value_in(units.AU), zorder=100)
    ax[1,0].scatter(disk.x.value_in(units.AU), disk.z.value_in(units.AU), s=1)
    # ax[1,0].set_ylim(com.z.value_in(units.AU) - 0.1, com.z.value_in(units.AU) + 0.1)
    # ax[1,0].set_xlim(com.x.value_in(units.AU) - 25, com.x.value_in(units.AU) + 25)
    ax[1,0].set_xlabel('x [AU]')
    ax[1,0].set_ylabel('z [AU]')
    ax[1,0].ticklabel_format(useOffset=False)

    ax[1,1].scatter(orbiter.y.value_in(units.AU), orbiter.z.value_in(units.AU), zorder=100)
    ax[1,1].scatter(disk.y.value_in(units.AU), disk.z.value_in(units.AU), s=1)
    # ax[1,1].set_ylim(com.z.value_in(units.AU) - 0.1, com.z.value_in(units.AU) + 0.1)
    # ax[1,1].set_xlim(com.y.value_in(units.AU) - 25, com.y.value_in(units.AU) + 25)
    ax[1,1].set_xlabel('y [AU]')
    ax[1,1].set_ylabel('z [AU]')
    ax[1,1].ticklabel_format(useOffset=False)

    slice_dict = {0:[0,1],2:[0,2],3:[1,2]} #able to extract xy, xz and yz plane
    for i,axis in enumerate(ax.flatten()):
        if i == 1:
            continue
        ymin,ymax = axis.get_ylim()
        yrange = ymax - ymin
        print(yrange)
        # com_plot = data_to_axis(com.value_in(units.AU)[slice_dict[i]]) #get the relevant com coordinates
        com_plot = (0,0)

        #This sets the scale of the arrowhead, and it is now directly depedent on the yscale of the axis
        #which may mess up in the future so if it does, look here
        if i != 0:
            arrowhead = arrow_to_smbh[slice_dict[i]] * 0.25 * yrange
        else:
            arrowhead = arrow_to_smbh[slice_dict[i]] * 0.5 * yrange


        axis.annotate("", xy=arrowhead, xytext=com_plot,
                    arrowprops=dict(arrowstyle="->"),xycoords='data',textcoords='data',
                    )
        
        ##Uncomment if you want to plot the com and arrowhead as seperate scatters
        # axis_to_data = axis.transLimits.inverted().transform
        # print(axis_to_data(com_plot))
        # print(axis_to_data((0.5,0.5)))

        


    plt.tight_layout()
    # fig.savefig(movie_kwargs['image_folder'] + 'disk-snapshot-' + f'{int(model_time.value_in(units.day))}.png', 
    #             bbox_inches='tight',
    #             dpi=200)
    
    # plt.close()

    plt.show()



if __name__ == '__main__':
    data = read_set_from_file('smbh_binary_disk.hdf5')
    print(len(data))

    time = 1 | units.yr

    plot_binary_with_arrow(data)



    # gs = gridspec.GridSpec(4, 4)
    # fig = plt.figure(figsize=(10,10))
    # ax1 = fig.add_subplot(gs[:2, :2])
    # ax2 = fig.add_subplot(gs[:2, 2:])
    # ax3 = fig.add_subplot(gs[2:4, 1:3])
    # plot_smbh_and_stars(np.array([ax1, ax2, ax3]), data)
    # fig.suptitle(f'Time = {time.value_in(units.yr):.0f} year', fontsize=BIGGER_SIZE)
    # plt.tight_layout()
    # fig.show()
    

    # time = 1 | units.yr
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 9.5))
    # plot_stars(np.array([ax1, ax2, ax3]), data, lim=1)
    # plot_inc_ecc(ax4, np.linspace(0, 1, 100) | units.yr, np.linspace(0, 100, 100) | units.deg, np.linspace(0, 1, 100))
    # fig.suptitle(f'Time = {time.value_in(units.yr):.0f} year', fontsize=BIGGER_SIZE)
    # plt.tight_layout()
    # # ax4.set_aspect('equal')
    # fig.show()


    