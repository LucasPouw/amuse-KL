 
## Made by Lucas Pouw, Yannick Badoux and Tim van der Vuurst for the 
## course "Simulations and Modeling in Astrophysics" '24-'25. 

import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units, constants
from amuse.io import read_set_from_file
import os
import glob
from amuse.ext.orbital_elements import get_orbital_elements_from_binaries
from amuse.lab import Particles, Particle
from tqdm import tqdm
from scipy.optimize import curve_fit

from plotter import * # Importing also changes mpl.rcParams to make plots nice

# The runs where Fi crashed need to be processed again from the snapshots, because the summarizing diagnostics weren't saved

def dissect_system(particle_system):
    """ 
    Helper function that seperates disk, stars and center of mass from a given particle system
    """
    disk = particle_system[particle_system.name == 'disk'].copy()
    stars = particle_system[np.logical_or(particle_system.name == 'primary_star', particle_system.name == 'secondary_star')].copy()

    # Initialize com as a particle with the total stellar mass
    com = Particle()
    com.position = get_com(stars)
    com.velocity = get_com_vel(stars)
    com.mass = 0 | units.Msun
    for star in stars: # Works for both single and double star
        com.mass += star.mass 
    return disk, stars, com


def get_disk_orbital_elements(particle_system):
    """ 
    Wrapper function to distill oribital elements of the disk from the total particle system (i.e. with SMBH and stars in it as well) 
    """
    disk, _, com = dissect_system(particle_system)
    m1, m2, ax, ecc, anom, inc, asc, peri = get_orbital_elements_from_binaries(com, disk, G=constants.G)
    return m1, m2, ax, ecc, anom, inc, asc, peri
    

def get_N_bound(particle_system):
    """ 
    Gets the number of bound disk particles from  the total particle system (i.e. with SMBH and stars in it as well) 
    """
    _, _, _, eccs, _, _, _, _ = get_disk_orbital_elements(particle_system)
    bound = eccs < 1
    return np.sum(bound)


def get_sorted_files(file_dir):
    """ 
    Given a directory, retrieves all hdf5 snapshots within it. These are then sorted by filename and returned.
    """
    unsorted_datafiles = glob.glob(file_dir + '/*.hdf5')
    file_numbers = [filename.split('_')[-1].split('.')[0] for filename in unsorted_datafiles]
    file_numbers = list(map(int, file_numbers))
    _, datafiles = zip(*sorted(zip(file_numbers, unsorted_datafiles)))
    return datafiles


def get_nbound_over_time(file_dir, save_name):
    """ 
    Given a directory containing simulation snapshots, this function creates an array of the bound disk particles over time 
    (i.e., at every snapshot) and saves it. If save_name is not an absolute path, the array is saved in the working directory 
    (not recommended). This function should only be used for runs in which an unexpected crash caused the diagnostics to not be saved.
    """
    nbound = []
    for datafile in tqdm(get_sorted_files(file_dir)):
        snapshot = read_set_from_file(datafile)
        nbound.append(get_N_bound(snapshot))

    np.save(save_name, np.array(nbound))


def get_snapshot_paths():
    #function to scrape all snapshot and run roots from all runs (excluding any no_disk run explicitly)
    datadirs = []
    roots = []
    for root,dirs,files in os.walk(DATAPATH):
        for dir in dirs:
            if 'snapshot' in dir and 'no_disk' not in root:
                datadirs.append(os.path.join(root,dir))
                if root not in roots:
                    roots.append(root)
    return datadirs,roots

def rectify_filenames(root):
    #sometimes, the .npy files are named inconsistently. This fixes that.
    for elem in glob.glob(os.path.join(root,'*.npy')):
        if isinstance(elem,str):
            head,tail = os.path.split(elem)
            tail = tail.replace('rmin','').replace('rmax','')
            new_filename = os.path.join(head,tail)
            if elem != new_filename:
                os.rename(elem,new_filename)
                print(f'Renaming {e} to {new_filename}')
        else:
            for e in elem:
                print(e)
                head,tail = os.path.split(e)
                tail = tail.replace('rmin','').replace('rmax','')
                new_filename = os.path.join(head,tail)
                if e != new_filename:
                    os.rename(e,new_filename)
                    print(f'Renaming {e} to {new_filename}')


def _get_npy_paths(root):
    # get all paths to .npy files given the root directory of a run
    files = np.sort(glob.glob(os.path.join(root,'*.npy')))
    times,grav_energy,hydro_energy, nbound, rhalf = [],[],[],[],[]
    for f in files:
        if 'times' in f:
            times.append(f)
        elif 'grav' in f:
            grav_energy.append(f)
        elif 'hydro' in f:
            hydro_energy.append(f)
        elif 'Nbound' in f:
            nbound.append(f)
        elif 'Rhalf' in f:
            rhalf.append(f)

    return times,grav_energy,hydro_energy, nbound, rhalf

def get_npy_paths(root):
    #uses above function but with nicer output
    return list(zip(*_get_npy_paths(root)))

def get_rmin_rmax_from_run(root):
    info_strings = np.array(os.path.split(root)[-1].split('-'))
    print()
    if 'r_min' in info_strings and 'r_max' in info_strings:
        rmin = float(info_strings[np.where(np.array(info_strings) == 'r_min')[0][0] + 1])
        rmax = float(info_strings[np.where(np.array(info_strings) == 'r_max')[0][0] + 1])
    elif 'r_min' in info_strings and 'r_max' not in info_strings:
        rmin = float(info_strings[np.where(np.array(info_strings) == 'r_min')[0][0] + 1])
        rmax = 13.35 #default value
    elif 'r_max' in info_strings and 'r_min' not in info_strings:
        rmax = float(info_strings[np.where(np.array(info_strings) == 'r_max')[0][0] + 1])
        rmin = 6.55 #default vlaue
    else:
        rmin = 6.55
        rmax = 13.35
    
    return rmin, rmax


def process_run(root,savefigs=False):
    array_paths = glob.glob(root+'/*.npy')

    if len(array_paths) < 5: #these are the runs that crashed and only have a manually created Nbound array. We only plot Nbound for these runs
        for path in array_paths:
            if 'Nbound' not in path: #we plot the Nbound seperately now
                continue
            else:
                rmin, rmax = os.path.split(path)[-1].strip('.npy').split('_')[-1].split('-') #extract rmin and rmax from filepath
                print(rmin,rmax)
                rmin, rmax = float(rmin), float(rmax)
                Nbound = np.load(path)
                times = np.arange(1, len(Nbound)+1) 

                fig,ax = plt.subplots(figsize=(8,6))
                fig.suptitle(r'$R_{\rm min}=$' + f'{rmin:.2f} AU, ' + r'$\, R_{\rm max}=$' + f'{rmax:.2f} AU',fontsize=28)
                ax.plot(times,Nbound)
                ax.semilogx()
                ax.set(xlabel='Time [yr]',ylabel=r'$N_{\rm bound}$')
                if savefigs:
                    if 'amuse-KL' not in os.getcwd():
                        os.chdir('amuse-KL')
                    if not os.path.isdir('figures'):
                        os.mkdir('figures')
                    filename = f'./figures/rmin-{rmin}_rmax{rmax}'
                    if 'm_orb' in root:
                        filename += '_(single star)'
                    filename += '.pdf'
                    plt.savefig(filename,bbox_inches='tight')
                plt.show()
        
        return #dont make the 2x2 figure


    #given the root of a run (with various simulations in it), get the various arrays and make plots
    run_paths = get_npy_paths(root) #get relevant npy files     
    fig,axes = plt.subplots(figsize=(12,8),nrows=2,ncols=2,tight_layout=True)
    rmin_run, rmax_run = get_rmin_rmax_from_run(root)
    suptitle = r'$R_{\rm min}=$' + f'{rmin_run:.2f} AU,' + r'$\, R_{\rm max}=$' + f'{rmax_run:.2f} AU'
    if 'm_orb' in root:
        suptitle += ' (single star)'
    fig.suptitle(suptitle,fontsize=28)
    
    for i,all_paths in enumerate(run_paths): #iterate per sim
        arrays = []
        for i,path in enumerate(all_paths): #go over all arrays in order and see if we saved with units or not, if so remove units
            try:
                arr = np.load(path)
            except ValueError:
                arr = np.load(path,allow_pickle=True)
                if 'time' in path:
                    arr = [t.value_in(units.yr) for t in arr]
                elif 'grav' in path or 'hydro' in path:
                    arr = [t.value_in(units.J) for t in arr]
                elif 'Rhalf' in path:
                    arr = [t.value_in(units.AU) for t in arr]
            
            arrays.append(arr)

        times,grav_energy,hydro_energy,Nbound,Rhalf = arrays #unpack list of arrays
        grav_energy_error = (grav_energy - grav_energy[0]) / grav_energy[0]
        hydro_energy_error = (hydro_energy - hydro_energy[0]) / hydro_energy[0]

        rmin,rmax = os.path.split(all_paths[0])[-1].strip('.npy').split('-')[-2:] #extract rmin and rmax a filepath
        rmin,rmax = float(rmin),float(rmax)
        
        axes[0,0].plot(times[1:],Nbound,label = r'$R_{\rm min}=$' + f'{rmin:.2f}' + r'$\, R_{\rm max}=$' + f'{rmax:.2f}')
        axes[0,0].semilogx()
        axes[0,0].set(xlabel='Time [yr]',ylabel=r'$N_{\rm bound}$')

        axes[0,1].plot(times[1:],Rhalf)
        axes[0,1].semilogx()
        axes[0,1].set(xlabel='Time [yr]',ylabel=r'$R_{\rm half}$ [AU]')

        axes[1,0].plot(times,grav_energy_error)
        axes[1,0].semilogx()
        axes[1,0].set(xlabel='Time [yr]',ylabel=r'Gravity energy error [J]')

        axes[1,1].plot(times,hydro_energy_error)
        axes[1,1].semilogx()
        axes[1,1].set(xlabel='Time [yr]',ylabel=r'Hydro energy error [J]')
        
    if len(run_paths) > 1:
        handles,labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,bbox_to_anchor=(1.35,0.75))
    fig.tight_layout()
    if savefigs:
        if 'amuse-KL' not in os.getcwd():
            os.chdir('amuse-KL')
        if not os.path.isdir('figures'):
            os.mkdir('figures')
        filename = f'./figures/rmin-{rmin_run}_rmax{rmax_run}'
        if 'm_orb' in root:
            filename += '_(single star)'
        filename += '.pdf'
        plt.savefig(filename,bbox_inches='tight')
    plt.show()



 
if __name__ == '__main__':
    ##TODO: change to parser or input somehow
    ## best to do is find out where files such as these are made and saved and make sure its all in the same dir so only 
    ## one argument can scrape them all with a function (see below, already exists)
    eccs_single = np.load('/data2/pouw/amuse-project/amuseKL-output/eccs-m_orb-3.53-r_min-8.12-r_max-9.42-vary_radii-True.npy')
    incs_single = np.load('/data2/pouw/amuse-project/amuseKL-output/incs-m_orb-3.53-r_min-8.12-r_max-9.42-vary_radii-True.npy')
    ascs_single = np.load('/data2/pouw/amuse-project/amuseKL-output/ascs-m_orb-3.53-r_min-8.12-r_max-9.42-vary_radii-True.npy')
    peris_single = np.load('/data2/pouw/amuse-project/amuseKL-output/peris-m_orb-3.53-r_min-8.12-r_max-9.42-vary_radii-True.npy')
    times_single = np.load('/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data/m_orb-3.53-r_min-8.12-r_max-9.42-vary_radii-True/times-years-8.120-9.420.npy')

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_ang_ecc(ax, times_single | units.yr, incs_single | units.deg, ascs_single | units.deg, peris_single | units.deg, eccs_single)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times_single[::100], peris_single[::100])
    plt.show()


    ##Same with the parser stuff
    time = np.load('/data2/pouw/amuse-project/amuseKL-output/vary_radii-True/times-year-7.400-11.480.npy', allow_pickle=True)
    times = [t.value_in(units.yr) for t in time]
    print(f'Check if every step got saved: {int(times[-1])}, {len(times)}')

    rhalf = np.load('/data2/pouw/amuse-project/amuseKL-output/vary_radii-True/Rhalf_rmin7.400-rmax11.480.npy', allow_pickle=True)  # Does not include t=0
    nbound = np.load('/data2/pouw/amuse-project/amuseKL-output/vary_radii-True/Nbound_rmin7.400-rmax11.480.npy', allow_pickle=True)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(times[1:], rhalf)
    ax.loglog()
    plt.show()

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(times[1:], nbound)
    ax.loglog()
    plt.show()


 
    # We choose to analyze the disk at 25 kyr, in the middle of the plateaus
    file_idx = 25000

    ## Change again so no hardcoded paths are no longer needed
    ## probably best to specify runs that you can input with defaults leading to this.
    ## again, simulation (e.g. snapshot) output should be consistent
    dirs = np.array(['/data2/pouw/amuse-project/amuseKL-output/vary_radii-True/snapshots-rmin7.265-rmax12.025',
                    '/data2/pouw/amuse-project/amuseKL-output/r_min-7.589-r_max-10.989/snapshots-rmin7.589-rmax10.989',
                    '/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data/r_min-7.625-r_max-12.025-vary_radii-True/snapshots-rmin7.763-rmax11.723',
                    '/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data/r_min-7.625-r_max-12.025-vary_radii-True/snapshots-rmin8.027-rmax11.107'])
    
    labels = np.array([r'$R_{\rm min}=7.26, \, R_{\rm max}=12.03$', 
                    r'2, $R_{\rm min}=7.59, \, R_{\rm max}=10.99$',
                    r'$R_{\rm min}=7.76, \, R_{\rm max}=11.72$',
                    r'$R_{\rm min}=8.03, \, R_{\rm max}=11.11$'])


    medians = np.zeros(len(dirs))
    stds = np.zeros(len(dirs))

    plt.figure(figsize=(8,6))
    for i, snapshot_dir in enumerate(dirs):
        file = get_sorted_files(snapshot_dir)[file_idx]
        data = read_set_from_file(file)

        _, _, _, ecc, _, inc, asc, peri = get_disk_orbital_elements(data)
        bound = ecc < 1
        
        disk, _, com = dissect_system(data)
        disk.position -= com.position
        bound_disk_pos = np.linalg.norm(disk[bound].position.value_in(units.AU), axis=1)

        plt.hist(bound_disk_pos, density=True, bins=30, histtype='step', linewidth=5, label=labels[i])

        medians[i] = np.median(bound_disk_pos)
        stds[i] = np.std(bound_disk_pos)

    plt.ylabel('Probability density')
    plt.xlabel('Distance from binary')
    plt.grid()
    plt.legend()
    plt.show()

    inners_1sig = medians - stds / 2
    outers_1sig = medians + stds / 2
    inners_2sig = medians - stds
    outers_2sig = medians + stds
    print(f'The inner and outer radii at 1 sigma are {inners_1sig} and {outers_1sig}. This gives a mean of {np.mean(np.array(inners_1sig)):.2f} and {np.mean(np.array(outers_1sig)):.2f} AU.')
    print(f'The inner and outer radii at 2 sigma are {inners_2sig} and {outers_2sig}. This gives a mean of {np.mean(np.array(inners_2sig)):.2f} and {np.mean(np.array(outers_2sig)):.2f} AU.')


    ## Change to non-hardcoded path, again with parser
    DATAPATH = '/data2/AMUSE-KL-vdvuurst-pouw-badoux/'


    
    snapshot_dirs,run_roots = get_snapshot_paths()


    
    for root in run_roots:
        rectify_filenames(root)


    for root in run_roots:
        process_run(root,savefigs=True)
