 
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
import argparse
from plotter import * # Importing also changes mpl.rcParams to make plots nice

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


def get_snapshot_paths(datapath):
    #function to scrape all snapshot and run roots from all runs (excluding any no_disk run explicitly)
    datadirs = []
    roots = []
    for root,dirs,files in os.walk(datapath):
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
            if 'rmin' not in elem and 'rmax' not in elem:
                continue
            head,tail = os.path.split(elem)
            tail = tail.replace('rmin','').replace('rmax','')
            new_filename = os.path.join(head,tail)
            if elem != new_filename:
                os.rename(elem,new_filename)
                print(f'Renaming {elem} to {new_filename}')
        else:
            for e in elem:
                if 'rmin' not in e and 'rmax' not in e:
                    continue
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
        ftail = os.path.split(f)[-1] # Look only at the tail for comparison
        if 'times' in ftail:
            times.append(f)
        elif 'grav' in ftail:
            grav_energy.append(f)
        elif 'hydro' in ftail:
            hydro_energy.append(f)
        elif 'Nbound' in ftail:
            nbound.append(f)
        elif 'Rhalf' in ftail:
            rhalf.append(f)

    return times,grav_energy,hydro_energy, nbound, rhalf

def get_npy_paths(root):
    #uses above function but with nicer output
    return list(zip(*_get_npy_paths(root)))

def get_rmin_rmax_from_run(root):
    if root.endswith('/'):
        root = root[:-1]
    info_strings = np.array(os.path.split(root)[-1].split('-'))
    if 'r_min' in info_strings and 'r_max' in info_strings:
        rmin = float(info_strings[np.where(np.array(info_strings) == 'r_min')[0][0] + 1])
        rmax = float(info_strings[np.where(np.array(info_strings) == 'r_max')[0][0] + 1])
    elif 'r_min' in info_strings and 'r_max' not in info_strings:
        rmin = float(info_strings[np.where(np.array(info_strings) == 'r_min')[0][0] + 1])
        rmax = 13.35 #default value
    elif 'r_max' in info_strings and 'r_min' not in info_strings:
        rmax = float(info_strings[np.where(np.array(info_strings) == 'r_max')[0][0] + 1])
        rmin = 4.45 #default vlaue
    else:
        #defaults
        rmin = 4.45
        rmax = 13.35
    
    return rmin, rmax


def nbound_rhalf_plot(root: str,savedir : str | None = None) -> None:
    """ Creates a 2x2 figure. All subfigures show the evolution of a quantity over simulation-time, across simulations. 
        From the top-left to the bottom right is shown the number of bound particles, the half-mass radius of the disk, 
        the gravity energy error and the hydrodynamical energy error respectively. The legend specifies inner and outer radii of 
        disks across simulations in AU. If these 4 quantities do not all exist, only the number of bound particles is plotted. If this 
        itself does not exist, the code is terminated with a warning.

    Args:
        root (str): Root folder in which the simulation output is stored.
        savedir (str | None, optional): Path to directory where the resulting plot may be saved in. If None, will only show the plot. Defaults to None.

    """
    array_paths = glob.glob(root+'/*.npy')
    
    #Extract rmin and rmax from root name
    rmin_run, rmax_run = get_rmin_rmax_from_run(root)

    stable_run_rmin, stable_run_rmax = 8.12,9.42 #Hardcoded to mark explicitly in the plot for in the report. Does not cause errors if this run does not exist

    if len(array_paths) == 0:
        raise FileNotFoundError("No files ending in .npy found in the specified directory. This likely means you have to re-run your simulation(s), "+
                 "or use the get_nbound_over_time() functionality in this file to manually create the Nbound array.")
    
    # These are the runs that crashed and only have a manually created Nbound array. We only plot Nbound for these runs
    elif len(array_paths) < 5:
        fig,ax = plt.subplots(figsize=(8,6))
        for path in array_paths:
            if 'Nbound' not in path: # We plot the Nbound seperately now
                continue
            else:
                rmin, rmax = os.path.split(path)[-1].strip('.npy').split('_')[-1].split('-') #extract rmin and rmax from filepath
                rmin, rmax = float(rmin), float(rmax)
                
                Nbound = np.load(path)
                size = (Nbound.shape[0] // 50) * 50 
                arr = np.append(Nbound[:size].reshape(-1,50).mean(axis=1),np.mean(Nbound[size:])) # Take the mean of every 50 points to smooth out the plot

                times = np.arange(1, len(Nbound)+1) # The times array likely is not saved, so we manually recreate it.
                ax.plot(times,Nbound,label = r'$R_{\rm min}=$' + f'{rmin:.2f} AU, ' + r'$\, R_{\rm max}=$' + f'{rmax:.2f} AU')

        # ax.semilogx()
        ax.set(xlabel='Time [yr]',ylabel=r'$N_{\rm bound}$')
        if savedir is not None:
            filename = f'{savedir}/rmin-{rmin_run}_rmax{rmax_run}'
            if 'm_orb' in root:
                filename += '_(single star)'
            filename += '.pdf'
            plt.savefig(filename,bbox_inches='tight')
        plt.show()
        
        return # Don't make the 2x2 figure


    # Given the root of a run (with various simulations in it), get the various arrays and make plots
    run_paths = get_npy_paths(root) #get relevant npy files     
    fig,axes = plt.subplots(ncols=2,nrows=1,figsize=(10,6))
    suptitle = r'$R_{\rm min}=$' + f'{rmin_run:.2f} AU,' + r'$\, R_{\rm max}=$' + f'{rmax_run:.2f} AU'
    if 'm_orb' in root:
        suptitle += ' (single star)'
    fig.suptitle(suptitle,fontsize=28)
    
    for i,all_paths in enumerate(run_paths): # Iterate per sim
        arrays = []
        for i,path in enumerate(all_paths): # Go over all arrays in order and see if we saved with units or not, if so remove units
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
            
            size = (arr.shape[0] // 50) * 50 
            arr = np.append(arr[:size].reshape(-1,50).mean(axis=1),np.mean(arr[size:])) # Take the mean of every 50 points to smooth out the plot
            arrays.append(arr)

        times,_,_,Nbound,Rhalf = arrays # Unpack list of arrays, ignoring the grav energy and hydro energy arrays here

        rmin,rmax = os.path.split(all_paths[0])[-1].strip('.npy').split('-')[-2:] # Extract rmin and rmax from filepath
        rmin,rmax = float(rmin),float(rmax)
        
        if rmin == stable_run_rmin and rmax == stable_run_rmax: # Highlight the stable run for in the report
            axes[0].plot(times,Nbound,label = r'$R_{\rm min}=$' + f'{rmin:.2f}' + r'$\, R_{\rm max}=$' + f'{rmax:.2f}', lw = 3, zorder=100, c = 'black')
            axes[1].plot(times,Rhalf,label = r'$R_{\rm min}=$' + f'{rmin:.2f}' + r'$\, R_{\rm max}=$' + f'{rmax:.2f}', lw = 3,zorder=100, c = 'black')
        else:
            axes[0].plot(times,Nbound,label = r'$R_{\rm min}=$' + f'{rmin:.2f}' + r'$\, R_{\rm max}=$' + f'{rmax:.2f}')
            axes[1].plot(times,Rhalf,label = r'$R_{\rm min}=$' + f'{rmin:.2f}' + r'$\, R_{\rm max}=$' + f'{rmax:.2f}')
        
        axes[0].set(xlabel='Time [yr]',ylabel=r'$N_{\rm bound}$')
        axes[1].set(xlabel='Time [yr]',ylabel=r'$R_{\rm half}$ [AU]')

        axes[0].semilogx()
        axes[1].semilogx()

        axes[0].grid()
        axes[1].grid()

        
    if len(run_paths) > 1:
        handles,labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,bbox_to_anchor=(1,0), frameon=False,ncols=3,fontsize=16)

    fig.tight_layout()

    if savedir is not None:
        filename = f'{savedir}/nbound_rhalf_rmin-{rmin_run}_rmax{rmax_run}'
        if 'm_orb' in root:
            filename += '_(single star)'
        filename += '.pdf'
        plt.savefig(filename,bbox_inches='tight')

    plt.show()

def energy_error_plot(root: str, savedir: str | None = None):
    """ Creates a plot of the energy error over time for a given run. The run should have the relevant energy arrays saved in the root directory."""
    array_paths = glob.glob(root+'/*.npy')

    if len(array_paths) == 0:
        raise FileNotFoundError("No files ending in .npy found in the specified directory. This likely means you have to re-run your simulation(s), "+
                 "or use the get_nbound_over_time() functionality in this file to manually create the Nbound array.")

    # Given the root of a run (with various simulations in it), get the various arrays and make plots
    run_paths = get_npy_paths(root) #get relevant npy files     
    fig,axes = plt.subplots(ncols=1,nrows=1)
    rmin_run, rmax_run = get_rmin_rmax_from_run(root)
    suptitle = r'$R_{\rm min}=$' + f'{rmin_run:.2f} AU,' + r'$\, R_{\rm max}=$' + f'{rmax_run:.2f} AU'
    if 'm_orb' in root:
        suptitle += ' (single star)'
    fig.suptitle(suptitle,fontsize=28)
    
    for i,all_paths in enumerate(run_paths): # Iterate per sim
        arrays = []
        for i,path in enumerate(all_paths): # Go over all arrays in order and see if we saved with units or not, if so remove units
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

        times,grav_energy,hydro_energy,Nbound,Rhalf = arrays # Unpack list of arrays
        grav_energy_error = (grav_energy - grav_energy[0]) / grav_energy[0]

        rmin,rmax = os.path.split(all_paths[0])[-1].strip('.npy').split('-')[-2:] # Extract rmin and rmax from filepath
        rmin,rmax = float(rmin),float(rmax)

        axes.plot(times,grav_energy_error)
        axes.set(xlabel='Time [yr]',ylabel=r'Gravity energy error [J]')
        
    if len(run_paths) > 1:
        handles,labels = axes.get_legend_handles_labels()
        fig.legend(handles,labels,bbox_to_anchor=(1.35,0.75), frameon=False)

    fig.tight_layout()

    if savedir is not None:
        filename = f'{savedir}/energy_error_rmin-{rmin_run}_rmax{rmax_run}'
        if 'm_orb' in root:
            filename += '_(single star)'
        filename += '.pdf'
        plt.savefig(filename,bbox_inches='tight')

    plt.show()


def hydro_validation_plot(roots: list, savedir: str | None = None):
    #given a list of roots, merges them into one Nbound vs time plot
    #assumes every root has only a single run in it and that the roots all pertain to runs with the same rmin and rmax

    fig,ax = plt.subplots(figsize=(8,6))
    # fig.suptitle(r'$R_{\rm min}=$' + f'{rmin:.2f} AU,' + r'$\, R_{\rm max}=$' + f'{rmax:.2f} AU',fontsize = 28)
    for root in roots:
        # Extract the initial number of SPH particles in the disk from filename 
        rmin, rmax = get_rmin_rmax_from_run(roots[0])
        info_strings = np.array(os.path.split(root)[-1].split('-'))
        if 'n_disk' not in info_strings:
            ndisk = 1000
        else:
            ndisk = int(info_strings[np.where(info_strings == 'n_disk')[0][0] + 1])

        try:
            all_paths = get_npy_paths(root)[0] # Since we expect only 1 run per root, we can do this rectify shapes 
        except:
            path_lists = _get_npy_paths(root) # If some of the lists in get_npy_paths remain empty, it messed up the zip operation, so directly call the helper funcion in that case
            all_paths = [p[0] for p in path_lists if len(p) > 0]

        arrays = []
        for path in all_paths: # Go over all arrays in order and see if we saved with units or not, if so remove units
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
        try:
            times, _, _, Nbound, _ = arrays
        except ValueError: # In case arrays contains only Nbound
            Nbound = arrays[0]
            times = []

        if len(times) == 0:
            times = np.arange(0,len(Nbound)+1)

        frac_bound = Nbound / Nbound[0]
        ax.plot(times[1:],frac_bound, label = r'$R_{\rm min}=$' + f'{rmin:.2f}' + r'$\, R_{\rm max}=$' + f'{rmax:.2f}' + r'$\, N_{\rm disk}$=' + f"{ndisk} " )

    ax.set(xlabel = 'Time [yr]', ylabel = 'Nbound')
    ax.semilogx()
    ax.legend(bbox_to_anchor=(1.05,0.75),frameon=False, ncols=1)
    if savedir is not None:
        filename = f'{savedir}/hydro_validation.pdf'
        plt.savefig(filename,bbox_inches='tight')

    plt.show()



def KL_effect_plot(root: str, savedir: str | None = None):
    """ Creates a plot of the von Zeipel-Lidov-Kozai effect on orbital parameters of the disk, given the root in which simulation output is stored.
        If the necessary orbital elements are not yet saved in the directory, they will be calculated here.
        STRONGLY recommended to do this only for a run with a single snapshot folder specified as this function takes a while when running for the first time.

    Args:
        root (str): Root folder in which the simulation output is stored.
        savedir (str | None, optional): Path to directory where the resulting plot may be saved in. If None, will only show the plot. Defaults to None.
    """
    tail = os.path.split(root)[-1]

    # Check if the arrays containing the relevant data *all* exist already or not
    flag = True
    paths = []
    for attribute in ['eccs','incs','ascs','peris']:
        path = os.path.join(root,f'{attribute}-{tail}.npy')
        paths.append(path)
        flag *= os.path.isfile(path)
    

    if flag: # i.e. all arrays already exist for this root
        eccs_single = np.load(paths[0])
        incs_single = np.load(paths[1])
        ascs_single = np.load(paths[2])
        peris_single = np.load(paths[3])
    
    else: #the orbital parameters are not yet defined (at least not all of them), so we recalculate. This takes a while.
        
        eccs = []
        incs = []
        ascs = []
        peris = []
        for datafile in tqdm(get_sorted_files(root)):
            snapshot = read_set_from_file(datafile)
            _, _, _, ecc, _, inc, asc, peri = get_disk_orbital_elements(snapshot)
            bound = ecc < 1

            eccs.append(np.median(ecc[bound]))
            incs.append(np.median(inc[bound].value_in(units.deg)))
            ascs.append(np.median(asc[bound].value_in(units.deg)))
            peris.append(np.median(peri[bound].value_in(units.deg)))

        np.save(paths[0], eccs)
        np.save(paths[1], incs)
        np.save(paths[2], ascs)
        np.save(paths[3], peris)

    try:
        timepath = [t for t in os.listdir(root) if 'times' in t][0] #extract the time array
        times_single = np.load(timepath)
    except IndexError: #i.e., time array does not exist, we create it as we know it should be of the shape as the orbital parameters in steps of 1
        times_single = np.arange(1,len(eccs)+1) | units.yr

    # Create the plot TODO: Lucas wants to make some changes here.
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_ang_ecc(ax, times_single | units.yr, incs_single | units.deg, ascs_single | units.deg, peris_single | units.deg, eccs_single)
    
    rmin, rmax = get_rmin_rmax_from_run(root)
    if savedir is not None:
        filename = f'{savedir}/vZKL_effect_rmin-{rmin:.2f}_rmax-{rmax:.2f}'
        if 'm_orb' in root:
            filename += '_(single star)'
        filename += '.pdf'
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plateau_histogram(dirs: list = ['/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data-lucas/amuseKL-output/vary_radii-True/snapshots-rmin7.265-rmax12.025/',
                    '/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data-lucas/amuseKL-output/r_min-7.589-r_max-10.989/snapshots-rmin7.589-rmax10.989/',
                    '/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data/r_min-7.625-r_max-12.025-vary_radii-True/snapshots-rmin7.763-rmax11.723',
                    '/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data/r_min-7.625-r_max-12.025-vary_radii-True/snapshots-rmin8.027-rmax11.107'],
                    savedir: str | None = None):
    """ Creates a plot with multiple histograms signifying a PDF of disk particle distance from binary for specified runs.
        Requires a list of directories leading to the snapshots of a simulation, even if it is just one.

    Args:
        dirs (np.ndarray, optional): A list of directories leading to the snapshots of a simulation. Defaults to ['/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data-lucas/amuseKL-output/vary_radii-True/snapshots-rmin7.265-rmax12.025/', '/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data-lucas/amuseKL-output/r_min-7.589-r_max-10.989/snapshots-rmin7.589-rmax10.989/', '/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data/r_min-7.625-r_max-12.025-vary_radii-True/snapshots-rmin7.763-rmax11.723', '/data2/AMUSE-KL-vdvuurst-pouw-badoux/Data/r_min-7.625-r_max-12.025-vary_radii-True/snapshots-rmin8.027-rmax11.107'].
        savedir (str | None, optional): Path to directory where the resulting plot may be saved in. If None, will only show the plot. Defaults to None.
    """
    dirs = np.array(dirs)
    # We choose to analyze the disk at 25 kyr, in the middle of the observed plateaus
    file_idx = 25000

    #Create labels based on what snapshot file we are looking at
    labels = []
    for dir in dirs:
        rmin,rmax = os.path.split(dir)[-1].split('-')[1:]
        if 'rmin' in rmin:
            rmin= rmin[4:]
        if 'rmax' in rmax:
            rmax = rmax[4:]
        rmin,rmax = float(rmin),float(rmax)
        label = r'R_{\rm min}='+f'{rmin:.2f}'+r' AU,$\,$ R_{\rm max}='+f'{rmax:.2f} AU'
        labels.append(label)

    medians = np.zeros(len(dirs))
    stds = np.zeros(len(dirs))

    plt.figure(figsize=(8,6))
    for i, snapshot_dir in enumerate(dirs):
        file = get_sorted_files(snapshot_dir)[file_idx] #take the snapshot at file_idx years, i.e., 25 kyr
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

    if savedir is not None:
        filename = f'{savedir}/plateau_histogram.pdf'
        plt.savefig(filename, bbox_inches = 'tight')

    plt.show()

    inners_1sig = medians - stds / 2
    outers_1sig = medians + stds / 2
    inners_2sig = medians - stds
    outers_2sig = medians + stds
    print(f'The inner and outer radii at 1 sigma are {inners_1sig} and {outers_1sig}. This gives a mean of {np.mean(np.array(inners_1sig)):.2f} and {np.mean(np.array(outers_1sig)):.2f} AU.')
    print(f'The inner and outer radii at 2 sigma are {inners_2sig} and {outers_2sig}. This gives a mean of {np.mean(np.array(inners_2sig)):.2f} and {np.mean(np.array(outers_2sig)):.2f} AU.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze data gathered from simulations. Works for either a single run'+ 
                                     ' or for a directory of runs.')
    parser.add_argument('--root_folder', type = str, default='/data2/AMUSE-KL-vdvuurst-pouw-badoux/', help='Directory in which simulation output is stored.', required = True)
    parser.add_argument('--plot_KL_effect', type = bool, default = False, help = 'Controls whether to run the code that plots the KL effect on orbital elements. WARNING: takes a long time to run.')
    parser.add_argument('--plot_plateau_hist', type = bool, default = False, help = 'Controls whether to run the code that plots the histogram of bound particle plateaus.')
    parser.add_argument('--plot_nbound_rhalf', type = bool, default = False, help = 'Controls whether to run the code that plots the nbound over time')
    parser.add_argument('--plot_energy_error', type = bool, default = False, help = 'Controls whether to run the code that plots the energy error over time.')
    parser.add_argument('--plot_hydro_val', type = bool, default = True, help = 'Controls whether to run the code that plots Nbound over time for various disk particle numbers.')
    parser.add_argument('--save_dir', type = str, default = 'none', help = 'Output directory where plots will be stored. If "none", plots will only be output once and not stored.')
    args = parser.parse_args()

    if args.save_dir.lower() == 'none':
        save_dir = None
    else:
        save_dir = args.save_dir

    # Retrieve all snapshot directories and root folders from the given root folder.
    # In case the root of only a single run was given, run_roots will be a single entry list with that root
    snapshot_dirs,run_roots = get_snapshot_paths(args.root_folder)

    if args.plot_hydro_val:
        hydro_validation_plot(run_roots, save_dir)

    # For all recognized runs in a root directory, make the desired plots. Note the default values in the parser.
    for root in run_roots:

        if args.plot_KL_effect:
            KL_effect_plot(root, save_dir)

        if args.plot_plateau_hist:
            #TODO: add code (with input) that lets the user decide what simulations to include in the histogram
            plateau_histogram(savedir=save_dir)

        if args.plot_nbound_rhalf:
            rectify_filenames(root) # Sometimes, stored .npy files are inconsistent, this is made to rectify any mistakes there
            nbound_rhalf_plot(root, save_dir)
        
        if args.plot_energy_error:
            rectify_filenames(root)
            energy_error_plot(root, save_dir)