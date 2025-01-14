import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units
from amuse.io import read_set_from_file
import os
from tqdm import tqdm
import argparse
from plotter import * # Importing also changes mpl.rcParams to make plots nice
from data_processing import *


def _append_bound_disk_elements(snapshot, eccs_disk, incs_disk, ascs_disk, peris_disk):
    _, _, _, ecc, _, inc, asc, peri = get_disk_orbital_elements(snapshot)
    bound = ecc < 1

    eccs_disk.append(np.median(ecc[bound]))
    incs_disk.append(np.median(inc[bound].value_in(units.deg)))
    ascs_disk.append(np.median(asc[bound].value_in(units.deg)))
    peris_disk.append(np.median(peri[bound].value_in(units.deg)))


def _append_binary_elements(snapshot, eccs_binary, incs_binary, ascs_binary, peris_binary):
    _, _, _, ecc, _, inc, asc, peri = get_binary_orbital_elements(snapshot)

    eccs_binary.append(ecc)
    incs_binary.append(inc.value_in(units.deg))
    ascs_binary.append(asc.value_in(units.deg))
    peris_binary.append(peri.value_in(units.deg))


def _get_paths_to_elements(root, elements):
    tail = os.path.split(root)[-1]

    # Check if the arrays containing the relevant data *all* exist already or not
    flag = True
    paths = []
    for attribute in elements:
        path = os.path.join(root,f'{attribute}-{tail}.npy')
        paths.append(path)
        flag *= os.path.isfile(path)
    
    return flag, paths


def _load_elements(paths):
    eccs = np.load(paths[0])
    incs = np.load(paths[1])
    ascs = np.load(paths[2])
    peris = np.load(paths[3])
    return eccs, incs, ascs, peris


def _get_time_array(root, npoints):
    try:
        timepath = [t for t in os.listdir(root) if 'times' in t][0] #extract the time array
        times = np.load(timepath)
    except: #i.e., time array does not exist, we create it as we know it should be of the shape as the orbital parameters in steps of 1
        print(f'WARNING: Time array at {root} does not exist, assuming dt=1 yr')
        times = np.arange(1, npoints+1)
    return times


def average_every(array, decimation):
    array = array[:int(len(array)//decimation * decimation)]  # Make array size a multiple of decimation
    return array.reshape(-1, decimation).mean(axis=1)


def _check_orbital_element_files_binary(bin_root: str, sin_root: str, binary_snapshot_path: str, single_snapshot_path: str):

    binary_elements = ['eccs','incs','ascs','peris',                                  # Disk orbital elements around binary
                         'eccs-binary', 'incs-binary', 'ascs-binary', 'peris-binary'] # Binary orbital elements
    single_elements = ['eccs-single', 'incs-single', 'ascs-single', 'peris-single']   # Disk orbital elements around a single star

    bin_flag, binary_paths = _get_paths_to_elements(bin_root, binary_elements)
    sin_flag, single_paths = _get_paths_to_elements(sin_root, single_elements)

    if bin_flag: # i.e. all arrays already exist for this root
        eccs_disk, incs_disk, ascs_disk, peris_disk = _load_elements(np.array(binary_paths)[:4])
        eccs_binary, incs_binary, ascs_binary, peris_binary = _load_elements(np.array(binary_paths)[4:])    
    else: #the orbital parameters are not yet defined (at least not all of them), so we recalculate. This takes a while.
        print('Not all orbital elements found in the binary case. Generating files...')
        eccs_disk = []
        incs_disk = []
        ascs_disk = []
        peris_disk = []
        eccs_binary = []
        incs_binary = []
        ascs_binary = []
        peris_binary = []
        for datafile in tqdm(get_sorted_files(binary_snapshot_path)):
            snapshot = read_set_from_file(datafile)
            _append_bound_disk_elements(snapshot, eccs_disk, incs_disk, ascs_disk, peris_disk)
            _append_binary_elements(snapshot, eccs_binary, incs_binary, ascs_binary, peris_binary)

        np.save(binary_paths[0], eccs_disk)
        np.save(binary_paths[1], incs_disk)
        np.save(binary_paths[2], ascs_disk)
        np.save(binary_paths[3], peris_disk)
        np.save(binary_paths[4], eccs_binary)
        np.save(binary_paths[5], incs_binary)
        np.save(binary_paths[6], ascs_binary)
        np.save(binary_paths[7], peris_binary)

    if sin_flag:
        eccs_single, incs_single, ascs_single, peris_single = _load_elements(np.array(single_paths))
    else:
        print('Not all orbital elements found in the single star case. Generating files...')
        eccs_single = []
        incs_single = []
        ascs_single = []
        peris_single = []
        for datafile in tqdm(get_sorted_files(single_snapshot_path)):
            snapshot = read_set_from_file(datafile)
            _append_bound_disk_elements(snapshot, eccs_single, incs_single, ascs_single, peris_single)

        np.save(single_paths[0], eccs_single)
        np.save(single_paths[1], incs_single)
        np.save(single_paths[2], ascs_single)
        np.save(single_paths[3], peris_single)

    bin_time = _get_time_array(bin_root, len(eccs_binary))
    sin_time = _get_time_array(sin_root, len(eccs_single))

    return bin_time, sin_time, np.array(eccs_single), np.array(incs_single), np.array(ascs_single), np.array(peris_single), np.array(eccs_disk), np.array(incs_disk), np.array(ascs_disk), np.array(peris_disk), np.array(eccs_binary), np.array(incs_binary), np.array(ascs_binary), np.array(peris_binary)


def KL_effect_plot(bin_root: str, sin_root: str, binary_snapshot_path: str, single_snapshot_path: str, file_path: str | None = None):
    """ Creates a plot of the von Zeipel-Lidov-Kozai effect on orbital parameters of the disk, given the root in which simulation output is stored.
        If the necessary orbital elements are not yet saved in the directory, they will be calculated here.
        STRONGLY recommended to do this only for a run with a single snapshot folder specified as this function takes a while when running for the first time.

    Args:
    """
    bin_time, sin_time, eccs_single, incs_single, ascs_single, peris_single, eccs_disk, incs_disk, ascs_disk, peris_disk, eccs_binary, incs_binary, ascs_binary, peris_binary = _check_orbital_element_files_binary(bin_root, sin_root, binary_snapshot_path, single_snapshot_path)

    N = 500
    fig, ax = plt.subplots(3, 1, figsize=(10, 18))
    plot_ang_ecc(ax[0], average_every(bin_time, N) | units.yr, average_every(incs_binary, N) | units.deg, average_every(ascs_binary, N) | units.deg, average_every(peris_binary, N) | units.deg, average_every(eccs_binary, N), legend_kwargs={'loc': 'upper right'})
    plot_ang_ecc(ax[1], average_every(bin_time, N) | units.yr, average_every(incs_disk, N) | units.deg, average_every(ascs_disk, N) | units.deg, average_every(peris_disk, N) | units.deg, average_every(eccs_disk, N))
    plot_ang_ecc(ax[2], average_every(sin_time, N) | units.yr, average_every(incs_single, N) | units.deg, average_every(ascs_single, N) | units.deg, average_every(peris_single, N) | units.deg, average_every(eccs_single, N))

    ax[0].set_xlim(0, max(bin_time[-1], sin_time[-1]))
    ax[1].set_xlim(0, max(bin_time[-1], sin_time[-1]))
    ax[2].set_xlim(0, max(bin_time[-1], sin_time[-1]))

    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    
    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots comparing the KL effect in disks and binaries')
    parser.add_argument('--bin_snap_path', type = str, help = 'Directory in which snapshots to the binary+disk run can be found for the KL plot.', required=True)
    parser.add_argument('--sin_snap_path', type = str, help = 'Directory in which snapshots to the single+disk run can be found for the KL plot.', required=True)
    parser.add_argument('--bin_root', type = str, help='Directory in which the simulation output with binary star is stored.', required = True)
    parser.add_argument('--sin_root', type = str, help='Directory in which the simulation output with single star is stored.', required = True)
    parser.add_argument('--file_path', type = str, default = 'none', help = 'Full path to where plot is stored. If "none", plots will only be output once and not stored.')
    args = parser.parse_args()

    if args.file_path.lower() == 'none':
        file_path = None
    else:
        file_path = args.file_path

    KL_effect_plot(args.bin_root, args.sin_root, args.bin_snap_path, args.sin_snap_path, file_path)
