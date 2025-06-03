from run_sims import SimulationRunner
from main import orbital_period

from amuse.io import read_set_from_file
from amuse.units import units, nbody_system
from amuse.ext.sink import new_sink_particles

import numpy as np

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a production simulation for the AMUSE circumbinary disk project.')
    parser.add_argument('--save_dir', type=str, default='home/s2562898/data1/AMUSE_CB_disk/production_run',
                        help='Directory to save simulation data.')
    parser.add_argument('--initial_conditions', type=str, default='/home/s2562898/data1/AMUSE_CB_disk/amuse-KL/initial_conditions/...',)
    args = parser.parse_args()

    save_dir = args.save_dir
    initial_conditions_file = args.initial_conditions

    #check if save_dir exists, if not create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    else:
        #check for existing snapshot folders
        snapshot_folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f)) and f.startswith('snapshots')]
        if len(snapshot_folders) > 0:
            #find the latest snapshot folder
            snapshot_folders = sorted(snapshot_folders, key=lambda x: float(x.split('_')[1]))
            latest_snapshot_folder = snapshot_folders[-1]
            save_dir = os.path.join(save_dir, latest_snapshot_folder)
            print(f"Found latest existing snapshot folder: {save_dir}, starting from there.")

            #find latest snapshot file in the latest snapshot folder
            snapshot_files = os.listdir(save_dir)
            snapshot_files = sorted(snapshot_files, key=lambda x: float(x.split('_')[1].split('.hdf5')[0]))
            latest_snapshot_file = snapshot_files[-1]

            #find the final snapshot time from the latest snapshot file
            latest_file_time = float(latest_snapshot_file.split('_')[1].split('.hdf5')[0]) | units.day
            latest_folder_time = float(latest_snapshot_folder.split('_')[1]) | units.yr
            final_snapshot_time = latest_folder_time + latest_file_time # this is the time from the last snapshot file
            print(latest_file_time, latest_folder_time, final_snapshot_time)

            #starting from the latest snapshot file
            print(f"Using latest snapshot file: {os.path.join(save_dir, latest_snapshot_file)}")
            initial_conditions = read_set_from_file(os.path.join(save_dir, latest_snapshot_file))

            #create a new folder for the next run
            save_folder = os.path.join(args.save_dir, f'snapshots_{final_snapshot_time.value_in(units.yr)}')
            os.mkdir(save_folder)

        else:
            #start from initial conditions file if no snapshot folders exist
            print(f"No existing snapshot folders found in {save_dir}. Starting fresh.")

            #load initial conditions from the provided file
            initial_conditions = read_set_from_file(initial_conditions_file)
            print(f"Using initial conditions from: {initial_conditions_file}")

            #create a new folder for the first run
            save_folder = os.path.join(save_dir, 'snapshots_0')
            os.mkdir(save_folder)
            final_snapshot_time = 0 | units.yr


    print(f'Saving snapshots to: {save_folder}')
    print(f'Final snapshot time: {final_snapshot_time.in_(units.yr)}')

    #divide particles into subsets
    smbh_and_orbiter = initial_conditions[initial_conditions.mass > 0.1 | units.MSun]
    disk = initial_conditions[initial_conditions.mass < 0.1 | units.MSun]

    #reinitialize sink particles
    sink_rads = [18, 1.91, 0.68] | units.Rsun # SMBH, primary, secondary
    smbh_and_orbiter = new_sink_particles(smbh_and_orbiter, sink_radius=sink_rads)

    converter = nbody_system.nbody_to_si(3.53 | units.Msun, 44e-3 | units.parsec)

    binary_period = orbital_period(3.53 | units.Msun, 1.59 | units.AU)

    hydro_timestep = 0.01 * binary_period
    bridge_timestep = 0.1 * binary_period
    diagnostic_timestep = 1 | units.yr
    time_end = 3e5 | units.yr

    # run the simulation
    runner = SimulationRunner(
        smbh_and_orbiter,
        disk,
        converter,
        hydro_timestep=hydro_timestep,
        gravhydro_timestep=bridge_timestep,
        diagnostic_timestep=diagnostic_timestep,
        time_end=time_end,
        gravity_code='Hermite'
    )

    #run simulation
    model_time, grav_energy, times, accreted = runner.run_gravity_hydro_bridge_stopping_condition(
        save_folder=save_folder, N_init=int(1e5), SLURM_time_limit=168)
    
    total_sim_time = model_time + final_snapshot_time

    np.save(os.path.join(save_dir, f'{float(total_sim_time.value_in(units.yr))}_grav_energy.npy'), grav_energy.value_in(units.J))
    np.save(os.path.join(save_dir, f'{float(total_sim_time.value_in(units.yr))}_times.npy'), times.value_in(units.yr))
    np.save(os.path.join(save_dir, f'{float(total_sim_time.value_in(units.yr))}_accreted.npy'), accreted)

    print(f'Run completed. Time simulated this run: {model_time}')
    print(f'Total time simulated: {total_sim_time}')

    if model_time + final_snapshot_time >= time_end:
        print("-----------------------------------------------------------")
        print("Simulation completed successfully! No more requeues needed!")
        print("Have fun analyzing the data!")
        print("-----------------------------------------------------------")