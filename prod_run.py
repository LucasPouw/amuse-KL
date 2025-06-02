from run_sims import SimulationRunner
from main import orbital_period

from amuse.io import read_set_from_file
from amuse.units import units, nbody_system

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a production simulation for the AMUSE circumbinary disk project.')
    parser.add_argument('--save_dir', type=str, default='home/s2562898/data1/AMUSE_CB_disk/production_run',
                        help='Directory to save simulation data.')
    parser.add_argument('--initial_conditions', type=str, default='...',)
    args = parser.parse_args()

    save_dir = args.save_dir
    initial_conditions_file = args.initial_conditions

    #check if there are files in save_dir
    files = os.listdir(save_dir)
    if len(files) == 0: # first run
        initial_conditions = read_set_from_file(initial_conditions_file)
    else:
        # load the last file in the directory and use it as the initial conditions
        files.sort()
        last_file = files[-1]
        initial_conditions = read_set_from_file(os.path.join(save_dir, last_file))

    #divide particles into subsets
    smbh_and_orbiter = initial_conditions[initial_conditions.mass > 0.1 | units.MSun]
    disk = initial_conditions[initial_conditions.mass < 0.1 | units.MSun]

    converter = nbody_system.nbody_to_si(3.53 | units.Msun, 44e-3 | units.parsec)

    binary_period = orbital_period(3.53 | units.Msun, 1.59 | units.AU)

    hydro_timestep = 0.01 * binary_period
    bridge_timestep = 0.1 * binary_period
    diagnostic_timestep = 10 * binary_period
    time_end = 5e5 | units.yr

    # run the simulation
    runner = SimulationRunner(
        smbh_and_orbiter,
        disk,
        converter,
        hydro_timestep=hydro_timestep,
        bridge_timestep=bridge_timestep,
        diagnostic_timestep=diagnostic_timestep,
        time_end=time_end,
        gravity_code='Hermite'
    )

    runner.run_gravity_hydro_bridge_stopping_condition(save_folder=save_dir, N_init=int(1e5), SLURM_time_limit=168)