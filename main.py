## Made by Lucas Pouw, Yannick Badoux and Tim van der Vuurst for the 
## course "Simulations and Modeling in Astrophysics" '24-'25. 

import os
from make_system import SystemMaker
from run_sims import SimulationRunner
import numpy as np
from amuse.units import units
import argparse
import time
import sys
import shutil

if 'amuse-KL' in os.getcwd(): 
    os.chdir('..' ) # Move one directory upwards to avoid saving stuff in the github repo if relevant


def orbital_period(mass, radius):
    """ Calculates orbital period of circularly orbiting body.

    Args:
        mass (amuse.units.quantity): Total mass involved with mass units
        radius (amuse.units.quantity): Radius of orbit in length units.

    Returns:
        amuse.units.quantity: Orbital period in years. 
    """
    return np.sqrt(radius.value_in(units.AU)**3 / mass.value_in(units.MSun)) | units.yr

def get_parser():
    """Makes the argparser for terminal control of the simulation. 

    Returns:
        argparse.ArgumentParser: Argparser with relevant arguments and given values.
    """
    parser = argparse.ArgumentParser(description='Run hydro disk around a star or binary that is orbiting an SMBH')
    parser.add_argument('--m_smbh',     type=float, default=4.297e6,        help='SMBH mass in solar masses')
    parser.add_argument('--a_out',      type=float, default=44e-3,          help='Semimajor axis of the orbit around the SMBH in pc')
    parser.add_argument('--e_out',      type=float, default=0.32,           help='Eccentricity of the orbit around the SMBH')
    parser.add_argument('--m_orb',      type=float, default=[2.8, 0.73],    help='Mass(es) of orbiter(s) around the SMBH in solar masses in order [primary, secondary].', nargs='+', )  # 2.8, 0.73
    parser.add_argument('--a_in',       type=float, default=1.59,           help='Semimajor axis of the binary orbit in AU')
    parser.add_argument('--e_in',       type=float, default=0.45,           help='Eccentricity of the binary orbit')
    parser.add_argument('--i_mut',      type=float, default=102.55,         help='Mutual inclination of the inner and outer orbits in deg')
    parser.add_argument('--peri',       type=float, default=311.75,         help='Argument of periapse of the inner orbit in deg')
    parser.add_argument('--r_min',      type=float, default=4.45,           help='Inner radius of the hydro disk in AU')
    parser.add_argument('--r_max',      type=float, default=13.35,          help='Outer radius of the hydro disk in AU')
    parser.add_argument('--m_disk',     type=float, default=1.6e-6,         help='Hydro disk mass in solar masses')
    parser.add_argument('--n_disk',     type=int,   default=int(1e3),       help='Number of sph particles in the hydro disk')
    parser.add_argument('--dt',         type=float, default=1,              help='Timestep for saving and plotting diagnostics in years')
    parser.add_argument('--t_end',      type=float, default=250000,         help='End time of the simulation in years')
    parser.add_argument('--file_dir',   type=str,   default=os.getcwd(),    help='Directory for the folder containing all output')
    parser.add_argument('--no_disk',    type=bool,  default=False,          help='If True, no disk will be created. Simulation will be run using pure gravity.')
    parser.add_argument('--vary_radii', type=bool,  default=False,          help='If True, will start simulating with Rmin and Rmax for the disk and introduce stopping conditions for unbound particles, reducing the disk width and re-starting the simulation.')
    parser.add_argument('--grav_code',  type=str,   default='Hermite',      help='Gravity code to use. Allowed choices are Huayno, Hermite and HermiteGRX', choices= ['Huayno','Hermite','HermiteGRX'])
    parser.add_argument('--SLURM_time_limit', type=float, default=0,        help='Time limit for SLURM job in hours. If 0, no time limit is set.')
    return parser


if __name__ == '__main__':   
    start = time.time() # We want to know how long the code runs for

    # Initialize args from terminal
    parser = get_parser()
    args = parser.parse_args()

    # Get all non-default input args to put in directory name
    input_vars = vars(args)
    default_vars = vars(parser.parse_args([]))
    name = ''
    for key in input_vars.keys():
        if input_vars[key] != default_vars[key]:
            name += f'{key}-{input_vars[key]}-'  # Adds extra dash at the end
    if name == '':
        name = 'default'
    else:
        name = name[:-1]  # Remove superfluous dash at the end

    args.file_dir += '/amuseKL-output/'
    print(f'All output is saved in: {args.file_dir}')
    print(f'Current run is found in sub-directory {name}\n')

    if not os.path.isdir(args.file_dir):  # Check for output folder
        os.mkdir(args.file_dir)
        pass

    # Make folder for saving run with specified initial conditions
    args.file_dir += name
    if not os.path.isdir(args.file_dir):
        os.mkdir(args.file_dir)
    else:
        inp = 'y'
        # while inp not in ['y', 'n']:
        #     inp = input(f'Directory {args.file_dir} already exists. Do you want to erase the existing files? (y/n)')
        if inp.lower() == 'y':
            print('Erasing...')
            shutil.rmtree(args.file_dir)
            os.mkdir(args.file_dir)
        else:
            sys.exit('Exiting...')

    #Adding units to arguments where relevant
    smbh_mass = args.m_smbh | units.Msun
    outer_semimajor_axis = args.a_out | units.parsec
    outer_eccentricity = args.e_out
    orbiter_masses = [mass | units.Msun for mass in args.m_orb]
    inner_semimajor_axis = args.a_in | units.AU
    inner_eccentricity = args.e_in
    mutual_inclination = args.i_mut | units.deg
    arg_of_periapse = args.peri | units.deg
    inner_radius = args.r_min | units.AU
    outer_radius = args.r_max | units.AU
    disk_mass = args.m_disk | units.Msun
    diagnostic_timestep = args.dt | units.yr
    time_end = args.t_end | units.yr

    #Initializing timesteps
    binary_period = orbital_period(sum(args.m_orb) | units.Msun, inner_semimajor_axis) # Assumes circular orbit but works fine 
    hydro_timestep = 0.01 * binary_period
    gravhydro_timestep = 0.1 * binary_period
    print(f'\nHYDRO TIMESTEP: {hydro_timestep.value_in(units.yr):.3f} year, GRAVHYDRO TIMESTEP: {gravhydro_timestep.value_in(units.yr):.3f} year\n')
    
    # Make system, see make_system.py for details
    ShaiHulud = SystemMaker(smbh_mass,
                            orbiter_masses,
                            outer_semimajor_axis,
                            outer_eccentricity,
                            inner_semimajor_axis,
                            mutual_inclination,
                            inner_eccentricity,
                            arg_of_periapse,
                            inner_radius,
                            outer_radius,
                            disk_mass,
                            args.n_disk)  # Shai Hulud is the Maker

    # Returning terminal output specifiying what kind of run is being performed
    if args.no_disk:
        print('Initializing system WITHOUT disk...\n')
        smbh_and_binary, converter = ShaiHulud.make_system_no_disk()
        disk = None  # Variable needs to be defined to avoid error in initializing SimulationRunner
    else:
        print('Initializing system WITH disk...\n')
        smbh_and_binary, disk, converter = ShaiHulud.make_system()

    # Creating the runner of the simulation, see run_sims.py for details
    runner = SimulationRunner(smbh_and_binary,
                              disk,
                              converter,
                              hydro_timestep,
                              gravhydro_timestep,
                              diagnostic_timestep,
                              time_end,
                              args.grav_code)

    if args.no_disk: # Simple gravity-only run

        print(f'DOING A SINGLE GRAVITY-ONLY RUN UNTIL T={time_end}')
        dir_current_run = args.file_dir + f'/snapshots-rmin{args.r_min}-rmax{args.r_max}/'
        os.mkdir(dir_current_run)
        energy, times = runner.run_gravity_no_disk(dir_current_run)

        np.save(args.file_dir + f'/energy-joules-rmin{args.r_min:.3f}-rmax{args.r_max:.3f}.npy', energy.value_in(units.J))
        np.save(args.file_dir + f'/times-year-rmin{args.r_min:.3f}-rmax{args.r_max:.3f}.npy', times.value_in(units.yr))


#-----------------------STOP IF HALF DISK UNBOUND-----------------------#
    else:

        print('RUNNING WITH ADDITIONAL STOPPING CONDITION: IF HALF OR MORE OF THE SPH PARTICLES IN THE DISK IS UNBOUND, STOP.')

        # The vary_radii argument controls whether we use an additional stopping condition that stops the sim when half the disk has
        # been ejected from the system. If this is the case, we shrink the disk and re-start the run (ergo varying the radii). 

        if not args.vary_radii: # Run gravity + hydro until the end without additional stopping condition
            print(f'DOING A SINGLE GRAVHYDRO RUN UNTIL T={time_end}')
            dir_current_run = args.file_dir + f'/snapshots-rmin{args.r_min}-rmax{args.r_max}/'
            os.mkdir(dir_current_run)
            sim_time, grav_energy, times = runner.run_gravity_hydro_bridge_stopping_condition(dir_current_run, args.n_disk)

            # Save relevant data for later analysis
            np.save(args.file_dir + f'/grav-energy-joules-rmin{args.r_min}-rmax{args.r_max}.npy', grav_energy.value_in(units.J))
            np.save(args.file_dir + f'/times-year-rmin{args.r_min}-rmax{args.r_max}.npy', times.value_in(units.yr))

    #Show runtime of the code
    end = time.time()
    print(f'Elapsed time: {(end - start)/3600} hours')
