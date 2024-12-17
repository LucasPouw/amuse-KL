#### Is this block still needed? ####

# the following fixes are highly recommended

#allow oversubscription for openMPI
import os
os.environ["OMPI_MCA_rmaps_base_oversubscribe"]="true"

# use lower cpu resources for idle codes
from amuse.support import options
options.GlobalOptions.instance().override_value_for_option("polling_interval_in_milliseconds", 10)

#####################################

from make_system import SystemMaker
from run_sims import SimulationRunner
import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units
import argparse
import glob


Rmin_stable = 6.55 |units.AU  # minimum stable radius from MA criterion (TODO: change to couple with stable_radii.ipynb)
Rmax_stable = 13.35 |units.AU  # maximum stable radius from Hill radius (TODO: ^^)


def orbital_period(mass, radius):
    return np.sqrt(radius.value_in(units.AU)**3 / mass.value_in(units.MSun)) | units.yr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run hydro disk around a star or binary that is orbiting an SMBH')
    parser.add_argument('--m_smbh', type=float, default=4.297e6, help='SMBH mass in solar masses')
    parser.add_argument('--a_out', type=float, default=44e-3, help='Semimajor axis of the orbit around the SMBH in pc')
    parser.add_argument('--e_out', type=float, default=0.32, help='Eccentricity of the orbit around the SMBH')
    parser.add_argument('--m_orb', type=list, default=[2.8, 0.73], help='Mass(es) of orbiter(s) around the SMBH in solar masses in order [primary, secondary].')  # 2.8, 0.73
    parser.add_argument('--a_in', type=float, default=1.59, help='Semimajor axis of the binary orbit in AU')
    parser.add_argument('--e_in', type=float, default=0.45, help='Eccentricity of the binary orbit')
    parser.add_argument('--i_mut', type=float, default=102.55, help='Mutual inclination of the inner and outer orbits in deg')
    parser.add_argument('--peri', type=float, default=311.75, help='Argument of periapse of the inner orbit in deg')
    parser.add_argument('--r_min', type=float, default=10, help='Inner radius of the hydro disk in AU')
    parser.add_argument('--r_max', type=float, default=13.35, help='Outer radius of the hydro disk in AU')
    parser.add_argument('--m_disk', type=float, default=1.6e-6, help='Hydro disk mass in solar masses')
    parser.add_argument('--n_disk', type=int, default=int(1e3), help='Number of sph particles in the hydro disk')
    parser.add_argument('--dt', type=float, default=1, help='Timestep for saving and plotting diagnostics in years')
    parser.add_argument('--t_end', type=float, default=15, help='End time of the simulation in years')
    parser.add_argument('--file_dir',type=str,default='./snapshots-default/',help='Directory for AMUSE snapshots in hdf5 format')
    parser.add_argument('--no_disk', type=bool, default=False, help='If True, no disk will be created. Simulation will be run using pure gravity.')
    args = parser.parse_args()

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
    n_sph_particles = args.n_disk
    diagnostic_timestep = args.dt | units.yr
    time_end = args.t_end | units.yr
    no_disk = args.no_disk

    binary_period = orbital_period(sum(args.m_orb) | units.Msun, inner_semimajor_axis)  # This still assumes circular orbits
    hydro_timestep = 0.01 * binary_period     # Still fiducial value
    gravhydro_timestep = 0.1 * binary_period  # Same
    print(f'HYDRO TIMESTEP: {hydro_timestep.value_in(units.s)} s, GRAVHYDRO TIMESTEP: {gravhydro_timestep.value_in(units.s)} s')
    
    
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
                            n_sph_particles)  # Shai Hulud is the Maker

    if no_disk:
        print('Initializing system WITHOUT disk...')
        smbh_and_binary, converter = ShaiHulud.make_system_no_disk()
        disk = None  # Variable needs to be defined to avoid error in initializing SimulationRunner
    else:
        print('Initializing system WITH disk...')
        smbh_and_binary, disk, converter = ShaiHulud.make_system()
    
    if not os.path.isdir(args.file_dir):
        os.mkdir(args.file_dir)

    if len(os.listdir(args.file_dir)) != 0: 
        print(f'Found existing file(s) in {args.file_dir}, removing them...')
        files = glob.glob(args.file_dir + '*')
        for f in files:
            os.remove(f)

    runner = SimulationRunner(smbh_and_binary,
                              disk,
                              converter,
                              hydro_timestep,
                              gravhydro_timestep,
                              diagnostic_timestep,
                              time_end,
                              no_disk)

    if no_disk:
        runner.run_gravity_no_disk(args.file_dir)
    else:
        runner.run_gravity_hydro_bridge(args.file_dir)
