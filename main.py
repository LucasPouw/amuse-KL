import os
if 'amuse-KL' in os.getcwd(): 
    os.chdir('..' ) #move one directory upwards to avoid saving stuff in the github repo if relevant
print(f'Saving all files in: {os.getcwd()}\n')

from make_system import SystemMaker
from run_sims import SimulationRunner
import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units
import argparse
import glob

import time


Rmin_stable = 6.55 |units.AU  # minimum stable radius from MA criterion (TODO: change to couple with stable_radii.ipynb)
Rmax_stable = 13.35 |units.AU  # maximum stable radius from Hill radius (TODO: ^^)


def orbital_period(mass, radius):
    return np.sqrt(radius.value_in(units.AU)**3 / mass.value_in(units.MSun)) | units.yr

def parse_args():
    
    parser = argparse.ArgumentParser(description='Run hydro disk around a star or binary that is orbiting an SMBH')
    parser.add_argument('--m_smbh', type=float, default=4.297e6, help='SMBH mass in solar masses')
    parser.add_argument('--a_out', type=float, default=44e-3, help='Semimajor axis of the orbit around the SMBH in pc')
    parser.add_argument('--e_out', type=float, default=0.32, help='Eccentricity of the orbit around the SMBH')
    parser.add_argument('--m_orb', type=float, nargs='+', default=[2.8, 0.73], help='Mass(es) of orbiter(s) around the SMBH in solar masses in order [primary, secondary].')  # 2.8, 0.73
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
    parser.add_argument('--vary_radii',type=bool, default = False, help = 'If True, will start simulating with Rmin and Rmax for the disk and introduce stopping conditions for unbound particles, reducing the disk width and re-starting the simulation.')
    return parser.parse_args()


if __name__ == '__main__':
    
    start = time.time()


    args = parse_args()

    #Adding units to arguments
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

    binary_period = orbital_period(sum(args.m_orb) | units.Msun, inner_semimajor_axis)  # This still assumes circular orbits
    
    hydro_timestep = 0.01 * binary_period     # Still fiducial value
    gravhydro_timestep = 0.1 * binary_period  # Same
    print(f'HYDRO TIMESTEP: {hydro_timestep.value_in(units.yr):.3f} year, GRAVHYDRO TIMESTEP: {gravhydro_timestep.value_in(units.yr):.3f} year')
    
    
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

    if args.no_disk:
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
                              args.no_disk)

    if args.no_disk:
        runner.run_gravity_no_disk(args.file_dir)
    else:
        if not args.vary_radii:
            # runner.run_gravity_hydro_bridge(args.file_dir)
            _ = runner.run_gravity_hydro_bridge_stopping_condition(args.file_dir, args.n_disk)
        else:
            print('RUNNING WITH ADDITIONAL STOPPING CONDITION: IF HALF OR MORE OF THE SPH PARTICLES IN THE DISK IS UNBOUND, STOP.')
            print(f'WE ARE IN {os.getcwd()}')
            N_bound_over_time,N_lost_inner,N_lost_outer,sim_time = runner.run_gravity_hydro_bridge_stopping_condition(args.file_dir,
                                                                                                           args.n_disk)
            bound_fraction = N_bound_over_time[-1] / args.n_disk
            total_unbound_cases = N_lost_inner + N_lost_outer
            inner_fraction, outer_fraction = N_lost_inner / total_unbound_cases , N_lost_outer / total_unbound_cases

            #Extract array of half particle radii over time and save, which overwrites any other file with the same name
            Rhalf_array = runner.Rhalf_values
            Rhalf_filepath = os.path.join(os.getcwd(),f'Rhalf_{ShaiHulud.disk_inner_radius.value_in(units.AU)}-{ShaiHulud.disk_outer_radius.value_in(units.AU)}.npy')
            print(Rhalf_filepath)
            np.save(Rhalf_filepath,Rhalf_array)

            #also save N_bound_over_time for later data processing
            Nbound_filepath = os.path.join(os.getcwd(),f'Nbound_{ShaiHulud.disk_inner_radius.value_in(units.AU)}-{ShaiHulud.disk_outer_radius.value_in(units.AU)}.npy')
            np.save(Nbound_filepath,N_bound_over_time)

            print()
            print(f'Bound fraction: {bound_fraction}, inward fraction: {inner_fraction}, outward fraction: {outer_fraction}.')
            while bound_fraction == 0.5 or sim_time < time_end: #e.g. was the additional stopping condition called or did the simulation not run to its end
                #since inner_fraction and outer_fraction sum to 0.5, double them - now they signify relative fractions of the total decay

                #shrink the disk by a total of 0.5 AU, inner and outer disk relative to the number of lost particles
                ShaiHulud.disk_inner_radius += inner_fraction * 0.5|units.AU
                ShaiHulud.disk_outer_radius -= outer_fraction * 0.5|units.AU
                print(f'STOPPING CONDITION REACHED after t = {sim_time.in_(units.yr)}. ' + 
                      f'RUNNING AGAIN WITH Rmin = {ShaiHulud.disk_inner_radius} and Rmax = {ShaiHulud.disk_outer_radius}.')
                print()

                smbh_and_binary, disk, converter = ShaiHulud.make_system()
                runner = SimulationRunner(smbh_and_binary,
                                disk,
                                converter,
                                hydro_timestep,
                                gravhydro_timestep,
                                diagnostic_timestep,
                                time_end,
                                args.no_disk)
                
                N_bound_over_time,N_lost_inner,N_lost_outer, sim_time = runner.run_gravity_hydro_bridge_stopping_condition(args.file_dir,
                                                                                                                 args.n_disk)
                
                bound_fraction = N_bound_over_time[-1] / args.n_disk
                total_unbound_cases = N_lost_inner + N_lost_outer
                inner_fraction, outer_fraction = N_lost_inner / total_unbound_cases , N_lost_outer / total_unbound_cases

                #Extract array of half particle radii over time and save, which overwrites any other file with the same name
                Rhalf_array = runner.Rhalf_values
                Rhalf_filepath = os.path.join(os.getcwd(),f'Rhalf_{ShaiHulud.disk_inner_radius}-{ShaiHulud.disk_outer_radius}.npy')
                np.save(Rhalf_filepath,Rhalf_array)

                #also save N_bound_over_time for later data processing
                Nbound_filepath = os.path.join(os.getcwd(),f'Nbound_{ShaiHulud.disk_inner_radius}-{ShaiHulud.disk_outer_radius}.npy')
                np.save(Nbound_filepath,N_bound_over_time)
            
                print(f'Bound fraction: {bound_fraction}, inward fraction: {inner_fraction}, outward fraction: {outer_fraction}.')
                print()



    end = time.time()
    print(f'Elapsed time: {end-start} seconds')