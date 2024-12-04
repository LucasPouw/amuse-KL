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
    parser.add_argument('--r_min', type=float, default=6.55, help='Inner radius of the hydro disk in AU')
    parser.add_argument('--r_max', type=float, default=13.35, help='Outer radius of the hydro disk in AU')
    parser.add_argument('--m_disk', type=float, default=1.6e-6, help='Hydro disk mass in solar masses')
    parser.add_argument('--n_disk', type=int, default=int(1e3), help='Number of sph particles in the hydro disk')
    parser.add_argument('--dt', type=float, default=1, help='Timestep for saving and plotting diagnostics in years')
    parser.add_argument('--t_end', type=float, default=5, help='End time of the simulation in years')
    parser.add_argument('--image_dir',type=str,default='./images2/',help='Directory of plot for movie making')
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

    binary_period = orbital_period(sum(args.m_orb) | units.Msun, inner_semimajor_axis)  # This still assumes circular orbits
    hydro_timestep = 0.01 * binary_period     # Still fiducial value
    gravhydro_timestep = 0.1 * binary_period  # Same
    
    
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

    smbh_and_binary, disk, converter = ShaiHulud.make_system()
    
    # v = smbh_and_binary.velocity
    # vdisk = disk.velocity

    # plt.figure(figsize=(8,6))
    # plt.scatter(smbh_and_binary.x.value_in(units.AU)[1:], 
    #             smbh_and_binary.y.value_in(units.AU)[1:], 
    #             c=v.lengths().value_in(units.kms)[1:],
    #             s=np.log10(smbh_and_binary[1:].mass.number) + 10)
    # plt.scatter(smbh_and_binary.x.value_in(units.AU)[0],
    #             smbh_and_binary.y.value_in(units.AU)[0], 
    #             c='black', 
    #             s=np.log10(smbh_and_binary[0].mass.number) + 10)
    # plt.scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1, c=vdisk.lengths().value_in(units.kms))
    # plt.xlim(-10000, 10000)
    # plt.ylim(-10000, 10000)
    # plt.colorbar()
    # plt.savefig('test1.png')
    # plt.close()

    # plt.figure()
    # plt.scatter(smbh_and_binary.x.value_in(units.AU), smbh_and_binary.z.value_in(units.AU))
    # plt.scatter(disk.x.value_in(units.AU), disk.z.value_in(units.AU), s=1)
    # plt.xlim(-10000, 10000)
    # plt.ylim(-10000, 10000)
    # plt.savefig('test2.png')
    # plt.close() 


    # plt.figure(figsize=(8,6))
    # plt.scatter(smbh_and_binary.x.value_in(units.AU)[1:2], smbh_and_binary.y.value_in(units.AU)[1:2],zorder=100)
    # plt.scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1)
    # # plt.ylim(5300,5500)
    # # plt.xlim(-7300, -7200)
    # plt.colorbar()
    # plt.show()

    # plt.figure(figsize=(8,6))
    # plt.scatter(smbh_and_binary.x.value_in(units.AU)[1:2], smbh_and_binary.z.value_in(units.AU)[1:2],zorder=100)
    # plt.scatter(disk.x.value_in(units.AU), disk.z.value_in(units.AU), s=1)
    # plt.show()

    # plt.figure(figsize=(8,6))
    # plt.scatter(smbh_and_binary.x.value_in(units.AU)[1:2], smbh_and_binary.y.value_in(units.AU)[1:2],zorder=100)
    # plt.scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1)
    # # plt.ylim(5300,5500)
    # # plt.xlim(-7300, -7200)
    # plt.colorbar()
    # plt.show()

    
    if not os.path.isdir(args.image_dir):  # kinda ugly
        os.mkdir(args.image_dir)

    if len(os.listdir(args.image_dir)) != 0: 
        print(f'Found existing image(s) in {args.image_dir}, removing them...')
        files = glob.glob(args.image_dir+'/*.png')
        for f in files:
            os.remove(f)

    runner = SimulationRunner(smbh_and_binary,
                              disk,
                              converter,
                              hydro_timestep,
                              gravhydro_timestep,
                              diagnostic_timestep,
                              time_end)
    

    movie_kwargs = {'image_folder':args.image_dir, 'video_name': 'disk-evolution.avi', 'fps': 10}
    runner.run_gravity_hydro_bridge(movie_kwargs)

    # plt.figure(figsize=(8,6))
    # plt.scatter(smbh_and_binary.x.value_in(units.AU)[1:2], smbh_and_binary.y.value_in(units.AU)[1:2],zorder=100)
    # plt.scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1)
    # # plt.ylim(5300,5500)
    # # plt.xlim(-7300, -7200)
    # plt.colorbar()
    # plt.show()

    # plt.figure(figsize=(8,6))
    # plt.scatter(smbh_and_binary.x.value_in(units.AU)[1:2], smbh_and_binary.z.value_in(units.AU)[1:2],zorder=100)
    # plt.scatter(disk.x.value_in(units.AU), disk.z.value_in(units.AU), s=1)
    # plt.show()
