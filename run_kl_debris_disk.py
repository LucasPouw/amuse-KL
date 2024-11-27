# the following fixes are highly recommended

#allow oversubscription for openMPI
import os
os.environ["OMPI_MCA_rmaps_base_oversubscribe"]="true"

# use lower cpu resources for idle codes
from amuse.support import options
options.GlobalOptions.instance().override_value_for_option("polling_interval_in_milliseconds", 10)

import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants
from amuse.lab import Particles
from amuse.ext.orbital_elements import generate_binaries, orbital_elements

from amuse.community.huayno.interface import Huayno
from amuse.units import nbody_system

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KL on debris disk')
    parser.add_argument('--n_pl', type=int, default=61, help='Number of planetesimals')
    parser.add_argument('--total_mass_pl', type=float, default=1.6, help='Total mass of planetesimals in e-6 Msun')
    parser.add_argument('--a_disk', type=float, default=0.4, help='Semi-major axis of planetesimals in AU')
    parser.add_argument('--stepsize', type=float, default=100, help='Stepsize in yr')
    args = parser.parse_args()

    n_pl = args.n_pl
    total_mass_pl = args.total_mass_pl * 1e-6 | units.MSun
    a_disk = args.a_disk | units.AU
    stepsize = args.stepsize

    print(f"Running with n_pl={n_pl}, total_mass_pl={total_mass_pl}, a_disk={a_disk}, stepsize={stepsize}")

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


    SMBH_ORBITAL_RADIUS = 44 * 1e-3 | units.parsec
    SMBH_ECCENTRICITY = 0.32

    PRIMARY_MASS = 2.8 | units.MSun
    SECONDARY_MASS = 0.73 | units.MSun

    BINARY_SEPARATION = 1.59 | units.AU
    BINARY_INCLINATION = 102.55 | units.deg
    BINARY_ECCENTRICITY = 0.45
    BINARY_PERIAPSE = 311.75 | units.deg


    SMBH_MASS = 4.297e6 | units.MSun

    def orbital_period(mass, radius):
        return np.sqrt(radius.value_in(units.AU)**3 / mass.value_in(units.MSun)) | units.yr


    smbh_binary_period = orbital_period(SMBH_MASS, SMBH_ORBITAL_RADIUS)
    binary_period = orbital_period(PRIMARY_MASS + SECONDARY_MASS, BINARY_SEPARATION)
    kozai_timescale = smbh_binary_period**2 / binary_period

    smbh_and_binary = Particles(1)
    smbh = smbh_and_binary[0]
    smbh.mass = SMBH_MASS
    smbh.position = (0, 0, 0) | units.km
    smbh.velocity = (0, 0, 0) | units.km / units.s

    primary, secondary = generate_binaries(PRIMARY_MASS, 
                                            SECONDARY_MASS, 
                                            BINARY_SEPARATION, 
                                            eccentricity=BINARY_ECCENTRICITY, 
                                            true_anomaly=0 | units.rad, # fixed value for reproducibility
                                            inclination=BINARY_INCLINATION,
                                            argument_of_periapsis=BINARY_PERIAPSE)
    binary_com_velocity = (constants.G * smbh.mass / SMBH_ORBITAL_RADIUS).sqrt()

    primary.position += (1, 0, 0) * SMBH_ORBITAL_RADIUS
    secondary.position += (1, 0, 0) * SMBH_ORBITAL_RADIUS

    primary.velocity += (0, 1, 0) * binary_com_velocity
    secondary.velocity += (0, 1, 0) * binary_com_velocity

    smbh_and_binary.add_particle(primary)
    smbh_and_binary.add_particle(secondary)

    smbh_and_binary.move_to_center()

    #add planetesimal disk around primary star
    planetesimals = Particles()
    true_anomalies = np.linspace(0, 2 * np.pi, n_pl) | units.rad
    true_anomalies = true_anomalies[0:-1]
    mass_pl = total_mass_pl / n_pl

    for f in true_anomalies:
        _, pl = generate_binaries(PRIMARY_MASS, mass_pl, a_disk, true_anomaly=f, eccentricity=0, inclination=BINARY_INCLINATION, argument_of_periapsis=BINARY_PERIAPSE)
        pl.position += smbh_and_binary[1].position
        pl.velocity += smbh_and_binary[1].velocity
        planetesimals.add_particle(pl)

    smbh_and_binary.add_particles(planetesimals)

    converter = nbody_system.nbody_to_si(smbh_and_binary.mass.sum(), SMBH_ORBITAL_RADIUS)

    gravity = Huayno(converter)
    gravity.set_integrator('OK')

    gravity.particles.add_particles(smbh_and_binary)

    channel = gravity.particles.new_channel_to(smbh_and_binary)

    start_time = 0  # yr
    end_time = int(2e5)  # yr
    init_times = np.arange(start_time, end_time, stepsize) | units.yr
    eccentricities = []
    inclinations = [] | units.deg
    semimajors = [] | units.AU
    times = [] | units.yr

    for i,time in enumerate(tqdm(init_times)):
        gravity.evolve_model(time)
        channel.copy()        
        
        binary = smbh_and_binary[1:3]
        _, _, axs, ecc, _, inc, _, _  = orbital_elements(binary, G=constants.G)  # mass1, mass2, semimajor_axis, eccentricity, true_anomaly.value_in(units.deg), inclination.value_in(units.deg), long_asc_node.value_in(units.deg), arg_per.value_in(units.deg))
        eccentricities.append(ecc)
        inclinations.append(inc)
        semimajors.append(axs)
        times.append(time)

    gravity.stop()


    #plot the e and i evolution
    fig, ax1 = plt.subplots(figsize=(20,6))
    ax2 = ax1.twinx()

    ax1.plot(times.value_in(units.yr), np.array(eccentricities), color='red', label='Eccentricity')
    ax2.plot(times.value_in(units.yr), inclinations.value_in(units.deg), color='blue', label='Inclination')

    ax1.set_xlabel('Time [yr]')
    ax1.set_ylabel('Eccentricity', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim(0,1)

    ax2.set_ylabel('Inclination [deg]', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 180)
    fig.savefig(f'figures/ecc_inc_{a_disk}_{n_pl}.png')
    # plt.show()

    plt.figure(figsize=(10,10))
    plt.scatter(smbh_and_binary.x.value_in(units.AU), smbh_and_binary.y.value_in(units.AU), c=smbh_and_binary.mass.value_in(units.Msun), alpha=0.5, vmax = 2, vmin=0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig(f'figures/pos_after_{a_disk}_{n_pl}.png')
