import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import periodogram

from amuse.units import units, constants
from amuse.ext.orbital_elements import orbital_elements
from amuse.community.huayno.interface import Huayno
from amuse.community.hermite.interface import Hermite

from make_system import SystemMaker

# We define some properties for the figures
import matplotlib as mpl
SMALL_SIZE = 10 * 2 
MEDIUM_SIZE = 12 * 2
BIGGER_SIZE = 14 * 2

plt.rc('text', usetex=True)
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

if __name__ == '__main__':
    #orbital parameters from Peissker et al. 2024
    SMBH_ORBITAL_RADIUS = 44 * 1e-3 | units.parsec
    SMBH_ECCENTRICITY = 0.32

    PRIMARY_MASS = 2.8 | units.MSun
    SECONDARY_MASS = 0.73 | units.MSun
    SMBH_MASS = 4.297e6 | units.MSun

    BINARY_SEPARATION = 1.59 | units.AU
    BINARY_ORBITAL_RADIUS = BINARY_SEPARATION
    BINARY_INCLINATION = 102.55 | units.deg
    BINARY_ECCENTRICITY = 0.45
    BINARY_PERIAPSE = 311.75 | units.deg

    def orbital_period(mass, radius):
        return np.sqrt(radius.value_in(units.AU)**3 / mass.value_in(units.MSun)) | units.yr

    smbh_binary_period = orbital_period(SMBH_MASS, SMBH_ORBITAL_RADIUS)
    binary_period = orbital_period(PRIMARY_MASS + SECONDARY_MASS, BINARY_SEPARATION)
    kozai_timescale = smbh_binary_period**2 / binary_period
    print(smbh_binary_period, binary_period)
    print(kozai_timescale.in_(units.yr))

    #generate initial conditions
    system_maker = SystemMaker(SMBH_MASS,
                            [PRIMARY_MASS, SECONDARY_MASS],
                            SMBH_ORBITAL_RADIUS,
                            SMBH_ECCENTRICITY,
                            BINARY_SEPARATION,
                            BINARY_INCLINATION,
                            BINARY_ECCENTRICITY,
                            BINARY_PERIAPSE,
                            0|units.au,
                            0|units.au,
                            10|units.kg,
                            10) # last 4 parameters are set to arbitrary values

    smbh_and_binary, converter = system_maker.make_system_no_disk()

    #specify the time steps for the simulation
    start_time = 0  # yr
    end_time = 2e5  # yr
    stepsize = 1  # yr, diagnistic timestep

    times = np.arange(start_time, end_time, stepsize) | units.yr

    #run the simulation with Huayno
    eccentricities_huayno = []
    inclinations_huayno = [] | units.deg
    semimajors_huayno = [] | units.AU

    gravity = Huayno(converter)
    gravity.particles.add_particles(smbh_and_binary)
    gravity.set_integrator('OK')

    for time in tqdm(times, miniters=1):
        gravity.evolve_model(time)
        
        binary = gravity.particles[1:]
        _, _, axs, ecc, _, inc, _, _  = orbital_elements(binary, G=constants.G)
        
        eccentricities_huayno.append(ecc)
        inclinations_huayno.append(inc)
        semimajors_huayno.append(axs)

    gravity.stop()

    #run the simulation with Hermite
    gravity2 = Hermite(converter)
    gravity2.particles.add_particles(smbh_and_binary)

    eccentricities_hermite = []
    inclinations_hermite = [] | units.deg
    semimajors_hermite = [] | units.AU

    for time in tqdm(times, miniters=1):
        gravity2.evolve_model(time)

        binary = gravity2.particles[1:]
        _, _, axs, ecc, _, inc, _, _  = orbital_elements(binary, G=constants.G)

        eccentricities_hermite.append(ecc)
        inclinations_hermite.append(inc)
        semimajors_hermite.append(axs)

    gravity2.stop()

    #plot the comparison between Huayno and Hermite
    fig, ax1 = plt.subplots(figsize=(15,7))
    ax2 = ax1.twinx()

    ax1.plot(times.value_in(units.yr), np.array(eccentricities_huayno), color='red', label='Eccentricity Huayno')
    ax2.plot(times.value_in(units.yr), inclinations_huayno.value_in(units.deg), color='blue', label='Inclination Huayno')

    ax1.plot(times.value_in(units.yr), np.array(eccentricities_hermite), color='red', linestyle=':', label='Eccentricity Hermite', lw=2)
    ax2.plot(times.value_in(units.yr), inclinations_hermite.value_in(units.deg), color='blue', linestyle=':', label='Inclination Hermite',lw=2)

    ax1.set_xlabel('Time [yr]')
    ax1.set_ylabel('Eccentricity', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim(0,1)

    ax2.set_ylabel('Inclination [deg]', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 180)

    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
            labels=ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1],
            ncols=2, loc='lower left', frameon=False)

    plt.savefig(f'grav_code_comparison{stepsize}.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig(f'grav_code_comparison{stepsize}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # plt.figure(figsize=(8,6))
    # plt.plot(times.value_in(units.yr), semimajors_huayno.value_in(units.AU), color='blue', label='Semimajor axis Huayno')
    # plt.plot(times.value_in(units.yr), semimajors_hermite.value_in(units.AU), color='blue', linestyle=':', label='Semimajor axis Hermite')
    # plt.xlabel('Time [yr]')
    # plt.ylabel('Semimajor axis [AU]')
    # plt.legend(frameon=False)
    # plt.savefig('semimajor_axis_comparison.pdf', dpi=300)
    # plt.show()

    #plot a periodogram to check if the period of the oscillations is the same
    freq, strength = periodogram(eccentricities_huayno, fs=1/stepsize)

    plt.figure(figsize=(8,6))
    plt.plot(1/freq[1:], strength[1:])

    freq, strength = periodogram(eccentricities_hermite, fs=1/stepsize)

    plt.plot(1/freq[1:], strength[1:])
    plt.xlabel('Period [yr]')
    plt.ylabel('Strength')
    plt.yscale('log')
    plt.savefig('periodogram_comparison.pdf', dpi=300)
    plt.show()