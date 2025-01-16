import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units
from plotter import * #to change rc params


def hill_radius(mass_low, mass_high, semimajor_axis, eccentricity):
    """
    Calculates the Hill sphere radius for a low-mass object orbiting a high-mass object.

    Args:
    mass_low: Mass of the low mass object
    mass_high: Mass of the high mass object
    semimajor_axis: semi-major axis of the low mass objects orbit
    eccentricity: eccentricity of the orbit
    """
    return semimajor_axis * (1 - eccentricity) * (mass_low / (3 * (mass_low + mass_high)))**(1/3)


def MA_criterion(a_in, e_out, i_mut, m_bin, m_3):
    """
    Calculates the minimum semi major axis necesary for a third object to be in a stable orbit around a binary. 
    From: Mardling & Aarseth (2001).
    
    Args:
    a_in: semi-major axis of the inner (binary) orbit
    e_out: eccentricty of the outer orbit
    i_mut: mutual inclination between the binary and the third object
    m_bin: total mass of the binary
    m_3: mass of the third object orbiting the binary
    """
    return 2.8 * a_in * (1 - e_out)**(-1) * (1 - 0.3 * i_mut.value_in(units.deg) / 180) * ((1 + m_3 / m_bin ) * ((1 + e_out) / np.sqrt(1 - e_out)))**(2/5)

if __name__ == '__main__':
    #orbital parameters of D9 from Peissker et al. (2024)
    SMBH_ORBITAL_RADIUS = 44 * 1e-3 | units.parsec
    SMBH_ECCENTRICITY = 0.32
    SMBH_MASS = 4.297e6 | units.MSun

    PRIMARY_MASS = 2.8 | units.MSun
    SECONDARY_MASS = 0.73 | units.MSun
    DISK_MASS = 1.16e-6 | units.MSun

    BINARY_SEPARATION = 1.59 | units.AU
    BINARY_INCLINATION = 102.55 | units.deg
    BINARY_ECCENTRICITY = 0.45
    BINARY_PERIAPSE = 311.75 | units.deg

    #calculate the stable radii for a range of binary masses
    binary_mass_array = np.linspace(0.01, 10, 100)  # Solar masses

    # Stability of the disk is between an inner radius of MA crit. and outer radius of 1/3 * R_hills
    hill_radii = hill_radius(binary_mass_array, SMBH_MASS.value_in(units.MSun), SMBH_ORBITAL_RADIUS.value_in(units.AU), SMBH_ECCENTRICITY)
    upper_limit = hill_radii / 3
    lower_limit = MA_criterion(BINARY_SEPARATION.number, 0, 0|units.rad, binary_mass_array, DISK_MASS.number)

    #find the inner and outer disk radii for the D9 system
    measured_upper = hill_radius((PRIMARY_MASS + SECONDARY_MASS).value_in(units.Msun), SMBH_MASS.value_in(units.Msun), SMBH_ORBITAL_RADIUS.value_in(units.AU), SMBH_ECCENTRICITY) / 3
    measured_lower = MA_criterion(BINARY_SEPARATION.number, 0, 0|units.rad, (PRIMARY_MASS + SECONDARY_MASS).number, DISK_MASS.number)

    print(f'MA crit. D9: {measured_lower:.2f} AU, 1/3 Hill radius D9: {measured_upper:.2f} AU')

    #plot the inner and outer disk radius curves
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(binary_mass_array, upper_limit, label=r'$R_{\rm H} / 3$', color='blue', linewidth=3)
    ax.plot(binary_mass_array, lower_limit, color='red', label=r'MA criterion', linewidth=3)

    ax.scatter([(PRIMARY_MASS + SECONDARY_MASS).number, (PRIMARY_MASS + SECONDARY_MASS).number], [measured_upper, measured_lower], color='magenta', zorder=5, s=75)
    ax.vlines((PRIMARY_MASS + SECONDARY_MASS).number, 0, np.max(upper_limit), color='black', label='Observed', linewidth=3)

    ax.set_xlabel(r'Binary mass [$M_{\odot}$]')
    ax.set_ylabel('Distance from binary COM [AU]')
    ax.set_xlim(0, np.max(binary_mass_array))
    ax.set_ylim(0, np.max(upper_limit))

    ax.legend()
    ax.grid()
    fig.savefig('./figures/stability_criterion.pdf', bbox_inches='tight')
    plt.show()