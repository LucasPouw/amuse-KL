from make_system import SystemMaker
import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units


SMBH_ORBITAL_RADIUS = 44 * 1e-3 | units.parsec
SMBH_ECCENTRICITY = 0.32

PRIMARY_MASS = 2.8 | units.MSun
SECONDARY_MASS = 0.73 | units.MSun

BINARY_SEPARATION = 1.59 | units.AU
BINARY_INCLINATION = 102.55 | units.deg
BINARY_ECCENTRICITY = 0.45
BINARY_PERIAPSE = 311.75 | units.deg

SMBH_MASS = 4.297e6 | units.MSun

Rmax = 12 | units.AU
Rmin = 8 | units.AU
Mdisk = 1.6e-6 | units.MSun
Ndisk = 1000


if __name__ == '__main__':

    # TODO: add parser
    
    ShaiHulud = SystemMaker(SMBH_MASS,
                            [PRIMARY_MASS, SECONDARY_MASS],
                            SMBH_ORBITAL_RADIUS,
                            SMBH_ECCENTRICITY,
                            BINARY_SEPARATION,
                            BINARY_INCLINATION,
                            BINARY_ECCENTRICITY,
                            BINARY_PERIAPSE,
                            Rmin,
                            Rmax,
                            Mdisk,
                            n_disk=Ndisk)  # Shai Hulud is the Maker
    
    smbh_and_binary, disk, converter = ShaiHulud.make_system()
    
    v = smbh_and_binary.velocity
    vdisk = disk.velocity

    plt.figure(figsize=(8,6))
    plt.scatter(smbh_and_binary.x.value_in(units.AU)[1:], smbh_and_binary.y.value_in(units.AU)[1:], c=v.lengths().value_in(units.kms)[1:])
    plt.scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1, c=vdisk.lengths().value_in(units.kms))
    plt.colorbar()
    plt.savefig('test1.png')
    plt.close()

    plt.figure()
    plt.scatter(smbh_and_binary.x.value_in(units.AU)[1:], smbh_and_binary.z.value_in(units.AU)[1:])
    plt.scatter(disk.x.value_in(units.AU), disk.z.value_in(units.AU), s=1)
    plt.savefig('test2.png')
    plt.close() 

