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


Rmin_stable = 6.55 |units.AU  # minimum stable radius from MA criterion (TODO: change to couple with stable_radii.ipynb)
Rmax_stable = 13.35 |units.AU  # maximum stable radius from Hill radius (TODO: ^^)


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

    smbh_and_binary, disk = ShaiHulud.make_system()
    
    v = smbh_and_binary.velocity
    vdisk = disk.velocity

    plt.figure(figsize=(8,6))
    plt.scatter(smbh_and_binary.x.value_in(units.AU)[1:], 
                smbh_and_binary.y.value_in(units.AU)[1:], 
                c=v.lengths().value_in(units.kms)[1:],
                s=np.log10(smbh_and_binary[1:].mass.number) + 10)
    plt.scatter(smbh_and_binary.x.value_in(units.AU)[0],
                smbh_and_binary.y.value_in(units.AU)[0], 
                c='black', 
                s=np.log10(smbh_and_binary[0].mass.number) + 10)
    plt.scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1, c=vdisk.lengths().value_in(units.kms))
    plt.xlim(-10000, 10000)
    plt.ylim(-10000, 10000)
    plt.colorbar()
    plt.savefig('test1.png')
    plt.show()

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