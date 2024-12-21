import sys
from amuse.units.quantities import ScalarQuantity
from amuse.units import units, constants
from amuse.lab import Particles, Particle
from amuse.ext.orbital_elements import generate_binaries
from amuse.ext.protodisk import ProtoPlanetaryDisk
from amuse.units import nbody_system
import numpy as np

from amuse.ext.orbital_elements import get_orbital_elements_from_binaries
import matplotlib.pyplot as plt


class SystemMaker:

    # TODO: finish documentation

    """
    Class for simulating orbiting stars or binaries with disks around a black hole.

    Attributes
    ------------
    smbh_mass
    com_orbiter_mass
    primary_mass: only if the orbiter is a binary
    secondary_mass: only if the orbiter is a binary
    outer_semimajor_axis
    outer_eccentricity
    inner_semimajor_axis
    mutual_inclination
    inner_eccentricity
    inner_arg_of_periapse
    disk_inner_radius
    disk_outer_radius
    disk_mass
    n_orbiters (int): Number of masses orbiting the smbh
    n_disk (int): Number of sph particles in the hydro disk
    no_disk (bool): If True, no disk will be created, other disk/hydro parameters are ignored.

    Methods
    ------------

    """

    def __init__(self, 
                 smbh_mass:ScalarQuantity, 
                 orbiter_mass:list,
                 outer_semimajor_axis:ScalarQuantity,
                 outer_eccentricity:float,
                 inner_semimajor_axis:ScalarQuantity,
                 mutual_inclination:ScalarQuantity,
                 inner_eccentricity:float,
                 inner_arg_of_periapse:ScalarQuantity,
                 disk_inner_radius:ScalarQuantity,
                 disk_outer_radius:ScalarQuantity,
                 disk_mass:ScalarQuantity,
                 n_disk:int) -> None:
        
        self.smbh_mass = smbh_mass
        self.outer_semimajor_axis = outer_semimajor_axis
        self.outer_eccentricity = outer_eccentricity
        self.inner_semimajor_axis = inner_semimajor_axis
        self.mutual_inclination = mutual_inclination
        self.inner_eccentricity = inner_eccentricity
        self.inner_arg_of_periapse = inner_arg_of_periapse
        self.disk_inner_radius = disk_inner_radius
        self.disk_outer_radius = disk_outer_radius
        self.disk_mass = disk_mass
        self.n_disk = n_disk

        self.n_orbiters = len(orbiter_mass)

        if self.n_orbiters == 2:
            self.primary_mass, self.secondary_mass = orbiter_mass
            self.com_orbiter_mass = self.primary_mass + self.secondary_mass + self.disk_mass
            print('Initializing a binary around a black hole.')
    
        elif self.n_orbiters == 1:
            self.com_orbiter_mass = orbiter_mass[0] + self.disk_mass
            print('Initializing a single star around a black hole.')

        else:
            sys.exit(f'Detected {len(orbiter_mass)} masses to orbit the SMBH. \
                       This code currently only supports 1 or 2 orbiters. Quitting.')
            

    @staticmethod
    def move_particles_to_com(particles, com):
        particles.position += com.position
        particles.velocity += com.velocity


    @staticmethod
    def rotation_matrix(inclination, arg_of_periapse):
        """
        Lord forgive me, for I have used ChatGPT...
        
        Parameters:
            inclination (float): Inclination angle in radians.
            arg_of_periapse (float): Argument of periapsis in radians.
        
        Returns:
            Rotation matrix
        """
        
        # Rotation matrices
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(inclination), -np.sin(inclination)],
            [0, np.sin(inclination), np.cos(inclination)]
        ])
        
        
        R_z = np.array([
            [np.cos(arg_of_periapse), -np.sin(arg_of_periapse), 0],
            [np.sin(arg_of_periapse), np.cos(arg_of_periapse), 0],
            [0, 0, 1]
        ])

        combined = R_x @ R_z
        
        return combined
    

    def rotate_orbit(self, particle, inclination, arg_of_periapse, inverse=False):

        matrix = self.rotation_matrix(inclination.value_in(units.rad), arg_of_periapse.value_in(units.rad))
        if inverse:
            matrix = matrix.T

        particle.position = (matrix @ particle.position.value_in(units.m).T).T | units.m
        particle.velocity = (matrix @ particle.velocity.value_in(units.m / units.s).T).T | (units.m / units.s)


    def _make_smbh_and_orbiter(self, true_anomaly=0|units.rad):
        orbiter, smbh = generate_binaries(self.com_orbiter_mass,
                                          self.smbh_mass,
                                          self.outer_semimajor_axis,
                                          eccentricity=self.outer_eccentricity,
                                          true_anomaly=true_anomaly)
        smbh.name = 'SMBH'
        orbiter.name = 'primary_star'
        return orbiter, smbh
    

    def _make_binary(self, true_anomaly=0|units.rad):
        primary, secondary = generate_binaries(self.primary_mass, 
                                               self.secondary_mass, 
                                               self.inner_semimajor_axis, 
                                               eccentricity=self.inner_eccentricity, 
                                               true_anomaly=true_anomaly,
                                               argument_of_periapsis=self.inner_arg_of_periapse,
                                               inclination=self.mutual_inclination)
        primary.name = 'primary_star'
        secondary.name = 'secondary_star'
        
        ### YOU CAN CHECK THAT THIS UNDOES THE ANGLES OF THE BINARY AS VERIFICATION OF THE ROTATION MATRICES ###

        # rotate_orbit(primary, self.mutual_inclination, self.inner_arg_of_periapse, inverse=True)
        # rotate_orbit(secondary, self.mutual_inclination, self.inner_arg_of_periapse, inverse=True)

        ########################################################################################################
        
        return primary, secondary
    
    def _make_disk(self, R=1|units.AU):
        """R is needed to make Rmin and Rmax dimensionless, Sets the scale of the disk, should not be changed."""
        converter = nbody_system.nbody_to_si(self.com_orbiter_mass, R)  # This converter is only used here, no need to return it

        disk = ProtoPlanetaryDisk(self.n_disk, 
                                  convert_nbody=converter, 
                                  Rmin=self.disk_inner_radius/R, 
                                  Rmax=self.disk_outer_radius/R, 
                                  discfraction=self.disk_mass/self.com_orbiter_mass).result

        disk.name = 'disk'
        disk.move_to_center()

        ## bisection algorithm for finding Rhalf that should work but is not necessary 
        # half_particles = self.n_disk // 2 #always an integer
        # Rhalf = self.disk_inner_radius.value_in(units.AU)
        # inner_particle_keys = disk[disk.position.length().value_in(units.AU) <= Rhalf].key
        # num_inner_particles = len(inner_particle_keys)

        # while num_inner_particles != half_particles:
        #     if num_inner_particles < half_particles:
        #         Rhalf += 0.5 * (self.disk_outer_radius.value_in(units.AU) - Rhalf) #move halfway to outer radius 
        #     else:
        #         Rhalf += 0.5 * (self.disk_inner_radius.value_in(units.AU) - Rhalf) #move halfway to inner radius

        #     inner_particle_keys = disk[disk.position.length().value_in(units.AU) <= Rhalf].key
        #     num_inner_particles = len(inner_particle_keys)

        # disk.inner_particle = np.isin(disk.key, inner_particle_keys)

        return disk


    def make_system(self, true_anomaly=0|units.rad, R=1|units.AU):

        # Might as well return the converter for the whole system here
        converter = nbody_system.nbody_to_si(self.com_orbiter_mass + self.smbh_mass, self.outer_semimajor_axis)

        if self.n_orbiters == 1:

            orbiter, smbh = self._make_smbh_and_orbiter(true_anomaly)

            smbh_and_orbiter = Particles(0)
            smbh_and_orbiter.add_particle(smbh)
            smbh_and_orbiter.add_particle(orbiter)
            smbh_and_orbiter.move_to_center()

            disk = self._make_disk(R)
            self.rotate_orbit(disk, self.mutual_inclination, self.inner_arg_of_periapse)  # Give disk same initial angles as binary
            self.move_particles_to_com(disk, orbiter)  # Disk should be around the binary COM or single star

            return smbh_and_orbiter, disk, converter
            
        elif self.n_orbiters == 2:

            orbiter, smbh = self._make_smbh_and_orbiter(true_anomaly)
            primary, secondary = self._make_binary(true_anomaly)
            self.move_particles_to_com(primary, orbiter)
            self.move_particles_to_com(secondary, orbiter)

            smbh_and_binary = Particles(0)
            smbh_and_binary.add_particle(smbh)
            smbh_and_binary.add_particle(primary)
            smbh_and_binary.add_particle(secondary)
            smbh_and_binary.move_to_center()

            disk = self._make_disk(R)

            # zero = Particle()
            # zero.position = (0,0,0) | units.m
            # zero.velocity = (0,0,0) | (units.m / units.s)
            # zero.mass = self.primary_mass + self.secondary_mass

            # fig, ax = plt.subplots()
            # _, _, _, eccs, _, incs, _, _ = get_orbital_elements_from_binaries(zero, disk, G=constants.G)
            # ax2 = ax.twinx()
            # ax.hist(eccs, density=True, histtype="step", cumulative=True, bins=30, label='CDF', color='blue')
            # ax2.hist(eccs, density=True, histtype="step", bins=30, label='PDF', color='red')
            # ax.set_ylabel('CDF', color='blue')
            # ax2.set_ylabel('PDF', color='red')
            # ax.set_xlabel('e')
            # plt.title('Disk around 0 before rotation')
            # plt.show()

            self.rotate_orbit(disk, self.mutual_inclination, self.inner_arg_of_periapse)  # Give disk same initial angles as binary

            # fig, ax = plt.subplots()
            # _, _, _, eccs, _, incs, _, _ = get_orbital_elements_from_binaries(zero, disk, G=constants.G)
            # ax2 = ax.twinx()
            # ax.hist(eccs, density=True, histtype="step", cumulative=True, bins=30, label='CDF', color='blue')
            # ax2.hist(eccs, density=True, histtype="step", bins=30, label='PDF', color='red')
            # ax.set_ylabel('CDF', color='blue')
            # ax2.set_ylabel('PDF', color='red')
            # ax.set_xlabel('e')
            # plt.title('Disk around 0 after rotation')
            # plt.show()

            self.move_particles_to_com(disk, orbiter)  # Disk should be around the binary COM or single star

            # fig, ax = plt.subplots()
            # _, _, _, eccs, _, incs, _, _ = get_orbital_elements_from_binaries(orbiter, disk, G=constants.G)
            # ax2 = ax.twinx()
            # ax.hist(eccs, density=True, histtype="step", cumulative=True, bins=30, label='CDF', color='blue')
            # ax2.hist(eccs, density=True, histtype="step", bins=30, label='PDF', color='red')
            # ax.set_ylabel('CDF', color='blue')
            # ax2.set_ylabel('PDF', color='red')
            # ax.set_xlabel('e')
            # plt.title('Disk around binary COM after rotation')
            # plt.show()

            return smbh_and_binary, disk, converter

        else:
            sys.exit('If you are seeing this, something broke in initializing this class...')


    def make_system_no_disk(self, true_anomaly=0|units.rad):

        converter = nbody_system.nbody_to_si(self.com_orbiter_mass + self.smbh_mass, self.outer_semimajor_axis)

        if self.n_orbiters == 1:

            orbiter, smbh = self._make_smbh_and_orbiter(true_anomaly)

            smbh_and_orbiter = Particles(0)
            smbh_and_orbiter.add_particle(smbh)
            smbh_and_orbiter.add_particle(orbiter)
            smbh_and_orbiter.move_to_center()

            return smbh_and_orbiter, converter
            
        elif self.n_orbiters == 2:

            orbiter, smbh = self._make_smbh_and_orbiter(true_anomaly)
            primary, secondary = self._make_binary(true_anomaly)
            self.move_particles_to_com(primary, orbiter)
            self.move_particles_to_com(secondary, orbiter)

            smbh_and_binary = Particles(0)
            smbh_and_binary.add_particle(smbh)
            smbh_and_binary.add_particle(primary)
            smbh_and_binary.add_particle(secondary)
            smbh_and_binary.move_to_center()

            return smbh_and_binary, converter
        