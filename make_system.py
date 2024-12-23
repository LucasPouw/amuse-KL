import sys
from amuse.units.quantities import ScalarQuantity
from amuse.units import units
from amuse.lab import Particles
from amuse.ext.orbital_elements import generate_binaries
from amuse.ext.protodisk import ProtoPlanetaryDisk
from amuse.units import nbody_system
import numpy as np


class SystemMaker:
    """
    Class for creating an amuse.lab.Particles instance representing a single or 
    binary with a hydro disk that is orbiting a supermassive black hole (SMBH).

    Attributes
    -----------
    smbh_mass : ScalarQuantity
        Mass of the supermassive black hole (SMBH).
    com_orbiter_mass : ScalarQuantity
        Total mass of the center of mass of the orbiter(s) (= the total mass of the star(s)).
    primary_mass : ScalarQuantity
        Mass of the primary component (only defined if orbiter is a binary).
    secondary_mass : ScalarQuantity
        Mass of the secondary component (only defined if orbiter is a binary).
    outer_semimajor_axis : ScalarQuantity
        Semi-major axis of the outer orbit (binary/single around the SMBH).
    outer_eccentricity : float
        Eccentricity of the outer orbit.
    inner_semimajor_axis : ScalarQuantity
        Semi-major axis of the inner orbit (only used if orbiter is a binary).
    mutual_inclination : ScalarQuantity
        Mutual inclination angle between the inner and outer orbits.
    inner_eccentricity : float
        Eccentricity of the inner orbit (only used if orbiter is a binary).
    inner_arg_of_periapse : ScalarQuantity
        Argument of periapse for the inner orbit (only used if orbiter is a binary).
    disk_inner_radius : ScalarQuantity
        Inner radius of the circumbinary disk.
    disk_outer_radius : ScalarQuantity
        Outer radius of the circumbinary disk.
    disk_mass : ScalarQuantity
        Total mass of the circumbinary disk.
    n_orbiters : int
        Number of orbiting bodies (1 for single star, 2 for binary).
    n_disk : int
        Number of SPH particles in the circumbinary disk.

    Methods
    ----------
    rotate_orbit():
        Applies a rotation matrix to a particle's position and velocity.
    make_system():
        Creates a complete system including SMBH, orbiters, and disk.
    make_system_no_disk():
        Creates a system (binary/single around SMBH) without a disk.
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
        """
        Initializes the SystemMaker object.

        Parameters
        ----------
        smbh_mass: ScalarQuantity
            Mass of the SMBH.
        orbiter_mass: list
            List of masses for the orbiters (single star or binary).
        outer_semimajor_axis: ScalarQuantity
            Semi-major axis of the outer orbit.
        outer_eccentricity: float
            Eccentricity of the outer orbit.
        inner_semimajor_axis: ScalarQuantity
            Semi-major axis of the binary orbit (if applicable).
        mutual_inclination: ScalarQuantity
            Inclination of the binary relative to the outer orbit.
        inner_eccentricity: float
            Eccentricity of the binary orbit.
        inner_arg_of_periapse: ScalarQuantity
            Argument of periapse for the binary orbit.
        disk_inner_radius: ScalarQuantity
            Inner radius of the circumbinary disk.
        disk_outer_radius: ScalarQuantity
            Outer radius of the circumbinary disk.
        disk_mass: ScalarQuantity
            Total mass of the circumbinary disk.
        n_disk: int
            Number of SPH particles in the circumbinary disk.
        """
        
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
        Constructs a rotation matrix for applying an orbital inclination and argument of periapse.

        Parameters
        ----------
        inclination: float
            Inclination angle in radians.
        arg_of_periapse: float
            Argument of periapse in radians.

        Returns
        -------
        np.ndarray
            Combined rotation matrix.
        """
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
        return R_x @ R_z
    

    def rotate_orbit(self, particle, inclination, arg_of_periapse, inverse=False):
        """
        Rotates a particle's position and velocity using a rotation matrix which encodes
        an inclination angle and argument of periapse.

        Parameters
        ----------
        particle: Particle
            Particle to rotate.
        inclination: ScalarQuantity
            Inclination angle.
        arg_of_periapse: ScalarQuantity
            Argument of periapse.
        inverse: bool, optional
            Whether to apply the inverse rotation. Default is False.
        """
        matrix = self.rotation_matrix(inclination.value_in(units.rad), arg_of_periapse.value_in(units.rad))
        if inverse:
            matrix = matrix.T  # Rotation matrices are orthogonal -> inverse = transpose

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
        return disk


    def make_system(self, true_anomaly=0|units.rad, R=1|units.AU):
        """
        Creates the entire system: SMBH, orbiters (single or binary), and circumbinary disk.

        Parameters
        ----------
        true_anomaly: ScalarQuantity
            True anomaly of the outer orbit (just in case you'd ever want to change it).
        R: ScalarQuantity
            Scale radius of the disk.

        Returns
        -------
        system: Particles
            The complete system.
        """
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
            self.rotate_orbit(disk, self.mutual_inclination, self.inner_arg_of_periapse)  # Give disk same initial angles as binary
            self.move_particles_to_com(disk, orbiter)  # Disk should be around the binary COM or single star

            return smbh_and_binary, disk, converter

        else:
            sys.exit('self.n_orbiters is not 1 or 2, but that should have quit the code before...how did you do that?')


    def make_system_no_disk(self, true_anomaly=0|units.rad):
        """
        Creates the system without a circumbinary disk.

        Parameters
        ----------
        true_anomaly: ScalarQuantity
            True anomaly of the outer orbit (just in case you'd ever want to change it).
        inner_true_anomaly: ScalarQuantity
            True anomaly of the binary orbit (if applicable).

        Returns
        -------
        system: Particles
            The system without a disk.
        """
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
        