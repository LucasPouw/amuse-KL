import sys
from amuse.units.quantities import ScalarQuantity
from amuse.units import units
from amuse.lab import Particles
from amuse.ext.orbital_elements import generate_binaries
from amuse.ext.protodisk import ProtoPlanetaryDisk
from amuse.units import nbody_system


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
            

    def _make_smbh_and_orbiter(self, true_anomaly=0|units.rad):
        orbiter, smbh = generate_binaries(self.com_orbiter_mass,
                                          self.smbh_mass,
                                          self.outer_semimajor_axis,
                                          eccentricity=self.outer_eccentricity,
                                          true_anomaly=true_anomaly,
                                          inclination=self.mutual_inclination)
        smbh.name = 'SMBH'
        orbiter.name = 'primary_star'
        return orbiter, smbh
    

    def _make_binary_at_orbiter(self, orbiter, true_anomaly=0|units.rad, inclination=0|units.rad):
        primary, secondary = generate_binaries(self.primary_mass, 
                                               self.secondary_mass, 
                                               self.inner_semimajor_axis, 
                                               eccentricity=self.inner_eccentricity, 
                                               true_anomaly=true_anomaly,
                                               inclination=inclination,
                                               argument_of_periapsis=self.inner_arg_of_periapse)
        primary.position += orbiter.position
        secondary.position += orbiter.position
        primary.velocity += orbiter.velocity
        secondary.velocity += orbiter.velocity
        primary.name = 'primary_star'
        secondary.name = 'secondary_star'
        return primary, secondary
    

    def _make_disk_at_orbiter(self, orbiter, R=1|units.AU):
        """R is needed to make Rmin and Rmax dimensionless, Sets the scale of the disk, should not be changed."""
        converter = nbody_system.nbody_to_si(self.com_orbiter_mass, R)  # This converter is only used here, no need to return it

        disk = ProtoPlanetaryDisk(self.n_disk, 
                                  convert_nbody=converter, 
                                  Rmin=self.disk_inner_radius/R, 
                                  Rmax=self.disk_outer_radius/R, 
                                  discfraction=self.disk_mass/self.com_orbiter_mass).result
        
        # Move sph particles to correct position
        disk.position += orbiter.position
        disk.velocity += orbiter.velocity
        disk.name = 'disk'
        return disk


    def make_system(self, true_anomaly=0|units.rad, inclination=0|units.rad, R=1|units.AU):

        # Might as well return the converter for the whole system here
        converter = nbody_system.nbody_to_si(self.com_orbiter_mass + self.smbh_mass, self.outer_semimajor_axis)

        if self.n_orbiters == 1:

            orbiter, smbh = self._make_smbh_and_orbiter(true_anomaly)

            smbh_and_orbiter = Particles(0)
            smbh_and_orbiter.add_particle(smbh)
            smbh_and_orbiter.add_particle(orbiter)
            smbh_and_orbiter.move_to_center()
            disk = self._make_disk_at_orbiter(orbiter, R)

            return smbh_and_orbiter, disk, converter
            
        elif self.n_orbiters == 2:

            orbiter, smbh = self._make_smbh_and_orbiter(true_anomaly)
            primary, secondary = self._make_binary_at_orbiter(orbiter, true_anomaly, inclination)

            smbh_and_binary = Particles(0)
            smbh_and_binary.add_particle(smbh)
            smbh_and_binary.add_particle(primary)
            smbh_and_binary.add_particle(secondary)
            smbh_and_binary.move_to_center()
            disk = self._make_disk_at_orbiter(orbiter, R)

            return smbh_and_binary, disk, converter

        else:
            sys.exit('If you are seeing this, something broke in initializing this class...')
            

    def make_system_no_disk(self, true_anomaly=0|units.rad, inclination=0|units.rad):

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
            primary, secondary = self._make_binary_at_orbiter(orbiter, true_anomaly, inclination)

            smbh_and_binary = Particles(0)
            smbh_and_binary.add_particle(smbh)
            smbh_and_binary.add_particle(primary)
            smbh_and_binary.add_particle(secondary)
            smbh_and_binary.move_to_center()

            return smbh_and_binary, converter