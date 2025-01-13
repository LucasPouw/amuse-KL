## Made by Lucas Pouw, Yannick Badoux and Tim van der Vuurst for the 
## course "Simulations and Modeling in Astrophysics" '24-'25. 

from amuse.community.fi.interface import Fi
from amuse.couple import bridge
from amuse.community.huayno.interface import Huayno
from amuse.community.hermite.interface import Hermite
from amuse.community.hermite_grx.interface import HermiteGRX
from amuse.ext.composition_methods import *
from amuse.units import units,constants
from amuse.io import write_set_to_file
from amuse.ext.orbital_elements import get_orbital_elements_from_binaries
from amuse.lab import Particle
import numpy as np
from plotter import get_com, get_com_vel


class SimulationRunner():
    """
    Class for running a bridged hydrodynamics + gravity simulation of a SPH disk around a binary (or single) star system which itself
    is in orbit around a SMBH.

    Attributes:
    -------------
    smbh_and_orbiter : amuse.lab.Particles
        Particles representing the supermassive black hole (SMBH) and binary/single orbiter system.
    disk : amuse.lab.Particles
        Particles representing the hydro disk.
    converter : amuse.units.nbody_system.nbody_to_si
        Unit converter for translating between N-body and SI units.
    hydro_timestep : amuse.units.quantity.Quantity
        Timestep for the hydrodynamics code.
    gravhydro_timestep : amuse.units.quantity.Quantity
        Timestep for the bridge coupling the gravity and hydrodynamics codes.
    diagnostic_timestep : amuse.units.quantity.Quantity
        Timestep for diagnostic outputs (e.g., snapshots and energy checks).
    time_end : amuse.units.quantity.Quantity
        Simulation end time.

    Methods:
    ------------
    run_gravity_hydro_bridge(save_folder):
        Runs the simulation with a gravity-hydro bridge.
    run_gravity_hydro_bridge_stopping_condition(save_folder, N_init):
        Runs the simulation with a stopping condition based on the number of bound particles (or end time).
    get_bound_disk_particles(particle_system):
        Analyzes the disk particles to determine which are bound to the system.
    """

    def __init__(self, 
                 smbh_and_orbiter, 
                 disk, 
                 converter,
                 hydro_timestep,
                 gravhydro_timestep,
                 diagnostic_timestep,
                 time_end,
                 gravity_code) -> None:
        """
        Initialize the SimulationRunner with the given parameters.

        Parameters:
        -------------
        smbh_and_orbiter : amuse.lab.Particles
            Particles representing the supermassive black hole (SMBH) and binary/single orbiter system.
        disk : amuse.lab.Particles
            Particles representing the hydro disk.
        converter : amuse.units.nbody_system.nbody_to_si
            Unit converter for translating between N-body and SI units.
        hydro_timestep : amuse.units.quantity.Quantity
            Timestep for the hydrodynamics code.
        gravhydro_timestep : amuse.units.quantity.Quantity
            Timestep for the bridge coupling the gravity and hydrodynamics codes.
        diagnostic_timestep : amuse.units.quantity.Quantity
            Timestep for diagnostic outputs (e.g., snapshots and energy checks).
        time_end : amuse.units.quantity.Quantity
            Simulation end time.
        """
        self.smbh_and_orbiter = smbh_and_orbiter
        self.disk = disk
        self.converter = converter
        self.hydro_timestep = hydro_timestep
        self.gravhydro_timestep = gravhydro_timestep
        self.diagnostic_timestep = diagnostic_timestep
        self.time_end = time_end
        self.gravity_string = gravity_code
        match self.gravity_string:
            case 'Huayno': 
                self.gravity_code = Huayno
            case 'Hermite':
                self.gravity_code = Hermite

            case _:
                raise ValueError("Given gravity code is not allowed, please specify either 'Huayno' or 'Hermite'.")
    
    def _initialize_gravity(self):
        """
        Initialize the specified gravitational N-body code.

        Returns:
        ---------
        Gravity instance with initialized parameters.
        """
        gravity = self.gravity_code(self.converter)
        if self.gravity_string == 'Huayno': #Huayno requires an extra argument
            gravity.set_integrator('OK')

        return gravity
        

    def _initialize_hydro(self):
        """
        Initialize the hydrodynamics code (Fi).

        Returns:
        ---------
        Fi instance with initialized parameters.
        """
        hydro = Fi(self.converter, mode="openmp")
        hydro.parameters.use_hydro_flag = True
        hydro.parameters.radiation_flag = False
        hydro.parameters.timestep = self.hydro_timestep  # 0.01 * binary_period
        return hydro


    def _initialize_bridge(self, gravity, hydro):
        """
        Initialize the gravity-hydro bridge.

        Parameters:
        -------------
        gravity : Huayno or Hermite
            Gravity code.
        hydro : Fi
            Hydrodynamics code.

        Returns:
        ---------
        Bridge instance coupling gravity and hydrodynamics.
        """
        gravhydro = bridge.Bridge(use_threading=False)
        gravhydro.add_system(gravity, (hydro,))
        gravhydro.add_system(hydro, (gravity,))
        gravhydro.timestep = self.gravhydro_timestep  # 0.1 * binary_period
        return gravhydro


    def _initialize_codes(self):
        """
        Initialize the simulation codes and setup the system.

        Returns:
        ---------
        tuple:
            - gravity (Huayno or Hermite): Gravity code.
            - hydro (Fi): Hydrodynamics code.
            - gravhydro (Bridge): Coupled gravity-hydro code.
            - channel (dict): Channels for updating the combined particle set.
            - bodies (amuse.lab.Particles): Combined particle set.
        """        
        bodies = self.smbh_and_orbiter.copy()

        gravity = self._initialize_gravity()

        if self.gravity_string in ['Huayno','Hermite']:
            gravity.particles.add_particles(bodies)

        else: # This should never be called due to the catch in self.__init__, but better safe than sorry
            raise ValueError("Invalid gravity code passed. Please specify either 'Huayno' or 'Hermite'.")

        channel = {"from stars": bodies.new_channel_to(gravity.particles),
                    "to_stars": gravity.particles.new_channel_to(bodies)}
        
        hydro = self._initialize_hydro()
        hydro.particles.add_particles(self.disk)

        bodies.add_particles(self.disk)

        channel.update({"from_disk": bodies.new_channel_to(hydro.particles)})
        channel.update({"to_disk": hydro.particles.new_channel_to(bodies)})

        gravhydro = self._initialize_bridge(gravity, hydro)

        return gravity, hydro, gravhydro, channel, bodies
    

    def run_gravity_hydro_bridge(self, save_folder):
        """
        Run the gravity-hydro simulation until the specified end time without additional stopping conditions.

        Parameters:
        -------------
        save_folder : str
            Directory to save snapshots and diagnostics.

        Returns:
        ---------
        tuple:
            - grav_energy (list of amuse.units.quantity.Quantity): Gravitational energy over time.
            - hydro_energy (list of amuse.units.quantity.Quantity): Hydrodynamical energy over time.
            - times (list of amuse.units.quantity.Quantity): Diagnostic times.
        """
        gravity, hydro, gravhydro, channel, bodies = self._initialize_codes()  

        grav_energy = [] | units.J
        hydro_energy = [] | units.J
        times = [] | units.yr
        
        initial_total_energy = gravity.get_total_energy() + hydro.get_total_energy()
        grav_energy.append(gravity.get_total_energy())
        hydro_energy.append(hydro.get_total_energy())

        model_time = 0 | units.Myr
        times.append(model_time)

        #controls the printing in the terminal
        self.verbose_timestep = 10 * self.diagnostic_timestep

        write_set_to_file(bodies, save_folder + f'/snapshot_0.hdf5')  # Save initial conditions
        while (model_time < self.time_end): # Stopping condition is only pre-specified simulation endtime

            model_time += self.diagnostic_timestep

            gravhydro.evolve_model(model_time)
            channel["to_stars"].copy()
            channel["to_disk"].copy()

            relative_dE = initial_total_energy / (gravity.get_total_energy() + hydro.get_total_energy()) - 1
            grav_energy.append(gravity.get_total_energy())
            hydro_energy.append(hydro.get_total_energy())
            times.append(model_time)
            
            # So as not to clutter up the terminal too much
            if not int(model_time.value_in(units.yr) % self.verbose_timestep.value_in(units.yr)): 
                print(f"Time: {model_time.value_in(units.yr):.2E} yr, Relative energy error dE={relative_dE:.3E}")
            
            write_set_to_file(bodies, save_folder + f'/snapshot_{int(model_time.value_in(units.day))}.hdf5')

        gravity.stop()
        hydro.stop()

        return grav_energy, hydro_energy, times


    def get_bound_disk_particles(self, particle_system):
        """
        Determine the bound disk particles based on their orbital elements.

        Note that the inner/outer classification of unbound particles is made after they have been unbound,
        so the particles have had max. 1 yr travel time (usually outwards) before we classify them. This 
        could give a slight bias to outwards particles.

        Parameters:
        -------------
        particle_system : amuse.lab.Particles
            The system containing the SMBH, star(s) and disk particles.

        Returns:
        ---------
        tuple:
            - N_bound (int): Number of bound particles.
            - N_unbound (int): Number of unbound particles.
            - num_inner_unbound (int): Number of newly unbound inner particles corrected for rebounding.
            - num_outer_unbound (int): Number of newly unbound outer particles corrected for rebounding.
        """
        # Particle_system is always bodies as returned by self._initialize_code,
        # but to avoid confusion it is called particle_system here
        disk = particle_system[particle_system.name == 'disk'].copy()
        stars = particle_system[np.logical_or(particle_system.name == 'primary_star', particle_system.name == 'secondary_star')].copy()

        # Initialize com as a particle with the total stellar mass
        com = Particle()
        com.position = get_com(stars)
        com.velocity = get_com_vel(stars)
        com.mass = 0 | units.Msun
        for star in stars: # Works for both single and double star
            com.mass += star.mass 
        _, _, _, eccs, _, _, _, _ = get_orbital_elements_from_binaries(com, disk, G=constants.G)
        bound = eccs < 1 # Define bound condition

        # Move disk to COM frame and isolate bound particles to define half particle radius
        disk.position -= com.position
        bound_disk_particles = disk[bound]

        N_bound = np.sum(bound)
        Nhalf = N_bound // 2  # Half the total number of bound particles, rounded to integer so it can be used as indexer    

        bound_Rs = np.sort(np.linalg.norm(bound_disk_particles.position.value_in(units.AU), axis=1))
        Rhalf = (bound_Rs[Nhalf - 1] + bound_Rs[Nhalf]) / 2 # Halfway between the largest inner particle radius and smallest outer particle radius
        self.Rhalf_values.append(Rhalf)

        # Save whether a particle is inner as a boolean in the particle set
        disk.is_inner = np.linalg.norm(disk.position.value_in(units.AU), axis=1) < Rhalf
        bound_disk_particles = disk[bound] # Redefine so is_inner attribute exists in the particle set
        unbound_disk_particles = disk[np.invert(bound)]

        # Make a dictionary that saves the inner/outer classification at time of unbounding
        for key in unbound_disk_particles.key: 
            if key not in self.unbound_dict.keys():
                self.unbound_dict[key] = unbound_disk_particles[unbound_disk_particles.key == key].is_inner.astype(bool)

        # Particles may get rebound
        rebounded = bound_disk_particles[np.isin(bound_disk_particles.key, self.already_unbound)].key

        # Remove rebounded particles from already_unbound and compensate for them in the counting
        n_inner_rebound = 0
        n_outer_rebound = 0
        if len(rebounded) > 0:
            self.already_unbound = list(np.array(self.already_unbound)[np.invert(np.isin(self.already_unbound, rebounded))])

            for key in rebounded:  # Avoid double counting by subtracting the #rebounds from the #bounds
                if self.unbound_dict[key] == True:  # i.e. an inner particle got rebound
                    n_inner_rebound += 1
                elif self.unbound_dict[key] == False:  # i.e. an outer particle got rebound
                    n_outer_rebound += 1
                else:
                    print(self.unbound_dict[key], 'is_inner value non-boolean. Something is going wrong, consider aborting code.')

                self.unbound_dict.pop(key)
        
        # Isolate the unbound particles that have not previously gone unbound (based on their key)
        already_unbound_mask = np.invert(np.isin(unbound_disk_particles.key, self.already_unbound))
        new_unbound_disk_particles = unbound_disk_particles[already_unbound_mask]

        self.already_unbound += list(new_unbound_disk_particles.key)

        num_inner_unbound = np.sum(new_unbound_disk_particles.is_inner.astype(bool)) - n_inner_rebound
        num_outer_unbound = np.sum(np.invert(new_unbound_disk_particles.is_inner.astype(bool))) - n_outer_rebound

        return N_bound, len(unbound_disk_particles), num_inner_unbound, num_outer_unbound
    

    def run_gravity_hydro_bridge_stopping_condition(self, save_folder, N_init):
        """
        Runs the gravity-hydro simulation until specified end-time or until an additional stopping condition is reached.
        The additional stopping condition is defined as that the code will stop as soon as at least half the particles of the disk
        become unbounded.

        Parameters:
        -------------
        save_folder : str
            Directory to save snapshots and diagnostics.
        N_init : int
            Initial number of disk particles.

        Returns:
        ---------
        tuple:
            - N_bound_over_time (list): Number of bound particles over time.
            - N_inner (int): Total number of inner unbound particles.
            - N_outer (int): Total number of outer unbound particles.
            - model_time (amuse.units.quantity.Quantity): Final simulation time.
            - grav_energy (list of amuse.units.quantity.Quantity): Gravitational energy over time.
            - hydro_energy (list of amuse.units.quantity.Quantity): Hydrodynamical energy over time.
            - times (list of amuse.units.quantity.Quantity): Diagnostic times.
        """
        gravity, hydro, gravhydro, channel, bodies = self._initialize_codes()  

        grav_energy = [] | units.J
        hydro_energy = [] | units.J
        times = [] | units.yr
        
        initial_total_energy = gravity.get_total_energy() + hydro.get_total_energy()
        grav_energy.append(gravity.get_total_energy())
        hydro_energy.append(hydro.get_total_energy())

        model_time = 0 | units.Myr
        times.append(model_time)

        N_bound = N_init
        N_bound_over_time = []
        N_inner, N_outer = 0, 0
        self.already_unbound = []
        self.Rhalf_values = []
        self.unbound_dict = {}

        write_set_to_file(bodies, save_folder + f'/snapshot_0.hdf5')  # Save initial conditions

        #controls the printing in the terminal, could be a function argument but hardcoded for laziness
        self.verbose_timestep = 10 * self.diagnostic_timestep

        while (model_time < self.time_end) and (N_bound > (N_init // 2)): #add condition that num. of bound particles should not be halved
            model_time += self.diagnostic_timestep
                        
            gravhydro.evolve_model(model_time)
            channel["to_stars"].copy()
            channel["to_disk"].copy()

            relative_dE = initial_total_energy / (gravity.get_total_energy() + hydro.get_total_energy()) - 1
            grav_energy.append(gravity.get_total_energy())
            hydro_energy.append(hydro.get_total_energy())
            times.append(model_time)

            #find the number of bound particles as well as new unbound particles and if they were inner or outer particles
            N_bound, N_unbound, new_n_inwards, new_n_outwards = self.get_bound_disk_particles(bodies)
            N_bound_over_time.append(N_bound)
            N_inner += new_n_inwards
            N_outer += new_n_outwards

            if not int(model_time.value_in(units.yr) % self.verbose_timestep.value_in(units.yr)):
                print(f"Time: {model_time.value_in(units.yr):.2E} yr, Relative energy error dE={relative_dE:.3E}")
                print(f"#Bound: {N_bound}, #unbound {N_unbound}.")
                print(f"Inner particles lost: {N_inner} and outer particles: {N_outer}.")
                print()

            write_set_to_file(bodies, save_folder + f'/snapshot_{int(model_time.value_in(units.day))}.hdf5')

        gravity.stop()
        hydro.stop()

        return N_bound_over_time, N_inner, N_outer, model_time, grav_energy, hydro_energy, times


    def run_gravity_no_disk(self, save_folder):
        """
        Run the gravity-only simulation (no disk) until the specified end time.

        This method evolves only the SMBH and binary orbiter system under gravitational interaction,
        without including the disk particles.

        Parameters:
        -------------
        save_folder : str
            Directory to save snapshots and diagnostics.

        Returns:
        ---------
        tuple:
            - energy (list of amuse.units.quantity.Quantity): Gravitational energy over time.
            - times (list of amuse.units.quantity.Quantity): Diagnostic times.
        """
        bodies = self.smbh_and_orbiter.copy()

        gravity = self._initialize_gravity()

        if self.gravity_string in ['Huayno','Hermite']:
            gravity.particles.add_particles(bodies)

        else: # This should never be called due to the catch in self.__init__, but better safe than sorry
            raise ValueError("Invalid gravity code passed. Please specify either 'Huayno' or 'Hermite'.")
        
        channel = gravity.particles.new_channel_to(bodies)

        energy = [] | units.J
        times = [] | units.yr

        initial_total_energy = gravity.get_total_energy()
        energy.append(initial_total_energy)
        model_time = 0 | units.Myr
        times.append(model_time)

        write_set_to_file(gravity.particles, save_folder + f'/snapshot_0.hdf5')  # Save initial conditions

        # controls the printing in the terminal
        self.verbose_timestep = 100 * self.diagnostic_timestep

        while model_time < self.time_end:

            model_time += self.diagnostic_timestep
            gravity.evolve_model(model_time)
            channel.copy()

            relative_dE = initial_total_energy / gravity.get_total_energy() - 1
            energy.append(gravity.get_total_energy())
            times.append(model_time)

            if not int(model_time.value_in(units.yr) % self.verbose_timestep.value_in(units.yr)):
                print(f"Time: {model_time.value_in(units.yr):.2E} yr. Relative energy error dE={relative_dE:.3E}")

            write_set_to_file(bodies, save_folder + f'/snapshot_{int(model_time.value_in(units.day))}.hdf5')

        gravity.stop()

        return energy, times