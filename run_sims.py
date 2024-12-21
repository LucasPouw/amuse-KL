from amuse.community.fi.interface import Fi
from amuse.couple import bridge
from amuse.community.huayno.interface import Huayno
# from amuse.community.hermite.interface import HermiteGRX
from amuse.ext.composition_methods import *
from amuse.units import units,constants
from amuse.io import write_set_to_file
from amuse.ext.orbital_elements import get_orbital_elements_from_binaries
from amuse.lab import Particle
import numpy as np

from plotter import get_com, get_com_vel

# TODO: finish documentation on all functions


class SimulationRunner():


    """
    Class for running a bridged hydro + gravity code

    Attributes:
    -------------
    smbh_and_orbiter
    disk
    converter
    hydro_timestep
    gravhydro_timestep
    time_end


    Methods
    ------------

    """

    def __init__(self, 
                 smbh_and_orbiter, 
                 disk, 
                 converter,
                 hydro_timestep,
                 gravhydro_timestep,
                 diagnostic_timestep,
                 time_end,
                 no_disk=False) -> None:
        
        self.smbh_and_orbiter = smbh_and_orbiter
        self.disk = disk
        self.converter = converter
        self.hydro_timestep = hydro_timestep
        self.gravhydro_timestep = gravhydro_timestep
        self.diagnostic_timestep = diagnostic_timestep
        self.time_end = time_end

    
    def _initialize_gravity(self):

        gravity = Huayno(self.converter)
        gravity.set_integrator('OK')
        
        return gravity
        

    def _initialize_hydro(self):

        hydro = Fi(self.converter, mode="openmp")
        hydro.parameters.use_hydro_flag = True
        hydro.parameters.radiation_flag = False
        hydro.parameters.timestep = self.hydro_timestep  # 0.01 * binary_period

        return hydro


    def _initialize_bridge(self, gravity, hydro):

        gravhydro = bridge.Bridge(use_threading=False)
        gravhydro.add_system(gravity, (hydro,))
        gravhydro.add_system(hydro, (gravity,))
        gravhydro.timestep = self.gravhydro_timestep  # 0.1 * binary_period

        return gravhydro


    def _initialize_codes(self):
        
        bodies = self.smbh_and_orbiter.copy()

        gravity = self._initialize_gravity()
        gravity.particles.add_particles(bodies)

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

        # Note that bodies is everything (incl disk) and self.smbh_and_orbiter is the smbh + the binary
        gravity, hydro, gravhydro, channel, bodies = self._initialize_codes()  

        initial_total_energy = gravity.get_total_energy() + hydro.get_total_energy()
        model_time = 0 | units.Myr

        write_set_to_file(bodies, save_folder + f'/snapshot_0.hdf5')  # Save initial conditions
        while (model_time < self.time_end):

            model_time += self.diagnostic_timestep
            dE_gravity = initial_total_energy / (
                gravity.get_total_energy() + hydro.get_total_energy()
            )
            print(f"Time:", model_time.in_(units.yr), "dE=", dE_gravity - 1)
            
            gravhydro.evolve_model(model_time)
            channel["to_stars"].copy()
            channel["to_disk"].copy()

            write_set_to_file(bodies, save_folder + f'/snapshot_{int(model_time.value_in(units.day))}.hdf5')

        gravity.stop()
        hydro.stop()

    @staticmethod
    def get_bound_disk_particles(particle_system,already_unbound):
        # particle_system is always bodies as returned by self._initialize_code but to avoid confusion 
        # it is called particle_system here
        disk = particle_system[particle_system.name == 'disk']
        stars = particle_system[np.logical_or(particle_system.name == 'primary_star', particle_system.name == 'secondary_star')]

        #initialize com as a particle with the total stellar mass
        com = Particle()
        com.position = get_com(stars)
        com.velocity = get_com_vel(stars)
        com.mass = 0 | units.Msun
        for star in stars: #works for both single and double star
            com.mass += star.mass 
        _, _, _, eccs, _, _, _, _ = get_orbital_elements_from_binaries(com, disk, G=constants.G)
        bound = eccs < 1

        #particles may get rebound
        #remove those particles from already_bounded
        rebounded = disk[bound][np.isin(disk[bound].key,already_unbound)].key
        already_unbound = np.array(already_unbound)[np.invert(np.isin(already_unbound,rebounded))] 

        #isolate unbound particles and find # inward and # outward escaped particles
        unbound_disk_particles = disk[np.invert(bound)]
        print(f"#UNBOUND ({len(unbound_disk_particles)}) + #BOUND ({np.sum(bound)}) = {len(unbound_disk_particles) + np.sum(bound)}")
        
        #isolate the unbound particles that have not previously gone unbound (based on their key)
        already_unbound_mask = np.invert(np.isin(unbound_disk_particles.key, already_unbound))
        new_unbound_disk_particles = unbound_disk_particles[already_unbound_mask]
        print(f"#new unbounds: {len(new_unbound_disk_particles)}")

        new_unbound_disk_particles.position -= com.position
        new_unbound_disk_particles.velocity -= com.velocity
         # #I *really* don't like that this has to go via list comprehension, but it's the best way I found
        # #since new_unbound_disk_particles.position.length() gives a single f-ing value...
        dot_products = np.array([np.dot(part.velocity/part.velocity.length(),part.position / part.position.length()) for part in new_unbound_disk_particles])

        num_flown_inwards = np.sum(dot_products < 0)
        num_flown_outwards = np.sum(dot_products > 0)


        return np.sum(bound), num_flown_inwards, num_flown_outwards, new_unbound_disk_particles.key

    def run_gravity_hydro_bridge_stopping_condition(self, save_folder,N_init):
        #Rmin and Rmax should be specified here since they matter to the stopping condition
        #N_init is needed since we need to know how many disk particles we start with and 
        #we return the relative bound fractions

        # Note that bodies is everything (incl disk) and self.smbh_and_orbiter is the smbh + the binary
        gravity, hydro, gravhydro, channel, bodies = self._initialize_codes()  
        
        initial_total_energy = gravity.get_total_energy() + hydro.get_total_energy()
        model_time = 0 | units.Myr

        Nbound = N_init
        N_inwards, N_outwards = 0, 0
        unbound_keys = []

        write_set_to_file(bodies, save_folder + f'/snapshot_0.hdf5')  # Save initial conditions

        #controls the printing in the terminal, could be a function argument but hardcoded for laziness
        verbose_timestep = 10 * self.diagnostic_timestep

        while (model_time < self.time_end) and (Nbound > (N_init // 2)): #add condition that num. of bound particles should not be halved
            model_time += self.diagnostic_timestep
            dE_gravity = initial_total_energy / (
                gravity.get_total_energy() + hydro.get_total_energy() )
                        
            gravhydro.evolve_model(model_time)
            channel["to_stars"].copy()
            channel["to_disk"].copy()

            Nbound, new_n_inwards, new_n_outwards, new_unbound_keys = self.get_bound_disk_particles(bodies,unbound_keys) 
            N_inwards += new_n_inwards
            N_outwards += new_n_outwards
            unbound_keys += list(new_unbound_keys)

            if model_time % verbose_timestep == 0:
                print(f"Time:", model_time.in_(units.yr), "dE=", dE_gravity - 1)
                print(f"Number of bound particles: {Nbound}. Number lost inwards: {N_inwards} and outwards: {N_outwards}. Sums to {Nbound + N_inwards + N_outwards}")
                print()

            write_set_to_file(bodies, save_folder + f'/snapshot_{int(model_time.value_in(units.day))}.hdf5')

        gravity.stop()
        hydro.stop()

        return Nbound/N_init, N_inwards/N_init, N_outwards/N_init, model_time


#BELOW is the code that just checks the number of bound particles and only checks whether it flew inward or outward
#at the end, after the stopping condition has been reached. This is probably wrong. 
##############################################################################################################
    # @staticmethod
    # def get_bound_disk_particles(particle_system,already_unbound=[]):
    #     # particle_system is always bodies as returned by self._initialize_code but to avoid confusion 
    #     # it is called particle_system here
    #     disk = particle_system[particle_system.name == 'disk']
    #     stars = particle_system[np.logical_or(particle_system.name == 'primary_star', particle_system.name == 'secondary_star')]

    #     #initialize com as a particle with the total stellar mass
    #     com = Particle()
    #     com.position = get_com(stars)
    #     com.velocity = get_com_vel(stars)
    #     com.mass = 0 | units.Msun
    #     for star in stars: #works for both single and double star
    #         com.mass += star.mass 
    #     _, _, _, eccs, _, _, _, _ = get_orbital_elements_from_binaries(com, disk, G=constants.G)
    #     bound = eccs < 1

    #     unbound_disk_particles = disk[np.invert(bound)]
    #     unbound_disk_particles.position -= com.position
    #     unbound_disk_particles.velocity -= com.velocity
    #     # #I *really* don't like that this has to go via list comprehension, but it's the best way I found
    #     # #since new_unbound_disk_particles.position.length() gives a single f-ing value...
    #     dot_products = np.array([np.dot(part.velocity/part.velocity.length(),part.position / part.position.length()) for part in unbound_disk_particles])

    #     num_flown_inwards = np.sum(dot_products < 0)
    #     num_flown_outwards = np.sum(dot_products > 0)

    #     print(f"At this timestep: Num inward = {num_flown_inwards} and num outward = {num_flown_outwards}")

    #     return bound

    # @staticmethod
    # def get_inward_outward_particles(particle_system,bound):
    #     # particle_system is always bodies as returned by self._initialize_code but to avoid confusion 
    #     # it is called particle_system here
    #     disk = particle_system[particle_system.name == 'disk']
    #     stars = particle_system[np.logical_or(particle_system.name == 'primary_star', particle_system.name == 'secondary_star')]

    #     #initialize com as a particle with the total stellar mass
    #     com = Particle()
    #     com.position = get_com(stars)
    #     com.velocity = get_com_vel(stars)
    #     com.mass = 0 | units.Msun
    #     for star in stars: #works for both single and double star
    #         com.mass += star.mass 

    #     # #particles may get rebound
    #     # #remove those particles from already_bounded
    #     # rebounded = disk[bound][np.isin(disk[bound].key,already_unbound)].key
    #     # already_unbound = np.array(already_unbound)[np.invert(np.isin(already_unbound,rebounded))] 

    #     #isolate unbound particles and find # inward and # outward escaped particles
    #     unbound_disk_particles = disk[np.invert(bound)]
    #     print(f"#UNBOUND ({len(unbound_disk_particles)}) + #BOUND ({np.sum(bound)}) = {len(unbound_disk_particles) + np.sum(bound)}")

    #     # #isolate the unbound particles that have not previously gone unbound (based on their key)
    #     # already_unbound_mask = np.invert(np.isin(unbound_disk_particles.key, already_unbound))
    #     # new_unbound_disk_particles = unbound_disk_particles[already_unbound_mask]
    #     # print(f"#new unbounds: {len(new_unbound_disk_particles)}")

    #     # new_unbound_disk_particles.position -= com.position
    #     # new_unbound_disk_particles.velocity -= com.velocity
    #     #  # #I *really* don't like that this has to go via list comprehension, but it's the best way I found
    #     # # #since new_unbound_disk_particles.position.length() gives a single f-ing value...
    #     # dot_products = np.array([np.dot(part.velocity/part.velocity.length(),part.position / part.position.length()) for part in new_unbound_disk_particles])

    #     unbound_disk_particles.position -= com.position
    #     unbound_disk_particles.velocity -= com.velocity
    #     # #I *really* don't like that this has to go via list comprehension, but it's the best way I found
    #     # #since new_unbound_disk_particles.position.length() gives a single f-ing value...
    #     dot_products = np.array([np.dot(part.velocity/part.velocity.length(),part.position / part.position.length()) for part in unbound_disk_particles])

    #     num_flown_inwards = np.sum(dot_products < 0)
    #     num_flown_outwards = np.sum(dot_products > 0)

    #     # return np.sum(bound), num_flown_inwards, num_flown_outwards, new_unbound_disk_particles.key
    #     return num_flown_inwards, num_flown_outwards

    # def run_gravity_hydro_bridge_stopping_condition(self, save_folder,N_init):
    #     #Rmin and Rmax should be specified here since they matter to the stopping condition
    #     #N_init is needed since we need to know how many disk particles we start with and 
    #     #we return the relative bound fractions

    #     # Note that bodies is everything (incl disk) and self.smbh_and_orbiter is the smbh + the binary
    #     gravity, hydro, gravhydro, channel, bodies = self._initialize_codes()  

    #     initial_total_energy = gravity.get_total_energy() + hydro.get_total_energy()
    #     model_time = 0 | units.Myr

    #     Nbound = N_init
    #     N_inwards, N_outwards = 0, 0
    #     unbound_keys = []

    #     write_set_to_file(bodies, save_folder + f'/snapshot_0.hdf5')  # Save initial conditions

    #     while (model_time < self.time_end) and (Nbound > 850): #TODO: change back to Ninit // 2 #add condition that num. of bound particles should not be halved
    #         model_time += self.diagnostic_timestep
    #         dE_gravity = initial_total_energy / (
    #             gravity.get_total_energy() + hydro.get_total_energy() )
        
    #         print(f"Time:", model_time.in_(units.yr), "dE=", dE_gravity - 1)
        
    #         gravhydro.evolve_model(model_time)
    #         channel["to_stars"].copy()
    #         channel["to_disk"].copy()

    #         bound = self.get_bound_disk_particles(bodies)
    #         Nbound = np.sum(bound)
    #         print(f'Number of bound particles: {Nbound}')

    #         # Nbound, new_n_inwards, new_n_outwards, new_unbound_keys = self.get_bound_disk_particles(bodies,unbound_keys) 
    #         # N_inwards += new_n_inwards
    #         # N_outwards += new_n_outwards
    #         # unbound_keys += list(new_unbound_keys)
    #         # print(f"Number of bound particles: {Nbound}. Number lost inwards: {N_inwards} and outwards: {N_outwards}. Sums to {Nbound + N_inwards + N_outwards}")
    #         # print()

    #         write_set_to_file(bodies, save_folder + f'/snapshot_{int(model_time.value_in(units.day))}.hdf5')

    #     gravity.stop()
    #     hydro.stop()

    #     N_inwards,N_outwards = self.get_inward_outward_particles(bodies,bound)

    #     print(f"Number of bound particles: {Nbound}. Number lost inwards: {N_inwards} and outwards: {N_outwards}. Sums to {Nbound + N_inwards + N_outwards}")

    #     return Nbound/N_init, N_inwards/N_init, N_outwards/N_init, model_time
##############################################################################################################



    def run_gravity_no_disk(self, save_folder):

        bodies = self.smbh_and_orbiter.copy()

        gravity = self._initialize_gravity()
        gravity.particles.add_particles(bodies)
        channel = gravity.particles.new_channel_to(bodies)

        model_time = 0 | units.Myr
        initial_total_energy = gravity.get_total_energy()
        write_set_to_file(gravity.particles, save_folder + f'/snapshot_0.hdf5')  # Save initial conditions
        while model_time < self.time_end:

            model_time += self.diagnostic_timestep
            dE_gravity = initial_total_energy / gravity.get_total_energy()
            print(f"Time:", model_time.in_(units.yr), "dE=", dE_gravity - 1)

            gravity.evolve_model(model_time)
            channel.copy()

            write_set_to_file(bodies, save_folder + f'/snapshot_{int(model_time.value_in(units.day))}.hdf5')

        gravity.stop()
