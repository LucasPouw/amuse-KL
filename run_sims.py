from amuse.community.fi.interface import Fi
from amuse.couple import bridge
from amuse.community.huayno.interface import Huayno
# from amuse.community.hermite.interface import HermiteGRX
from amuse.ext.composition_methods import *
from amuse.units import units
from amuse.io import write_set_to_file
import numpy as np


class SimulationRunner():

    # TODO: finish documentation

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
        while model_time < self.time_end:

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
