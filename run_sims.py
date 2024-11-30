from amuse.community.fi.interface import Fi
from amuse.couple import bridge
from amuse.community.huayno.interface import Huayno
from amuse.ext.composition_methods import *
from amuse.units import units


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
                 time_end):
        
        self.smbh_and_orbiter = smbh_and_orbiter
        self.disk = disk
        self.converter = converter
        self.hydro_timestep = hydro_timestep
        self.gravhydro_timestep = gravhydro_timestep
        self.time_end = time_end

    
    def _initialize_gravity(self):

        gravity = Huayno(self.converter)
        gravity.set_integrator('OK')
        
        return gravity
        

    def _initialize_hydro(self, gamma=1, eps=0.1|units.AU):

        hydro = Fi(self.converter, mode="openmp")
        hydro.parameters.use_hydro_flag = True
        hydro.parameters.radiation_flag = False
        hydro.parameters.gamma = gamma
        hydro.parameters.isothermal_flag = True
        hydro.parameters.integrate_entropy_flag = False
        hydro.parameters.timestep = self.hydro_timestep  # 0.01 * binary_period
        hydro.parameters.verbosity = 0
        hydro.parameters.eps_is_h_flag = False  # h_smooth is constant

        hydro.parameters.gas_epsilon = eps
        hydro.parameters.sph_h_const = eps

        return hydro


    def _initialize_bridge(self, gravity, hydro):

        gravhydro = bridge.Bridge(use_threading=False)  # , method=SPLIT_4TH_S_M4)
        gravhydro.add_system(gravity, (hydro,))
        gravhydro.add_system(hydro, (gravity,))
        gravhydro.timestep = self.gravhydro_timestep  # 0.1 * binary_period

        return gravhydro


    def _initialize_codes(self, gamma=1, eps=0.1|units.AU):
        
        bodies = self.smbh_and_orbiter
        orbiter = self.smbh_and_orbiter[1:]  # Can be either 1 or 2 stars

        gravity = self._initialize_gravity()
        gravity.particles.add_particles(bodies - orbiter)  # Because for some reason the example does this

        channel = {"from stars": bodies.new_channel_to(gravity.particles),
                    "to_stars": gravity.particles.new_channel_to(bodies)}
        
        hydro = self._initialize_hydro(gamma, eps)
        hydro.particles.add_particles(self.disk)
        hydro.dm_particles.add_particles(orbiter)

        channel.update({"from_disk": self.disk.new_channel_to(hydro.particles)})
        channel.update({"to_disk": hydro.particles.new_channel_to(self.disk)})
        channel.update({"from_binary": orbiter.new_channel_to(hydro.dm_particles)})
        channel.update({"to_binary": hydro.dm_particles.new_channel_to(orbiter)})

        bodies.add_particles(self.disk)

        gravhydro = self._initialize_bridge(gravity, hydro)

        return gravity, hydro, gravhydro, channel, bodies


    def run_gravity_hydro_bridge(self, gamma=1, eps=0.1|units.AU):

        gravity, hydro, gravhydro, channel, bodies = self._initialize_codes(gamma, eps)  # As of yet, "bodies" is not used

        gravity_initial_total_energy = gravity.get_total_energy() + hydro.get_total_energy()
        model_time = 0 | units.Myr
        dt = 1 | units.yr  # 1.0*Pinner
        while model_time < self.time_end:

            model_time += dt
            
            dE_gravity = gravity_initial_total_energy / (
                gravity.get_total_energy() + hydro.get_total_energy()
            )
            print("Time:", model_time.in_(units.yr), "dE=", dE_gravity)  # , dE_hydro

            gravhydro.evolve_model(model_time)
            channel["to_stars"].copy()
            channel["to_disk"].copy()
            channel["to_binary"].copy()

        gravity.stop()
        hydro.stop()

    