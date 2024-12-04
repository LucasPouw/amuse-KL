from amuse.community.fi.interface import Fi
from amuse.couple import bridge
from amuse.community.huayno.interface import Huayno
from amuse.community.hermite.interface import HermiteGRX
from amuse.ext.composition_methods import *
from amuse.units import units
import matplotlib.pyplot as plt
import sys
import os
import cv2
import glob


def moviemaker(image_folder, video_name, fps):

    print(f'Generating video {video_name} from {image_folder} with {fps} fps.')

    unsorted_images = glob.glob(image_folder + '*.png')
    image_numbers = [image_name.split('-')[-1].split('.')[0] for image_name in unsorted_images]
    image_numbers = list(map(int, image_numbers))

    _, sorted_images = zip(*sorted(zip(image_numbers, unsorted_images)))

    frame = cv2.imread(sorted_images[0])
    # height, width, layers = frame.shape  

    # Unsupported height and width gives errors, using (640, 480) for now, but maybe something else is better, TODO: find this out
    video = cv2.VideoWriter(video_name, 0, fps, (640,480))

    for image_name in sorted_images:
        img = cv2.imread(image_name)
        img = cv2.resize(img, (640,480))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


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
                 time_end):
        
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
        

    def _initialize_hydro(self, gamma=5/3, eps=0.1|units.AU):

        hydro = Fi(self.converter, mode="openmp")
        hydro.parameters.use_hydro_flag = True
        hydro.parameters.radiation_flag = False
        hydro.parameters.gamma = gamma
        hydro.parameters.isothermal_flag = True
        hydro.parameters.integrate_entropy_flag = False
        hydro.parameters.timestep = self.hydro_timestep  # 0.01 * binary_period
        hydro.parameters.verbosity = 0
        hydro.parameters.eps_is_h_flag = False  # h_smooth is constant

        # hydro.parameters.gas_epsilon = eps
        # hydro.parameters.sph_h_const = eps

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


    def run_gravity_hydro_bridge(self, movie_kwargs, gamma=1, eps=0.1|units.AU):

        if not os.path.isdir(movie_kwargs['image_folder']):  # kinda ugly
            os.mkdir(movie_kwargs['image_folder'])

        gravity, hydro, gravhydro, channel, bodies = self._initialize_codes(gamma, eps)  # As of yet, "bodies" is not used

        gravity_initial_total_energy = gravity.get_total_energy() + hydro.get_total_energy()
        model_time = 0 | units.Myr
        #this prints the initial condition
        orbiter = self.smbh_and_orbiter[(self.smbh_and_orbiter.mass > 0.5 |units.Msun) & (self.smbh_and_orbiter.mass < 10 |units.Msun)]
        if len(orbiter) == 2:
            pos1,pos2 = orbiter.position.in_(units.AU)
            print(f'INITIAL Binary distance = {abs(pos1 - pos2).length().in_(units.AU)}')
        # else:
            # print(f'I')
        while model_time < self.time_end:

            model_time += self.diagnostic_timestep
            
            dE_gravity = gravity_initial_total_energy / (
                gravity.get_total_energy() + hydro.get_total_energy()
            )
            if len(orbiter) == 2:
                print(f"Time:", model_time.in_(units.yr), "dE=", dE_gravity,end=' ')  # , dE_hydro
            else:
                print(f"Time:", model_time.in_(units.yr), "dE=", dE_gravity)
            gravhydro.evolve_model(model_time)
            channel["to_stars"].copy()
            channel["to_disk"].copy()
            channel["to_binary"].copy()


            # Making plot of orbiter + disk
            orbiter = self.smbh_and_orbiter[(self.smbh_and_orbiter.mass > 0.5 |units.Msun) & (self.smbh_and_orbiter.mass < 10 |units.Msun)]
            if len(orbiter) == 2:
                com = (orbiter[0].position * orbiter[0].mass + orbiter[1].position * orbiter[1].mass) / (orbiter[0].mass + orbiter[1].mass)
                pos1,pos2 = orbiter.position.in_(units.AU)
                print(f'Binary distance = {abs(pos1 - pos2).length().in_(units.AU)} ')
            elif len(orbiter) == 1:
                com = orbiter.copy()
            else:
                sys.exit(f"There are too many bodies orbiting the smbh: {len(orbiter)}")

            fig, ax = plt.subplots(2, 2, figsize=(10,10))
            ax[0,0].scatter(orbiter.x.value_in(units.AU), orbiter.y.value_in(units.AU), zorder=100)
            ax[0,0].scatter(self.disk.x.value_in(units.AU), self.disk.y.value_in(units.AU), s=1)
            # ax[0,0].set_ylim(com.y.value_in(units.AU) - 25, com.y.value_in(units.AU) + 25)
            # ax[0,0].set_xlim(com.x.value_in(units.AU) - 25, com.x.value_in(units.AU) + 25)
            ax[0,0].set_xlabel('x [AU]')
            ax[0,0].set_ylabel('y [AU]')

            ax[1,0].scatter(orbiter.x.value_in(units.AU), orbiter.z.value_in(units.AU), zorder=100)
            ax[1,0].scatter(self.disk.x.value_in(units.AU), self.disk.z.value_in(units.AU), s=1)
            # ax[1,0].set_ylim(com.z.value_in(units.AU) - 1, com.z.value_in(units.AU) + 1)
            # ax[1,0].set_xlim(com.x.value_in(units.AU) - 25, com.x.value_in(units.AU) + 25)
            ax[1,0].set_xlabel('x [AU]')
            ax[1,0].set_ylabel('z [AU]')

            ax[1,1].scatter(orbiter.y.value_in(units.AU), orbiter.z.value_in(units.AU), zorder=100)
            ax[1,1].scatter(self.disk.y.value_in(units.AU), self.disk.z.value_in(units.AU), s=1)
            # ax[1,1].set_ylim(com.z.value_in(units.AU) - 1, com.z.value_in(units.AU) + 1)
            # ax[1,1].set_xlim(com.y.value_in(units.AU) - 25, com.y.value_in(units.AU) + 25)
            ax[1,1].set_xlabel('y [AU]')
            ax[1,1].set_ylabel('z [AU]')
            plt.tight_layout()
            fig.savefig(movie_kwargs['image_folder'] + 'disk-snapshot-' + f'{int(model_time.value_in(units.day))}.png', 
                        bbox_inches='tight',
                        dpi=200)
            plt.close()

        gravity.stop()
        hydro.stop()

        moviemaker(**movie_kwargs)

    