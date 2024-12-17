from amuse.community.fi.interface import Fi
from amuse.couple import bridge
from amuse.community.huayno.interface import Huayno
# from amuse.community.hermite.interface import HermiteGRX
from amuse.ext.composition_methods import *
from amuse.units import units
import matplotlib.pyplot as plt
import sys
import os
import cv2
import glob
import numpy as np

from matplotlib.patches import ConnectionPatch



def moviemaker(image_folder, video_name, fps):

    print(f'Generating video {video_name} from {image_folder} with {fps} fps.')

    unsorted_images = glob.glob(image_folder + '*.png')
    image_numbers = [image_name.split('-')[-1].split('.')[0] for image_name in unsorted_images]
    image_numbers = list(map(int, image_numbers))

    _, sorted_images = zip(*sorted(zip(image_numbers, unsorted_images)))

    frame = cv2.imread(sorted_images[0])
    # height, width, layers = frame.shape  

    video = cv2.VideoWriter(video_name, 0, fps, (1440,1080))

    for image_name in sorted_images:
        img = cv2.imread(image_name)
        img = cv2.resize(img, (1440,1080))
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
        

    def _initialize_hydro(self):

        hydro = Fi(self.converter, mode="openmp")
        hydro.parameters.use_hydro_flag = True
        hydro.parameters.radiation_flag = False
        hydro.parameters.timestep = self.hydro_timestep  # 0.01 * binary_period

        return hydro


    def _initialize_bridge(self, gravity, hydro):

        gravhydro = bridge.Bridge(use_threading=False)  # , method=SPLIT_4TH_S_M4)
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

        channel.update({"from_disk": self.disk.new_channel_to(hydro.particles)})
        channel.update({"to_disk": hydro.particles.new_channel_to(self.disk)})

        bodies.add_particles(self.disk)

        gravhydro = self._initialize_bridge(gravity, hydro)

        return gravity, hydro, gravhydro, channel, bodies

    @staticmethod
    def get_com(orbiter):
        if len(orbiter) == 2:
            com = (orbiter[0].position * orbiter[0].mass + orbiter[1].position * orbiter[1].mass) / (orbiter[0].mass + orbiter[1].mass)
        else: #orbiter
            com = orbiter.copy().position
        return com

    def run_gravity_hydro_bridge(self, movie_kwargs):

        if not os.path.isdir(movie_kwargs['image_folder']):  # kinda ugly
            os.mkdir(movie_kwargs['image_folder'])

        # Note that bodies is everything (incl disk) and self.smbh_and_orbiter is the smbh + the binary
        gravity, hydro, gravhydro, channel, bodies = self._initialize_codes()  

        initial_total_energy = gravity.get_total_energy() + hydro.get_total_energy()
        model_time = 0 | units.Myr

        #extract orbiter which is the binary (or the single star)
        orbiter = bodies[np.logical_or((bodies.name == 'primary_star'),(bodies.name == 'secondary_star'))]
        smbh = bodies[(bodies.name == 'SMBH')] #extract BH
       
        #this prints the initial binary distance
        if len(orbiter) == 2:
            pos1, pos2 = orbiter.position.in_(units.AU)
            print(f'INITIAL Binary distance = {abs(pos1 - pos2).length().in_(units.AU)}')

        #Run simulation to end
        while model_time < self.time_end:

            model_time += self.diagnostic_timestep
            
            dE_gravity = initial_total_energy / (
                gravity.get_total_energy() + hydro.get_total_energy()
            )

            #Print some diagnostics
            if len(orbiter) == 2:
                print(f"Time:", model_time.in_(units.yr), "dE=", dE_gravity - 1, end=' ')  # , dE_hydro
                print(f'\nGRAVITY TIMESTEP: {gravity.get_timestep().value_in(units.s)} s,\
                        GRAVITY TIMESTEP PARAMETER: {gravity.get_timestep_parameter()}\n')
            else:
                print(f"Time:", model_time.in_(units.yr), "dE=", dE_gravity - 1)
            
            gravhydro.evolve_model(model_time)
            channel["to_stars"].copy()
            channel["to_disk"].copy()

            # Making plot of orbiter + disk
            #extract orbiter which is the binary (or the single star)
            orbiter = bodies[np.logical_or((bodies.name == 'primary_star'),(bodies.name == 'secondary_star'))]
            smbh = bodies[(bodies.name == 'SMBH')] #extract BH
            com = self.get_com(orbiter) #is a position vector

            arrow_to_smbh = smbh.position - com
            arrow_to_smbh = (arrow_to_smbh / arrow_to_smbh.length())[0] * 0.1 #normalize arrow to length 1
 
            if len(orbiter) == 2:
                pos1,pos2 = orbiter.position.in_(units.AU)
                print(f'Binary distance = {abs(pos1 - pos2).length().in_(units.AU)} ')
            elif len(orbiter) > 2:
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
            # ax[1,0].set_ylim(com.z.value_in(units.AU) - 0.1, com.z.value_in(units.AU) + 0.1)
            # ax[1,0].set_xlim(com.x.value_in(units.AU) - 25, com.x.value_in(units.AU) + 25)
            ax[1,0].set_xlabel('x [AU]')
            ax[1,0].set_ylabel('z [AU]')
            ax[1,0].ticklabel_format(useOffset=False)

            ax[1,1].scatter(orbiter.y.value_in(units.AU), orbiter.z.value_in(units.AU), zorder=100)
            ax[1,1].scatter(self.disk.y.value_in(units.AU), self.disk.z.value_in(units.AU), s=1)
            # ax[1,1].set_ylim(com.z.value_in(units.AU) - 0.1, com.z.value_in(units.AU) + 0.1)
            # ax[1,1].set_xlim(com.y.value_in(units.AU) - 25, com.y.value_in(units.AU) + 25)
            ax[1,1].set_xlabel('y [AU]')
            ax[1,1].set_ylabel('z [AU]')
            ax[1,1].ticklabel_format(useOffset=False)

            slice_dict = {0:[0,1],2:[0,2],3:[1,2]} #able to extract xy, xz and yz plane
            for i,axis in enumerate(ax.flatten()):
                if i == 1:
                    continue

                # com_plot = data_to_axis(com.value_in(units.AU)[slice_dict[i]]) #get the relevant com coordinates
                com_plot = (0.5,0.5)
                arrowhead = arrow_to_smbh[slice_dict[i]] #* (0.1 * yrange)

                axis.annotate("", xy=arrowhead, xytext=com_plot,
                            arrowprops=dict(arrowstyle="->"),xycoords='axes fraction',textcoords='axes fraction',
                            )
                
                
                data_to_axis = axis.transLimits.transform
                axis_to_data = axis.transLimits.inverted().transform

                print(axis_to_data(com_plot))
                print(axis_to_data((0.5,0.5)))
                axis.scatter(*axis_to_data(com_plot),marker='x',zorder=200)
                axis.scatter(*axis_to_data(arrowhead),marker='^',zorder=200)
                


            plt.tight_layout()
            fig.savefig(movie_kwargs['image_folder'] + 'disk-snapshot-' + f'{int(model_time.value_in(units.day))}.png', 
                        bbox_inches='tight',
                        dpi=200)
            
            plt.close()

            return    


        gravity.stop()
        hydro.stop()

        moviemaker(**movie_kwargs)

    