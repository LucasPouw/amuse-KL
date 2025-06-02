import numpy as np
import matplotlib.pyplot as plt
import glob
from amuse.io import read_set_from_file
from amuse.units import units
from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import Particles, Particle
import time

def bound(primary, disk_particle):
    if orbital_elements(primary, disk_particle)[3] > 1.0:
        return False
    else:
        return True

if __name__ == "__main__":
    run_paths = ['/home/s2562898/data1/AMUSE_CB_disk/slurm/amuseKL-output/t_end-100000.0-SLURM_time_limit-12.0',
                 '/home/s2562898/data1/AMUSE_CB_disk/slurm/amuseKL-output/r_min-7.26-r_max-12.03-t_end-100000.0-SLURM_time_limit-12.0',
                 '/home/s2562898/data1/AMUSE_CB_disk/slurm/amuseKL-output/r_min-7.59-r_max-10.99-t_end-100000.0-SLURM_time_limit-12.0',
                 '/home/s2562898/data1/AMUSE_CB_disk/slurm/amuseKL-output/r_min-8.03-r_max-11.11-t_end-100000.0-SLURM_time_limit-12.0']
    
    save_path = '/home/s2562898/data1/AMUSE_CB_disk'

    all_times = []
    all_nbounds = []
    all_r_min = []
    all_r_max = []

    for run_path in run_paths:
        start_time = time.time()
        print(f'Processing {run_path}')
        if 'r_min' in run_path:
            r_min = run_path.split('r_min-')[1].split('-')[0]
            r_max = run_path.split('r_max-')[1].split('-')[0]
        else:
            r_min = '4.45'
            r_max = '13.35'
        print(r_min, r_max)

        snapshot_files = glob.glob(run_path + f'/snapshots-rmin{r_min}-rmax{r_max}/snapshot_*.hdf5')

        times = []
        nbounds = []

        for snapshot_file in snapshot_files:
            t = float(snapshot_file.split('snapshot_')[1].split('.hdf5')[0])/365
            times.append(t)
            bodies = read_set_from_file(snapshot_file, 'hdf5')

            nbound = 0

            disk = bodies[bodies.name == 'disk']
            stars = Particles()
            stars.add_particles(bodies[bodies.name == 'primary_star'])
            stars.add_particles(bodies[bodies.name == 'secondary_star'])

            com = Particle()
            com.position = stars.center_of_mass()
            com.velocity = stars.center_of_mass_velocity()
            com.mass = stars.total_mass()

            for particle in disk:
                if bound(com, particle):
                    nbound += 1
            nbounds.append(nbound)

        all_times.append(times)
        all_nbounds.append(nbounds)
        all_r_min.append(float(r_min))
        all_r_max.append(float(r_max))

        print(f'Processed {len(snapshot_files)} snapshots in {time.time() - start_time:.2f} seconds')

    for i in range(len(all_times)):
        times = all_times[i]
        nbounds = all_nbounds[i]

        #sort times and nbounds
        times, nbounds = zip(*sorted(zip(times, nbounds)))
        times = np.array(times)
        nbounds = np.array(nbounds)

        r_min = all_r_min[i]
        r_max = all_r_max[i]

        np.save(f'{save_path}/times_rmin{r_min}_rmax{r_max}.npy', times)
        np.save(f'{save_path}/nbounds_rmin{r_min}_rmax{r_max}.npy', nbounds)

        plt.plot(times, nbounds, label=f'r_min={r_min}, r_max={r_max}')

    plt.xlabel('Time (yr)')
    plt.ylabel('Number of bound particles')
    plt.title('Number of bound particles over time')
    plt.xscale('log')

    plt.savefig(f'{save_path}/bound_particles.png')
    plt.savefig(f'{save_path}/bound_particles.pdf')