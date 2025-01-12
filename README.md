# Disk stability and von Zeipel-Lidov-Kozai effect in AMUSE

Recently, the detection of an S-star binary around Sgr A* has been reported. This system could have a circumbinary disk, which is suprising given the dynamic nature of the Galactic Center. The code in this repository can be used to test how long a disk can survive around an S-star binary that is orbiting a supermassive black hole (SMBH). It allows the user to simulate a four-body problem in which a stellar binary with a circumbinary disk is orbiting an SMBH. In such a four-body problem, the von Zeipel-Lidov-Kozai mechanism will have an effect on the binary and possibly on the disk, which can be investigated with this code as well. The simulation is implemented with a Bridged gravito-hydrodynamic code using the AMUSE suite.

## Contributors
- Yannick Badoux
- Lucas Pouw
- Tim van der Vuurst

## Features
- Choice of binary or single star orbiting the SMBH
- Option of gravity-only run without a disk
- Homing in on stable disk radii based on particle loss
- Huayno and Hermite available as gravity solver
- Multithreading of hydrodynamics solver (Fi)

## Usage

### Running a simulation
Running a simulation is done from `main.py`.

The distinction between a binary and single is made by the number of masses specified in the `--m_orb` argument. For example, a stellar binary around an SMBH without disk that runs for $10^4$ years
`python3 main.py --m_orb 2.72 0.80 --no_disk True --t_end 10000`

If you want to simulate a single star with a disk that has inner radius 3 AU and outer radius 10 AU, you may run
`python3 main.py --m_orb 3.52 --r_min 3 --r_max 10`. To do multiple runs, shrinking the disk each time based on particle loss, add `--vary_radii True`.

For convenience, here is a list of all arguments that can be specified:
`--m_smbh
--a_out
--e_out
--m_orb
--a_in
--e_in
--i_mut
--peri
--r_min
--r_max
--m_disk
--n_disk
--dt
--t_end
--file_dir
--no_disk
--vary_radii
--grav_code`

### Multithreading
To reduce the computation time, which is mainly set by the hydrodynamics solver, the user can multithread the hydro code. If there are N total cores available, N-2 can be dedicated to the hydro code, while the other two are reserved for running the gravity code and Python. To use multithreading on Linux, change the `OMP_NUM_THREADS` variable in the terminal from which you will run `main.py`. For example, if you have 12 cores available, type `export OMP_NUM_THREADS=10` in the terminal to dedicate 10 of those to the hydro code.

### Plotting
`plotter.py`

### Making a movie
`moviemaker.py`

### Data analysis
