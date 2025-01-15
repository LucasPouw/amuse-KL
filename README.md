# Disk stability and von Zeipel-Lidov-Kozai effect in AMUSE

Recently, the detection of an S-star binary around Sgr A* has been reported (Peissker et al., [2024](https://www.nature.com/articles/s41467-024-54748-3)). This system could have a circumbinary disk, which is suprising given the dynamic nature of the Galactic Center. The code in this repository can be used to test how long a disk can survive around a stellar binary that is orbiting a supermassive black hole (SMBH). Specifically, it allows the user to simulate a circumbinary disk. In such a system, the von Zeipel-Lidov-Kozai mechanism will have an effect on the binary and possibly on the disk, which can be investigated with this code as well. The simulation is implemented with a Bridged gravito-hydrodynamic code using the AMUSE suite. For more details on AMUSE, as well as installation requirements, we refer to [the AMUSE GitHub](https://github.com/spzwart/AMUSE), [the AMUSE book](https://iopscience.iop.org/book/mono/978-0-7503-1320-9) and the papers by Portegies Zwart et al. [(2009)](https://www.sciencedirect.com/science/article/abs/pii/S1384107608001085?via%3Dihub), Portegies Zwart et al. [(2013)](https://www.sciencedirect.com/science/article/abs/pii/S0010465512003116?via%3Dihub) and Pelupessy et al. [(2013)](https://www.aanda.org/articles/aa/full_html/2013/09/aa21252-13/aa21252-13.html).

## Contributors
- Yannick Badoux (badoux@strw.leidenuniv.nl)
- Lucas Pouw (pouw@strw.leidenuniv.nl)
- Tim van der Vuurst (vdvuurst@strw.leidenuniv.nl)
All authors contributed equally to the code.

## Features
- Choice of binary or single star orbiting the SMBH
- Option of gravity-only run without a disk
- Huayno and Hermite available as gravity solver
- Multithreading of hydrodynamics solver (Fi)
- Highly mutable parameter space (such as but not limited to BH mass, binary mass and various orbital parameters).

## Requirements and installation
Again, for a detailed overview we refer to the [AMUSE GitHub](https://github.com/spzwart/AMUSE). We strongly recommend using a ``conda`` virtual environment, and the code should work after following the AMUSE installation guide. A list of package versions is given in ```requirements.txt``` nonetheless. The code works on Linux machines and should work on MacOS as well (though this is untested) and is incompatible with Windows (since AMUSE is not compatible with Windows at time of writing). The code works perfectly fine on WSL, though.

## Usage

### Running a simulation
Running a simulation is done from `main.py`. If no arguments are specified, executing the command `python3 main.py` will start a single simulation of a binary with circumbinary disk orbiting an SMBH with all masses and orbital elements specified by Peissker et al. ([2024](https://www.nature.com/articles/s41467-024-54748-3)) for $2.5 \cdot 10^5$ years.

The distinction between a binary and single is made by the number of masses specified in the `--m_orb` argument. For example, a stellar binary around an SMBH without disk that runs for $10^4$ years is called with:
`python3 main.py --m_orb 2.72 0.80 --no_disk True --t_end 10000`

If you want to simulate a single star with a disk that has inner radius 3 AU and outer radius 10 AU, you may run
`python3 main.py --m_orb 3.52 --r_min 3 --r_max 10`. To do multiple runs, shrinking the disk with 10% its original width each iteration based on particle loss, add `--vary_radii True` (for details, see the report PDF). 

Currently, Huayno (Jänes et al., [2014](https://www.aanda.org/articles/aa/full_html/2014/10/aa23831-14/aa23831-14.html)) and Hermite (Makino, [1991](https://ui.adsabs.harvard.edu/abs/1991ApJ...369..200M/abstract)) are supported gravity solvers. The default is Huayno, but you can specify `--grav_code Hermite` to use Hermite. The only supported hydrodynamics solver is Fi (Hernquist and Katz, [1989](https://ui.adsabs.harvard.edu/abs/1989ApJS...70..419H/abstract); Gerritsen and Icke, [1997](https://adsabs.harvard.edu/full/1997A%26A...325..972G); Pelupessy et al., [2004](https://www.aanda.org/articles/aa/abs/2004/28/aa0071-04/aa0071-04.html)).

For convenience, here is a list of all arguments that can be specified:
`--m_smbh <BH mass>`
`--a_out <semi-major axis of outer orbit>`
`--e_out <eccentricity of outer orbit>`
`--m_orb <mass(es) of star(s)>`
`--a_in <semi-major axis of inner orbit>`
`--e_in <eccentricity of inner orbit>`
`--i_mut <mutual inclination of disk and binary>`
`--peri <argument of periapse of the inner orbit>`
`--r_min <inner disk radius>`
`--r_max <outer disk radius>`
`--m_disk <disk mass>`
`--n_disk <amount of SPH particles>`
`--dt <diagnostic timestep>`
`--t_end <simulation end time>`
`--file_dir <where simulation output will be stored>`
`--no_disk <run with or without disk>`
`--vary_radii <restart simulations with more stable radius until convergence>`
`--grav_code <specify Huayno or Hermite integrator>` 

### Simulation output
Running the `main.py` script will make a directory called `KL-output` located either one directory up from the GitHub clone, or at the path specified by `--file_dir`. Here, directories will be made named after the initial conditions used. Specifically, each non-default value will be put in the directory name. In these sub-folders, each diagnostic timestep (specified by `--dt`) an AMUSE snapshot will be stored in a snapshots folder. If `vary-radii True` is specified, there will be snapshot folders for each subsequent iteration. 

Not only snapshots are saved, but also `.npy` files containing the number of bound disk particles, the total energy of the particles in the gravity code, the total energy of the particles in the hydro code, the half-mass radius and the times at which these quantities have been evaluated (i.e., an array of the diagnostic times).

### Multithreading
To reduce the computation time, which is mainly set by the hydrodynamics solver, the user can multithread the hydro code. If there are $N$ total cores available, $N-2$ can be dedicated to the hydro code, while the other two are reserved for running the gravity code and Python. To use multithreading on Linux, change the `OMP_NUM_THREADS` variable in the terminal from which you will run `main.py`. For example, if you have 12 cores available, type `export OMP_NUM_THREADS=10` in the terminal to dedicate 10 of those to the hydro code.

### Making a movie
A series of plots can be made with `plotter.py`. Again, this can be controlled with terminal input thanks to an argument parser. You can specify the snapshot directory which you want to create plots of (`--file_dir`), control if you want to make a plot of every snapshot or every $n$-th one (`--step_size`) and specify if the simulation you are plotting is with or without disk (`--no_disk`). 

After the plots have been made, the script `moviemaker.py` can make all images in a specified directory into a movie, where you can specify the FPS, height, width and codec of the movie (see source code for more details). NOTE: if you want to use the codec 'avc1', which is much lighter than the default 'XVID', you must seperately install ... . This is *not* included in `requirements.txt`. 

### Data analysis
Most of the figures in the report can be reproduced by running `data_processing.py`. Please have a thorough look at the parser in the source code before doing so. The parser allows one to control which plot to create when running the script. By default, none will be created so please make sure to specify which plot you would like to make. If a save directory (`--save_dir`) is not specified, plots will not be saved but only output. 

Some additional figures are not present in this file in an attempt to not clutter the parser too much. The von Zeipel-Lidov-Kozai plot (e.g. as in figure **VOEG TOE**) may be recreated by running `kl_analysis.py`. Note that since this needs information of many orbital elements which are not gathered during simulation, running this for the first time on the output of a given simulation will take a while (but then the output is saved and the plot can be recreated quickly). Moreover, figure 1 from the report may be recreated by running `stable_radii.py`. 

## A simple test case
To test that the code works well for you, we suggest running the following commands in the terminal as a (relatively) quick test. This, depending on your machine, might still take some time - but such is the life of one doing computer simulations. Note that the given terminal commands here are for a Linux machine.

First, set-up and simulate a simple, low-resolution disk (100 SPH particles) around the binary and run it for at most 5 kyr by running:

```python3 main.py --t_end 5000 --n_disk 100```

Make sure to set `--vary_radii` to False (which is the default), since if this is true you may run more simulations than desired, since the code will try to find a stable configuration. Again, you may also specify the simulation output by giving a full path to `--file_dir`. Once the simulation is complete, you can create plots and a video of your simulation by running the following:

```python3 plotter.py --snapshot_dir <path/to/snapshot/folder/> --step_size 10 ``` 
```python3 moviemaker.py --image_dir <path/to/image/folder/> --video_name <your_file_name> ```

This will create a plot of every 10th snapshot, quickening the process, and make a movie from those plots. Include ``--codec avc1`` when calling `moviemaker.py` for a much lighter video, but note the requirement mentioned above.

Congratulations, you've ran a simulation and created a video of it! You can now also recreate our plots using e.g. `data_processing.py`, though this won't be very interesting, so save yourself some time and only do this for good runs. 

For any questions, please feel free to contact us.