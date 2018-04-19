# simtrees
Framework to simplify loading data from merger trees in John Helly's format.

## Installation:
 - Download via web UI, or `git clone https://github.com/kyleaoman/simtrees.git`
 - Install dependencies if necessary (see [`setup.py`](https://github.com/kyleaoman/simtrees/blob/master/setup.py)), some may be found in [other repositories by kyleaoman](https://github.com/kyleaoman?tab=repositories).
 - Global install (Linux): 
   - cd to directory with [`setup.py`](https://github.com/kyleaoman/simtrees/blob/master/setup.py)
   - run `sudo pip install -e .` (`-e` installs via symlink, so pulling repository will do a 'live' update of the installation)
 - User install (Linux):
   - cd to directory with [`setup.py`](https://github.com/kyleaoman/simtrees/blob/master/setup.py)
   - ensure `~/lib/python3.6/site-packages` or similar is on your `PYTHONPATH` (e.g. `echo $PYTHONPATH`), if not, add it (perhaps in `.bash_profile` or similar)
   - run `pip install --prefix ~ -e .` (`-e` installs via symlink, so pulling repository will do a 'live' update of the installation)
 - cd to a directory outside the module and launch `python`; you should be able to do `from simtrees import Tree, TreeTables`

## Usage:

???