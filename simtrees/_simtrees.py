import numpy as np
from simfiles import SimFiles
from simfiles._hdf5_io import hdf5_get
from importlib.util import spec_from_file_location, module_from_spec
from os.path import expanduser
from ._util import _log
import astropy.units as U

h = .704 #dimensionless Hubble constant
print('Using h={0:.3f}'.format(h))

class _Node:
    def __init__(self, key, desc=None, tree=None, treetables=None):
        self.key = key
        self.desc = desc
        self._treetables = treetables
        self._tree = tree
        #can't call self._grow here unless we want a huge function stack
        #don't initialize progs here so that we can distinguish no progs (ok) from uninitilized (error)
        return

    def _grow(self):
        kwargs = {'desc': self, 'tree': self._tree, 'treetables': self._treetables}
        self.progs = np.array([_Node(key, **kwargs) \
                                   for key, value in self._treetables.tree_descid.items() \
                                   if ((value == self.key) and (key != self.key))])
        self.progs = list(self.progs[self._argsort_progs(self.progs)])
        return self.progs

    def _argsort_progs(self, progs):
        #order progenitors by decreasing mbpc - most bound particles contributed
        keys = [prog.key for prog in progs]
        return np.argsort([self._treetables.tree_mbpc[key] for key in keys])[::-1]

class _Root(_Node):
    def __init__(self, group, tree=None, treetables=None):
        key = treetables.sub_groups_r[group]
        super().__init__(key, desc=None, tree=tree, treetables=treetables)
        return


class Tree:
    def __init__(self, group, treetables=None):
        _log('Tree: beginning re-construction.')
        self.treetables = treetables
        self.root = _Root(group, tree=self, treetables=self.treetables)
        self.nodes = {self.root.key: self.root}
        to_grow = self.root._grow()
        snaplevel = 0
        while to_grow != []:
            for node in to_grow:
                self.nodes[node.key] = node
            to_grow_next = []
            for node in to_grow:
                to_grow_next += node._grow()
            to_grow = to_grow_next
            _log('  level: {0:.0f}, total nodes: {1:.0f}, nodes in next level: {2:.0f}'.format(snaplevel, len(self.nodes), len(to_grow)))
            snaplevel += 1
        self._make_trunk()
        _log('Tree: re-construction complete.')
        return

    def _make_trunk(self):
        self.trunk = [self.root]
        while self.trunk[-1].progs:
            self.trunk.append(self.trunk[-1].progs[0])
        return

#TreeTables not a subclass of Tree since many Tree instances may share a TreeTables instance
#TreeTable instances are pickle-able
class TreeTables:
    def __init__(self, snap_id, configfile, ncpu=1, phantom=10000000000000000, simfiles_config=None, use_snapshots=False):

        self.snap_id = snap_id
        self.configfile = configfile
        self.ncpu = ncpu
        self.phantom = phantom
        self.simfiles_config = simfiles_config
        self.use_snapshots = use_snapshots

        self._read_config()
        self._read_treetables()
        
        self.sgns = {}
        self.masstypes = {}
        if self.use_snapshots:
            self._read_snapshots()
        else:
            self._read_subfindtables()
        self._sort_tables()
        self._reverse()

        #explicit deletions to clean up memory a bit
        del self.tree_ids
        del self.tree_descids, self.tree_mbpcs
        del self.in_tab, self.sns, self.gns, self.sgns, self.masstypes
        if self.use_snapshots:
            del self.tree_tabposs

        _log('TreeTables initialized.')

        return

    def _read_treetables(self):
        _log('TreeTables: reading merger tree tables:')
        _log('  nodeIndex')
        self.tree_ids = hdf5_get(self.fpath, self.fbase, '/haloTrees/nodeIndex', ncpu=self.ncpu)
        _log('  snapshotNumber')
        self.sns = hdf5_get(self.fpath, self.fbase, '/haloTrees/snapshotNumber', ncpu=self.ncpu)
        _log('  fofIndex')
        self.gns = hdf5_get(self.fpath, self.fbase, '/haloTrees/fofIndex', ncpu=self.ncpu)
        if self.use_snapshots:
            _log('  positionInCatalogue')
            self.tree_tabposs = hdf5_get(self.fpath, self.fbase, '/haloTrees/positionInCatalogue', ncpu=self.ncpu)
        _log('  isInterpolated')
        self.in_tab = np.logical_not(
            hdf5_get(self.fpath, self.fbase, '/haloTrees/isInterpolated', ncpu=self.ncpu)
            )
        _log('  descendantIndex')
        self.tree_descids = hdf5_get(self.fpath, self.fbase, '/haloTrees/descendantIndex', ncpu=self.ncpu)
        _log('  mbpsContributed')
        self.tree_mbpcs = hdf5_get(self.fpath, self.fbase, '/haloTrees/mbpsContributed', ncpu=self.ncpu)

        return

    def _read_snapshots(self):
        unique_sns = np.unique(self.sns)
        _log('TreeTables: Reading snapshots...')
        for i, sn in enumerate(unique_sns):
            _log('{0:.0f}/{1:.0f}'.format(i + 1, len(unique_sns)))
            SF = SimFiles(self.snap_id._replace(snap=sn), configfile=self.simfiles_config, ncpu=self.ncpu)
            SF.load(keys=('sgns', 'msubfind'))
            tf_sgns = SF.sgns
            tf_masstypes = SF.msubfind
            this_sn = np.logical_and(self.sns == sn, self.in_tab)
            for tree_id, tree_tabpos in zip(self.tree_ids[this_sn], self.tree_tabposs[this_sn]):
                self.sgns[tree_id] = tf_sgns[tree_tabpos].value
                self.masstypes[tree_id] = tf_masstypes[tree_tabpos]
            del SF['sgns'], SF['msubfind']
            del SF
        return

    def _read_subfindtables(self):
        _log('TreeTables: reading subfind tables:')
        _log('  nodeIndex')
        self.tf_tree_ids = hdf5_get(self.fpath, self.sfbase, '/Subhalo/nodeIndex', ncpu=self.ncpu)
        _log('  SubGroupNumber')
        self.tf_sgns = hdf5_get(self.fpath, self.sfbase, '/Subhalo/SubGroupNumber', ncpu=self.ncpu)
        _log('  MassType')
        self.tf_masstypes = hdf5_get(self.fpath, self.sfbase, '/Subhalo/MassType', ncpu=self.ncpu) * 1E10 / h * U.Msun
        return

    def _sort_tables(self):
        _log('TreeTables: sorting '+{True: 'snaphost', False: 'subfind'}[self.use_snapshots]+' tables.')
        if not self.use_snapshots:
            self.sgns = {tree_id: sgn for tree_id, sgn in zip(self.tf_tree_ids, self.tf_sgns)}
            self.masstypes = {tree_id: masstype for tree_id, masstype in zip(self.tf_tree_ids, self.tf_masstypes)}
        keylist = [key for key in self.tree_ids[self.in_tab]]
        self.sgns = np.array([self.sgns[key] for key in keylist])
        self.masstypes = np.array([self.masstypes[key].value for key in keylist]) * self.masstypes[keylist[0]].unit

        self.sub_groups = dict(zip(self.tree_ids[self.in_tab], np.vstack((self.sns[self.in_tab], self.gns[self.in_tab], self.sgns)).T))
        self.sub_masstypes = dict(zip(self.tree_ids[self.in_tab], self.masstypes))

        self.tree_descid = dict(zip(self.tree_ids, self.tree_descids))
        self.tree_mbpc = dict(zip(self.tree_ids, self.tree_mbpcs))

        return

    def _filter(self, include):
        _log('TreeTables: applying mask.')
        #kill dict items with keys not in include
        self.sub_groups = {k:self.sub_groups[k] for k in include}
        self.sub_masstypes = {k:self.sub_masstypes[k] for k in include}
        self.tree_descid = {k:self.tree_descid[k] for k in include}
        self.tree_mbpc = {k:self.tree_mbpc[k] for k in include}
        self._reverse()
        return

    def _reverse(self):
        _log('TreeTables: calculating "reversed" dicts.')
        #construct reverse dict so that new root nodes can obtain their key from group
        self.sub_groups_r = {tuple(value):key for key, value in self.sub_groups.items() \
                                 if key // self.phantom == 0}
        return

    def mass_filter(self, cut, particle_type=1):
        _log('TreeTables: evaluating mass filter.')
        #include only halos above mass cut for a mass of a given type 
        #(0:gas, 1:DM, 2:boundary, 3:boundary, 4:star, 5:BH)
        include = np.array([key for key, value in self.sub_masstypes.items() \
                                if value[particle_type] > cut]) 
        self._filter(include)
        return

    def _read_config(self):
        _log('TreeTables: reading config file.')
        try:
            spec = spec_from_file_location('config', expanduser(self.configfile))
            config = module_from_spec(spec)
            spec.loader.exec_module(config)
        except FileNotFoundError:
            raise FileNotFoundError("TreeTables: configfile '" + self.configfile + "' not found.")
        try:
            paths = config.paths
        except AttributeError:
            raise ValueError("TreeTables: configfile missing 'paths' definition.")
        try:
            self.fpath, self.fbase, self.sfbase = paths[self.snap_id]
        except KeyError:
            raise ValueError("TreeTables: unknown snapshot (not defined in configfile).")
    
        return
