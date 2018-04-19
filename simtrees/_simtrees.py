import numpy as np
from kyleaoman_utilities.hdf5_io import hdf5_get
from simfiles import SimFiles
from importlib.util import spec_from_file_location, module_from_spec
from os.path import expanduser
from ._util import _log

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
        while to_grow != []:
            for node in to_grow:
                self.nodes[node.key] = node
            to_grow_next = []
            for node in to_grow:
                to_grow_next += node._grow()
            to_grow = to_grow_next
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
    def __init__(self, snap_id, configfile, ncpu=1, phantom=10000000000000000, simfiles_config=None):

        self.snap_id = snap_id
        self.configfile = configfile
        self.phantom = phantom

        self._read_config()
        
        tree_ids = hdf5_get(self.fpath, self.fbase, '/haloTrees/nodeIndex', ncpu=ncpu)
        sns = hdf5_get(self.fpath, self.fbase, '/haloTrees/snapshotNumber', ncpu=ncpu)
        gns = hdf5_get(self.fpath, self.fbase, '/haloTrees/fofIndex', ncpu=ncpu)
        tree_tabposs = hdf5_get(self.fpath, self.fbase, '/haloTrees/positionInCatalogue', ncpu=ncpu)
        in_tab = np.logical_not(
            hdf5_get(self.fpath, self.fbase, '/haloTrees/isInterpolated', ncpu=ncpu)
            )

        tree_descids = hdf5_get(self.fpath, self.fbase, '/haloTrees/descendantIndex', ncpu=ncpu)
        tree_mbpcs = hdf5_get(self.fpath, self.fbase, '/haloTrees/mbpsContributed', ncpu=ncpu)

        sgns = {}
        masstypes = {}
        keylist = []
        unique_sns = np.unique(sns)
        _log('TreeTables: Reading snapshots...')
        for i, sn in enumerate(unique_sns):
            _log('{0:.0f}/{1:.0f}'.format(i + 1, len(unique_sns)))
            
            SF = SimFiles(snap_id._replace(snap=sn), configfile=simfiles_config, ncpu=ncpu)
            SF.load(keys=('sgns', 'msubfind'))
            tf_sgns = SF.sgns
            tf_masstypes = SF.msubfind
            this_sn = np.logical_and(sns == sn, in_tab)
            for tree_id, tree_tabpos in zip(tree_ids[this_sn], tree_tabposs[this_sn]):
                sgns[tree_id] = tf_sgns[tree_tabpos].value
                masstypes[tree_id] = tf_masstypes[tree_tabpos]
            del SF['sgns'], SF['msubfind']
            del SF

        keylist = [key for key in tree_ids[in_tab]]
        sgns = np.array([sgns[key] for key in keylist])
        masstypes = np.array([masstypes[key] for key in keylist]) * masstypes[keylist[0]].unit

        self.sub_groups = dict(zip(tree_ids[in_tab], np.vstack((sns[in_tab], gns[in_tab], sgns)).T))
        self.sub_masstypes = dict(zip(tree_ids[in_tab], masstypes))

        self.tree_descid = dict(zip(tree_ids, tree_descids))
        self.tree_mbpc = dict(zip(tree_ids, tree_mbpcs))

        self._reverse()

        #explicit deletions to clean up memory a bit
        del tree_ids, tree_tabposs
        del tree_descids, tree_mbpcs
        del in_tab, sns, gns, sgns, masstypes

        return

    def _filter(self, include):
        #kill dict items with keys not in include
        self.sub_groups = {k:self.sub_groups[k] for k in include}
        self.sub_masstypes = {k:self.sub_masstypes[k] for k in include}
        self.tree_descid = {k:self.tree_descid[k] for k in include}
        self.tree_mbpc = {k:self.tree_mbpc[k] for k in include}
        self._reverse()
        return

    def _reverse(self):
        #construct reverse dict so that new root nodes can obtain their key from group
        self.sub_groups_r = {tuple(value):key for key, value in self.sub_groups.items() \
                                 if key // self.phantom == 0}
        return

    def mass_filter(self, cut, particle_type=1):
        #include only halos above mass cut for a mass of a given type 
        #(0:gas, 1:DM, 2:boundary, 3:boundary, 4:star, 5:BH)
        include = np.array([key for key, value in self.sub_masstypes.items() \
                                if value[particle_type] > cut]) 
        self._filter(include)
        return

    def _read_config(self):
    
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
            self.fpath, self.fbase = paths[self.snap_id]
        except KeyError:
            raise ValueError("TreeTables: unknown snapshot (not defined in configfile).")
    
        return
