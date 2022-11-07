import h5py as h5py
import numpy as np
import multiprocessing
import os.path


class _hdf5_io():

    def __init__(self, path, fbase, ncpu=0, interval=None):
        self._path = path
        self._fbase = fbase
        self._parts = self._find_parts(self._path, self._fbase)
        self._nb_cpu = multiprocessing.cpu_count() - 1 if ncpu == 0 else ncpu
        self._interval = interval

    def _subitem_interval(self, name, parts, output, intervals):
        accumulator = []
        for part, interval in zip(parts, intervals):
            with h5py.File(part, 'r') as f:
                accumulator.append(f[name][interval[0]: interval[1]].copy())
        output.put(accumulator)
        return

    def __getitem__(self, name):
        items = []
        all_interval_parts = self._split_interval(name)
        all_parts = [p for p, i in zip(self._parts, all_interval_parts)
                     if not i is False]
        all_interval_parts = [i for i in all_interval_parts if not i is False]
        if self._nb_cpu > 1:
            parts_split = np.array_split(all_parts, self._nb_cpu)
            interval_parts_split = np.array_split(
                all_interval_parts, self._nb_cpu
            )
            procs = []
            outputs = []
            try:
                for parts, interval_parts in zip(
                        parts_split, interval_parts_split):
                    outputs.append(multiprocessing.Queue())
                    target = self._subitem_interval
                    args = (
                        name,
                        parts.tolist(),
                        outputs[-1],
                        interval_parts.tolist()
                    )
                    procs.append(
                        multiprocessing.Process(target=target, args=args)
                    )
                    procs[-1].start()
                for output in outputs:
                    items += output.get()
                for p in procs:
                    p.join()
            except IOError:
                self._nb_cpu = 1  # fallback to serial mode
                return self[name]
        else:
            for part, interval_part in zip(all_parts, all_interval_parts):
                with h5py.File(part, 'r') as f:
                    startslice, endslice = interval_part
                    items.append(f[name][startslice: endslice].copy())
        if len(items) == 0:
            raise KeyError("Unable to open object (Object '{:s}' doesn't exist"
                           " in file with path '{:s}' and basename '{:s}')"
                           .format(name, self._path, self._fbase))
        else:
            return np.concatenate(items)

    def _split_interval(self, name):
        slices = []
        start = 0
        for part in self._parts:
            with h5py.File(part, 'r') as f:
                try:
                    end = start + f[name].shape[0]
                except KeyError:
                    slices.append(False)
                    continue
                if self._interval is not None:
                    if self._interval[0] <= start:
                        startslice = 0
                    elif self._interval[0] <= end:
                        startslice = self._interval[0] - start
                    else:
                        slices.append(False)
                        start = end
                        continue
                    if self._interval[1] >= end:
                        endslice = end - start
                    elif self._interval[1] >= start:
                        endslice = self._interval[1] - start
                    else:
                        slices.append(False)
                        start = end
                        continue
                else:
                    startslice = 0
                    endslice = end - start
                slices.append((startslice, endslice))
                start = end
        if self._interval is not None:
            if end < self._interval[1]:
                raise ValueError('Mask interval contains larger indices than '
                                 'number of particles in files.')
        return slices

    def _find_parts(self, path, fbase):
        monolithic = os.path.join(
            path,
            '{:s}.hdf5'.format(fbase)
        )

        def part(fcount):
            return os.path.join(
                path,
                '{:s}.{:.0f}.hdf5'.format(fbase, fcount)
            )

        if os.path.exists(monolithic):
            return [monolithic]
        elif os.path.exists(part(0)):
            fcount = 0
            retval = []
            while os.path.exists(part(fcount)):
                retval.append(part(fcount))
                fcount += 1
            return retval
        else:
            raise IOError("Unable to open file (File with path '{:s}' and "
                          "basename '{:s}' doesn't exist)".format(path, fbase))

    def get_parts(self):
        return self._parts


def hdf5_get(path, fbase, hpath, attr=None, ncpu=0, interval=None):
    """
    Retrieve and assemble data from an hdf5 fileset.

    Parameters
    ----------
    path: str
        Directory containing hdf5 file(s).

    fbase: str
        Filename, omit '.X.hdf5' portion.

    hpath: str
        'Internal' path of data table to gather, e.g. '/PartType1/ParticleIDs'

    attr: str
        Name of attribute to fetch (optional).

    ncpu: int
        Read in parallel with the given cpu count (default: 0 -> all cpus).

    interval: tuple
        Read a subset of a dataset in the given interval (2-tuple) of indices.

    Returns
    -------
    out : DataSet or contents of attribute
        Contents of requested dataset or attribute.
    """

    if not attr:
        hdf5_file = _hdf5_io(path, fbase, ncpu=ncpu, interval=interval)
        retval = hdf5_file[hpath]
        return retval
    else:
        for fname in _hdf5_io(path, fbase, ncpu=ncpu).get_parts():
            with h5py.File(fname, 'r') as f:
                try:
                    return f[hpath].attrs[attr]
                except KeyError:
                    continue
        raise KeyError("Unable to open attribute (One of object '{:s}' or "
                       "attribute '{:s}' doesn't exist in file with path "
                       "'{:s}' and basename '{:s}')"
                       .format(hpath, attr, path, fbase))
