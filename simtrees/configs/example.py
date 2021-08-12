from simtrees._setup_cfg import paths
from collections import namedtuple

# define snapshot unique id tuple format
snap_id = namedtuple('snap_id', ['box', 'res', 'model', 'snap'])

path_base = '/gpfs/data/jch/Eagle/Database/TreeData'

boxes = \
    {
        'L0012': {
            'N0188': ['DMONLY', 'REFERENCE'],
        },
        'L0025': {
            'N0376': ['DMONLY'],
            'N0752': ['DMONLY', 'RECALIBRATED']
        },
        'L0100': {
            'N1504': ['DMONLY']
        }
    }

box_list = [(box, res, model) for box, v in boxes.items() for res, vv in v.items() for model in vv]

snap = 28
for box, res, model in box_list:
    if (box, res, model) == ('L0012', 'N0188', 'REFERENCE'):
        pathmodel = 'REF_COSMA5'
    else:
        pathmodel = model
    fpath = path_base + '/' + box + res + '_' + pathmodel + '/trees/treedir_{:03d}/'.format(snap)
    fbase = 'tree_{:03d}'.format(snap)
    paths[snap_id(box=box, res=res, model=model, snap=snap)] = \
        (fpath, fbase)
