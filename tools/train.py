import sys
from optparse import OptionParser

sys.path.append('./')

import yolo
from yolo.utils.process_config import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",
                  help="configure filename")
(options, args) = parser.parse_args()
if options.configure:
    conf_file = str(options.configure)
else:
    print('please sspecify --conf configure filename')
    exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
dataset = eval(dataset_params['name'])(common_params, dataset_params)
g_net = eval(net_params['g_name'])(common_params, net_params)
d_net = eval(net_params['d_name'])(common_params, net_params)
solver = eval(solver_params['name'])(dataset, d_net, g_net, common_params, solver_params)
solver.solve()
