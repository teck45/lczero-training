
import numpy as np
from net import Net
import proto.net_pb2 as pb

import os
from subprocess import Popen, PIPE
import sys


def get_weights(obj, weights=None):
    if weights is None:
        weights = []
    if obj.DESCRIPTOR.name == "Layer":
        weights.append(obj)
    for field_descriptor, value in obj.ListFields():
        field_name = field_descriptor.name
        # print(field_name)
        if field_descriptor.type == field_descriptor.TYPE_MESSAGE:
            if field_descriptor.label == field_descriptor.LABEL_REPEATED:
                for nested_obj in value:
                    get_weights(nested_obj, weights)
            else:
                get_weights(value, weights)
    return weights


def apply_spsa(net_path, save_path_p=None, save_path_n=None, c=0.1):
    orig_net, positive_net, negative_net = Net(), Net(), Net()
    orig_net.parse_proto(net_path)
    positive_net.parse_proto(net_path)
    negative_net.parse_proto(net_path)
    orig_layers = get_weights(orig_net.pb)
    np_weights = [orig_net.denorm_layer_v2(w) for w in orig_layers]
    stdevs = [np.std(w) for w in np_weights]
    adjustments = []
    for w, stdev in zip(np_weights, stdevs):
        adjustments.append(c * stdev * np.random.choice([-1, 1], w.shape))
    
    positive_layers = get_weights(positive_net.pb)
    negative_layers = get_weights(negative_net.pb)

    print(len(positive_layers), len(negative_layers), len(adjustments), len(np_weights))
    assert len(positive_layers) == len(negative_layers) == len(adjustments) == len(np_weights)

    for pl, nl, adj, np_weights in zip(positive_layers, negative_layers, adjustments, np_weights):
        positive_net.fill_layer_v2(pl, np_weights + adj)
        negative_net.fill_layer_v2(nl, np_weights - adj)
    
    positive_net.save_proto(save_path_p)
    negative_net.save_proto(save_path_n)

    return orig_net, adjustments

def do_iteration(net_path, save_path_p, save_path_n, r = 0.1):
    # orig_net, adjustments = apply_spsa(net_path, save_path_p, save_path_n)


    lc0_path = r"C:/Users/danie/Documents/chess/lc0/lc0.exe"
    cutechess_path = "cutechess-cli.exe"

    os.chdir(r"C:/Program Files (x86)/Cute Chess/")

    "-tb /home/admin/syzygy "

    'option.BackendOptions="backend=cuda-fp16"'

    options = f"{cutechess_path} -tournament gauntlet -rounds 1 -games 2 -concurrency 1 -repeat -recover -openings file=UHO_Lichess_4852_v1.epd format=epd order=random -resign movecount=2 score=500 twosided=true -draw movenumber=30 movecount=10 score=8"
    each_options = '-each proto=uci tc=inf nodes=1 option.VerboseMoveStats=false'
    p_options = f"name=positive cmd={lc0_path} option.WeightsFile={save_path_p}"
    n_options = f"name=negative cmd={lc0_path} option.WeightsFile={save_path_n}"
    command = f"{options} -engine {p_options} -engine {n_options} {each_options} -pgnout games.pgn"

    print(command)

    process = Popen(command, shell = True, stdout = PIPE)
    output = process.communicate()[0]
    if process.returncode != 0:
        sys.stderr.write('failed to execute command: %s\n' % command)
        return 2

    elo = 0 # this is where we would run the games




    for l, adj in zip(orig_net.pb.layer, adjustments):
        weight = orig_net.denorm_layer_v2(l)
        new_weight = weight + r * elo * adj
        orig_net.fill_layer_v2(l, orig_net.norm_layer_v2(new_weight))


    orig_net.save_proto(net_path)


if __name__ == "__main__":
    netdir = r"C:/Users/danie/Documents/chess/lc0e/networks/"
    orig_path = netdir + r"t1-256x10-distilled-swa-2432500.pb.gz"
    do_iteration(orig_path, orig_path, orig_path)
    #do_iteration(, netdir + "a.pb.gz", netdir + "b.pb.gz")


