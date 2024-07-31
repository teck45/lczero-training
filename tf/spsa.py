
import numpy as np
from net import Net
import net
import proto.net_pb2 as pb

import os
from subprocess import Popen, PIPE
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import sys

def is_number(s):
    return s.lstrip('-').replace('.','',1).isdigit()

def get_elo(output):
    # the elo is between "Elo difference:" and "LOS:"
    x = output.find(b"Elo difference:")
    y = output.find(b"LOS:")
    ticket = str(output[x:y])
    ticket = ticket.replace("Elo difference:", "").replace(",", "").replace("'", "").replace(" ", "").replace("b", "")
    ticket = ticket.split("+/-")
    if is_number(ticket[0]) and is_number(ticket[1]):
        return float(ticket[0]), float(ticket[1])
    else:
        return None, None

def run_cmd(command, results):
    process = Popen(command, shell = True, stdout = PIPE)
    output = process.communicate()[0]
    elo, error = get_elo(output)

    if elo is None:
        return
    
    results.put((elo, error))


def get_weights(obj, weights=None):

    return [net.nested_getattr(obj, "weights.policy.weights"),
            # net.nested_getattr(obj, "weights.policy.biases"),
            # net.nested_getattr(obj, "weights.policy1.weights"),
            ]
    return [net.nested_getattr(obj, "weights.ip_pol_w"), net.nested_getattr(obj, "weights.ip_pol_b")]

    if weights is None:
        weights = []
    if obj.DESCRIPTOR.name != "Layer":
        print(obj.DESCRIPTOR.full_name)

        print(dir(obj))
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


def apply_spsa(net_path, save_path_p=None, save_path_n=None, c=3.0):
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

    assert len(positive_layers) == len(negative_layers) == len(adjustments) == len(np_weights)

    for pl, nl, adj, np_weights in zip(positive_layers, negative_layers, adjustments, np_weights):
        print(np_weights.shape)
        positive_net.fill_layer_v2(pl, np_weights + adj)
        negative_net.fill_layer_v2(nl, np_weights - adj)
    
    positive_net.save_proto(save_path_p)
    negative_net.save_proto(save_path_n)

    return orig_net, adjustments

def do_iteration(net_path, save_path_p, save_path_n, save_path, r = 0.0005, do_spsa=True):
    if do_spsa:
        orig_net, adjustments = apply_spsa(net_path, save_path_p, save_path_n)


    lc0_path = r"/home/privateclient/spsa/lc0/lc0"
    cutechess_path = "/home/privateclient/spsa/cutechess/build/cutechess-cli"






    rounds = 204
    nodes = 64
    gpus = 7

    from time import time

    rounds_per_gpu = rounds // gpus

    start_time = time()

    elo = 0
    error = 0
    n = 0

    results    = multiprocessing.Queue()

    with ThreadPoolExecutor(max_workers=gpus) as executor:
        tasks = []

        for gpu in range(gpus):
        
            options = f"{cutechess_path} -tournament gauntlet -rounds {rounds_per_gpu} -games 2 -concurrency 1 -tb /home/privateclient/spsa/syzygy -repeat -recover -openings file=/home/privateclient/spsa/opbooks/UHO_Lichess_4852_v1.epd format=epd order=random -resign movecount=2 score=400 twosided=true -draw movenumber=30 movecount=10 score=20"
            each_options = f'-each proto=uci tc=inf nodes={nodes} option.VerboseMoveStats=false option.BackendOptions="gpu={gpu}"'
            p_options = f"name=positive cmd={lc0_path} option.WeightsFile={save_path_p}"
            n_options = f"name=negative cmd={lc0_path} option.WeightsFile={save_path_n}"
            cmd = f"{options} -engine {p_options} -engine {n_options} {each_options} -pgnout games.pgn -debug"
            
            tasks.append(executor.submit(run_cmd, cmd, results))

        executor.shutdown(wait=True)

        while True:
            try:
                item = results.get_nowait()
                elo += item[0]
                error += item[1]
                n += 1
            except:
                break
  
        if (n > 0):
            elo /= n
            error /= n ** (3/2)
        print(f"elo: {elo}, error: {error}")
            

    #elo = 0 # this is where we would run the games

    print(f"Time elapsed: {time() - start_time}")

    if do_spsa:

        for l, adj in zip(get_weights(orig_net.pb), adjustments):
            weight = orig_net.denorm_layer_v2(l)
            new_weight = weight + r * elo  * adj
            orig_net.fill_layer_v2(l, new_weight)


        orig_net.save_proto(save_path)


if __name__ == "__main__":
    iteration = 0
    ext = ".pb.gz"
    base_name = "t74"
    netdir = r"/home/privateclient/spsa/lc0/nets/"
    while True:
        print(f"Iteration {iteration}:")
        name = netdir + f"{base_name}-{iteration}"

        orig_path = name + ext
        p_path = name + "-p" + ext
        n_path = name + "-n" + ext
        save_path = netdir + f"{base_name}-{iteration + 1}" + ext
        do_iteration(orig_path, p_path, n_path, save_path)
        os.remove(p_path)
        os.remove(n_path)

        if iteration % 5 == 0:
            print("TESTING VS ORIGINAL")
            do_iteration("", name + ext, netdir + base_name + "-0" + ext, "", do_spsa=False)


        iteration += 1
    #do_iteration(, netdir + "a.pb.gz", netdir + "b.pb.gz")

