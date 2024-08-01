import numpy as np
from net import Net
import net
import proto.net_pb2 as pb
import os
from subprocess import Popen, PIPE
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import sys
from time import time
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='SPSA training script for Leela Chess Zero')
parser.add_argument('lc0_path', help='Path to lc0 executable')
parser.add_argument('book_path', help='Path to opening book')
parser.add_argument('net_dir', help='Directory for network files')
parser.add_argument('base_name', help='Base name for network files')
parser.add_argument('--ext', default=".pb.gz", help='File extension for network files')
parser.add_argument('--rounds', type=int, default=150, help='Number of rounds')
parser.add_argument('--nodes', type=int, default=64, help='Number of nodes')
parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate')
parser.add_argument('--perturbation_size', type=float, default=1.0, help='Perturbation size')
parser.add_argument('--test_interval', type=int, default=5, help='Interval for testing against original network')
parser.add_argument('--start_iteration', type=int, default=0, help='Iteration to start tuning from')
parser.add_argument('--syzygy', type=str, default="", help='Path to syzygy tablebase')



args = parser.parse_args()

# Configuration (now using parsed arguments)
LC0_PATH = args.lc0_path
BOOK_PATH = args.book_path
NET_DIR = args.net_dir
BASE_NAME = args.base_name
EXT = args.ext
ROUNDS = args.rounds
NODES = args.nodes
GPUS = args.gpus
LEARNING_RATE = args.learning_rate
PERTURBATION_SIZE = args.perturbation_size
TEST_INTERVAL = args.test_interval
START_ITERATION = args.start_iteration
SYZYGY = args.syzygy

if ROUNDS % GPUS != 0:
    ROUNDS = (ROUNDS // GPUS) * GPUS
    print("INFO: gpus does not divide rounds, reducing rounds accordingly")

# Rest of the code remains the same
def is_number(s):
    return s.lstrip('-').replace('.','',1).isdigit()

def get_elo_and_npm(output):
    elo, los, total_npm, npm_count = None, None, 0, 0
    for line in output.decode('utf-8').split('\n'):
        if line.startswith('tournamentstatus final'):
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'Elo:':
                    elo = float(parts[i+1])
                    # Find the LOS value, which should be two parts after 'LOS:'
                    los_index = parts.index('LOS:')
                    los = float(parts[los_index + 1].rstrip('%'))
                if part == 'npm':
                    total_npm += float(parts[i+1])
                    npm_count += 1
    avg_npm = total_npm / npm_count if npm_count > 0 else 0
    return elo, los, avg_npm

def run_cmd(command, results):
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    output, _ = process.communicate()
    elo, error, npm = get_elo_and_npm(output)
    if elo is not None:
        results.put((elo, error, npm))

def do_iteration(net_path, save_path_p, save_path_n, save_path, r=LEARNING_RATE, do_spsa=True):
    if do_spsa:
        orig_net, adjustments = apply_spsa(net_path, save_path_p, save_path_n)

    rounds_per_gpu = ROUNDS // GPUS
    start_time = time()

    elo = 0
    error = 0
    total_npm = 0
    n = 0

    results = multiprocessing.Queue()

    with ThreadPoolExecutor(max_workers=GPUS) as executor:
        tasks = []

        for gpu in range(GPUS):
            cmd = f"""{LC0_PATH} selfplay --player1.weights={save_path_p} --player2.weights={save_path_n} --openings-pgn={BOOK_PATH} --visits={NODES} --games={rounds_per_gpu*2} --mirror-openings=true --temperature=0 --noise-epsilon=0 --fpu-strategy=reduction --fpu-value=0.23 --fpu-strategy-at-root=absolute --fpu-value-at-root=1.0 --cpuct=1.32 --cpuct-at-root=1.9 --root-has-own-cpuct-params=true --policy-softmax-temp=1.4 --minibatch-size=256 --out-of-order-eval=true --max-collision-visits=9999 --max-collision-events=32 --cache-history-length=0 --smart-pruning-factor=1.33 --sticky-endgames=true --moves-left-max-effect=0.2 --moves-left-threshold=0.0 --moves-left-slope=0.007 --moves-left-quadratic-factor=0.85 --moves-left-scaled-factor=0.15 --moves-left-constant-factor=0.0  --openings-mode=random  --parallelism=48 --backend=multiplexing --backend-opts=backend=cuda-auto,gpu={gpu}"""
            if SYZYGY:
                cmd += f" --syzygy-paths={SYZYGY}"
            tasks.append(executor.submit(run_cmd, cmd, results))

        executor.shutdown(wait=True)

        while True:
            try:
                item = results.get_nowait()
                elo += item[0]
                error += item[1]
                total_npm += item[2]
                n += 1
            except:
                break
  
        if (n > 0):
            elo /= n
            error /= n ** (3/2)
            avg_npm = total_npm / n
        print(f"elo: {elo:.2f}, error: {error:.2f}, avg npm: {avg_npm:.2f}")

    print(f"Time elapsed: {time() - start_time:.2f} seconds")

    if do_spsa:
        for l, adj in zip(get_weights(orig_net.pb), adjustments):
            weight = orig_net.denorm_layer_v2(l)
            new_weight = weight + r * elo * adj
            orig_net.fill_layer_v2(l, new_weight)

        orig_net.save_proto(save_path, log=False)



def get_weights(obj, weights=None):
    return [net.nested_getattr(obj, "weights.ip_pol_w"), net.nested_getattr(obj, "weights.ip_pol_b")]

def apply_spsa(net_path, save_path_p=None, save_path_n=None, c=PERTURBATION_SIZE):
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
        positive_net.fill_layer_v2(pl, np_weights + adj)
        negative_net.fill_layer_v2(nl, np_weights - adj)
    
    positive_net.save_proto(save_path_p, log=False)
    negative_net.save_proto(save_path_n, log=False)

    return orig_net, adjustments


if __name__ == "__main__":
    iteration = START_ITERATION
    while True:
        print(f"\nStarting iteration {iteration} with {ROUNDS} rounds")

        name = os.path.join(NET_DIR, f"{BASE_NAME}-{iteration}")

        orig_path = name + EXT
        p_path = name + "-p" + EXT
        n_path = name + "-n" + EXT
        save_path = os.path.join(NET_DIR, f"{BASE_NAME}-{iteration + 1}" + EXT)
        do_iteration(orig_path, p_path, n_path, save_path)
        os.remove(p_path)
        os.remove(n_path)

        if iteration % TEST_INTERVAL == 0:
            new_path = name + EXT
            old_path = os.path.join(NET_DIR, BASE_NAME + "-0" + EXT)
            print(f"\nTesting new {new_path} against old {old_path}")
            do_iteration("", new_path, old_path, "", do_spsa=False)

        iteration += 1