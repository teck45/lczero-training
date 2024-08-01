import numpy as np
from net import Net
import net
import proto.net_pb2 as pb
import os
from subprocess import Popen, PIPE
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from time import time, sleep
import argparse
from math import pow, sqrt, log, log10, copysign, pi

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
parser.add_argument('--perturbation_size', type=float, default=0.5, help='Perturbation size')
parser.add_argument('--test_interval', type=int, default=5, help='Interval for testing against original network')
parser.add_argument('--start_iteration', type=int, default=0, help='Iteration to start tuning from')
parser.add_argument('--syzygy', type=str, default="", help='Path to syzygy tablebase')
parser.add_argument('--prof_name', type=str, default="", help='Where to dump log info')




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
PROF_NAME = args.prof_name

if ROUNDS % GPUS != 0:
    ROUNDS = (ROUNDS // GPUS) * GPUS
    print("INFO: gpus does not divide rounds, reducing rounds accordingly")

# taken from https://github.com/jw1912/SPRT/blob/main/sprt.py

def erf_inv(x):
    a = 8 * (pi - 3) / (3 * pi * (4 - pi))
    y = log(1 - x * x)
    z = 2 / (pi * a) + y / 2
    return copysign(sqrt(sqrt(z * z - y / a) - z), x)


def phi_inv(p):
    return sqrt(2)*erf_inv(2*p-1)


def elo(score: float) -> float:
    if score <= 0 or score >= 1:
        return 0.0
    return -400 * log10(1 / score - 1)


def elo_wld(wins, losses, draws):
    # win/loss/draw ratio
    N = wins + losses + draws
    if N == 0:
        return (0, 0, 0)

    p_w = float(wins) / N
    p_l = float(losses) / N
    p_d = float(draws) / N

    mu = p_w + p_d/2
    stdev = sqrt(p_w*(1-mu)**2 + p_l*(0-mu)**2 + p_d*(0.5-mu)**2) / sqrt(N)

    # 95% confidence interval for mu
    mu_min = mu + phi_inv(0.025) * stdev
    mu_max = mu + phi_inv(0.975) * stdev

    return (elo(mu_min), elo(mu), elo(mu_max))

def get_wld_and_npm(output):
    npm = None
    wins, losses, draws = None, None, None
    for line in output.decode('utf-8').split('\n'):
        if line.startswith('tournamentstatus final'):
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'npm':
                    npm = float(parts[i+1])
            wins = int(parts[3])
            losses = int(parts[4].replace('-', ''))
            draws = int(parts[5].replace('=', ''))

    if npm is not None:         
        # return {'npm': npm, 'w': wins, 'l': losses, 'd': draws}
        return (npm, wins, losses, draws)
    else:
        return None

def run_cmd(command, results):
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    output, _ = process.communicate()
    info = get_wld_and_npm(output)
    if info is not None:
        results.put(info)

def do_iteration(net_path, save_path_p, save_path_n, save_path, r=LEARNING_RATE, do_spsa=True):
    if do_spsa:
        orig_net, adjustments = apply_spsa(net_path, save_path_p, save_path_n)

    rounds_per_gpu = ROUNDS // GPUS
    start_time = time()

    w, l, d = 0, 0, 0
    total_npm = 0
    avg_npm = -1
    n = 0

    results = multiprocessing.Queue()

    with ThreadPoolExecutor(max_workers=GPUS) as executor:
        tasks = []

        for gpu in range(GPUS):
            cmd = f"""{LC0_PATH} selfplay --player1.weights={save_path_p} --player2.weights={save_path_n} --openings-pgn={BOOK_PATH} --visits={NODES} --games={rounds_per_gpu*2} --mirror-openings=true --temperature=0 --noise-epsilon=0 --fpu-strategy=reduction --fpu-value=0.33 --fpu-strategy-at-root=reduction --fpu-value-at-root=0.33 --cpuct=1.75 --root-has-own-cpuct-params=false --policy-softmax-temp=1.36 --minibatch-size=32 --max-collision-visits=80000 --max-collision-events=917 --moves-left-max-effect=0.03 --moves-left-threshold=0.80 --moves-left-slope=0.0027  --moves-left-quadratic-factor=-0.65 --moves-left-scaled-factor=1.65 --moves-left-constant-factor=0.0 --threads=1 --task-workers=0 --no-share-trees --openings-mode=random  --parallelism=48 --backend=multiplexing --backend-opts=backend=cuda-auto,gpu={gpu}"""
            if SYZYGY:   
                cmd += f" --syzygy-paths={SYZYGY}"
            tasks.append(executor.submit(run_cmd, cmd, results))

        executor.shutdown(wait=True)

        while True:
            try:
                sleep(0.1)
                item = results.get_nowait()
                total_npm += item[0]
                w += item[1]
                l += item[2]
                d += item[3]
                n += 1
            except:
                break
  
        if (n > 0):
            avg_npm = total_npm / n
        mu_min, mu, mu_max = elo_wld(w, l, d)
        # print(f"{w=}, {l=}, {d=}, avg npm: {avg_npm:.2f}")
        # print(f"elo: {mu:.2f}, ({mu_min:.2f}, {mu_max:.2f})")
        print(f"elo: {mu:.2f} ± {mu_max - mu:.2f}, avg npm: {avg_npm:.2f}")

    print(f"Time elapsed: {time() - start_time:.2f} seconds")

    if do_spsa:
        for layer, adj in zip(get_weights(orig_net.pb), adjustments):
            weight = orig_net.denorm_layer_v2(layer)
            new_weight = weight + r * mu * adj
            orig_net.fill_layer_v2(layer, new_weight)

        orig_net.save_proto(save_path, log=False)
    
    return mu



def get_weights(lcnet, weights=None):
    return [net.nested_getattr(lcnet, "weights.ip_pol_w"), net.nested_getattr(lcnet, "weights.ip_pol_b")]

    return [
        net.nested_getattr(obj, "weights.policy.weights"),
        net.nested_getattr(obj, "weights.policy.biases"),
        # net.nested_getattr(obj, "weights.policy1.weights"),
    ]
    


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

    prof_name =  os.path.join(NET_DIR, PROF_NAME)
    prof = None
    if prof_name:
        import pickle
        # if the file exists, load it, otherwise set prof to an empty dictionary
        if os.path.exists(prof_name):
            with open(prof_name, 'rb') as f:
                prof = pickle.load(f)
        else:
            prof = {"elo":{}}
    
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
            this_elo = do_iteration("", new_path, old_path, "", do_spsa=False)
            if prof is not None:
                prof["elo"][iteration] = this_elo
                with open(prof_name, 'wb') as f:
                    pickle.dump(prof, f)

        iteration += 1