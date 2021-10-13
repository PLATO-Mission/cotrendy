"""
Take a look at the cotrendy log file and grab all the MAP runtime
counts and do some stats. Things appear quite slow and we want to
try speeding them up otherwise processing everything will take forever

This script is used for older data where the cbvs.map_exec_times
object is not created
"""
import numpy as np
import argparse as ap

def arg_parse():
    """
    parse the command line arguments
    """
    p = ap.ArgumentParser()
    p.add_argument('logfile',
                   help='path to logfile')
    return p.parse_args()

if __name__ == "__main__":
    args = arg_parse()

    runtimes = []
    with open(args.logfile) as f:
        lines = f.readlines()

    for line in lines:
        if "Runtime: " in line:
            t = line.split("Runtime: ")[-1].split(" ")[0]
            runtimes.append(int(t))

    runtimes = np.array(runtimes)

    print(f"Average runtime: {np.average(runtimes)}")
    print(f"Std runtime: {np.std(runtimes)}")
    print(f"Min runtime: {np.min(runtimes)}")
    print(f"Max runtime: {np.max(runtimes)}")
