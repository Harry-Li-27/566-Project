# python acc.py <model> <filename> > r.out

import sys
import re

model = sys.argv[1]
log_file = sys.argv[2]

if model == "moco":
    acc_pattern = re.compile(r"\*\sAcc@1\s+([\d\.]+)")
    with open(log_file) as f:
        for line in f:
            match = acc_pattern.search(line)
            if match:
                acc = float(match.group(1))
                print(acc)
elif model == "dino":
    acc_pattern = re.compile(r"\"test_acc1\":\s+([\d\.]+)")
    with open(log_file) as f:
        for line in f:
            match = acc_pattern.search(line)
            if match:
                acc = float(match.group(1))
                print(acc)
