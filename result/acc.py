# python acc.py <filename> > r.out

import sys
import re

log_file = sys.argv[1]
acc_pattern = re.compile(r"\*\sAcc@1\s+([\d\.]+)")

with open(log_file) as f:
    for line in f:
        match = acc_pattern.search(line)
        if match:
            acc = float(match.group(1))
            print(acc)