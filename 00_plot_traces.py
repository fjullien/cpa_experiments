import matplotlib.pyplot as plt

import argparse
import cpa_utils

TRIGGER_POS  = 5000
SAMPLE_START = 0
SAMPLE_COUNT = 25000

parser = argparse.ArgumentParser()
parser.add_argument("--traces", help="Path to traces", required=True)
parser.add_argument("--num", help="Number of traces to use", type=int, default=1000)
args = parser.parse_args()

cyphers, traces = cpa_utils.load_npz_traces(args.num, args.traces, SAMPLE_START, SAMPLE_COUNT)

num_traces, num_samples = traces.shape

for i in range(num_traces):
	plt.plot(traces[i], alpha=0.03, color="black", zorder=1)
plt.axvline(x=TRIGGER_POS, color="blue", lw=3, zorder=2, label="AES-decrypt, Start")
plt.axvline(x=23500, color="red", lw=3, zorder=2, label="AES-decrypt, End")
plt.title(f"{args.n} overlaped traces")
plt.legend()
plt.show()