import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
import argparse
import cpa_utils

parser = argparse.ArgumentParser()
parser.add_argument("--traces", help="Path to traces", required=True)
parser.add_argument("--num", help="Number of traces to use", type=int, default=20000)
parser.add_argument("--start", help="Sample where we start the analysis", type=int, default=0)
parser.add_argument("--count", help="Number of sample to use", type=int, default=25000)
parser.add_argument("--sa", help="Index of the sample used for the start of the alignment window", type=int, default=10)
parser.add_argument("--ea", help="Index of the sample used for the end of the alignment window", type=int, default=110)
parser.add_argument("--note", help="Add a note to plots", default="")
args = parser.parse_args()

start_point_for_align = args.sa
end_point_for_align   = args.ea
num_traces            = args.num
sample_start          = args.start
sample_count          = args.count

cyphers, traces = cpa_utils.load_npz_traces(num_traces, args.traces, sample_start, sample_count)

num_traces, num_samples = traces.shape

print("Align traces...")

reference_trace = cpa_utils.average_trace(traces[:200], start_point_for_align, end_point_for_align)
aligned_traces  = np.array([cpa_utils.align_trace(reference_trace, trace, start_point_for_align, end_point_for_align) for trace in traces])

del traces

BNUM = 16

cpaoutput = [0]*BNUM

with Bar("Plotting CPA", max=BNUM) as bar:
    for bnum in range(0, BNUM):

        sumnum = np.zeros(num_samples)
        sumden1 = np.zeros(num_samples)
        sumden2 = np.zeros(num_samples)

        hyp = np.zeros(num_traces)

        # print(f"Create hyp table for key {kguess:02X}...")
        for tnum in range(0, num_traces):
            hyp[tnum] = cpa_utils.hw(cyphers[tnum][bnum])

        #Mean of hypothesis
        meanh = np.mean(hyp, dtype=np.float64)

        #Mean of all points in trace
        meant = np.mean(aligned_traces, axis=0, dtype=np.float64)

        #For each trace, do the following
        for tnum in range(0, num_traces):
            hdiff = (hyp[tnum] - meanh)
            tdiff = aligned_traces[tnum,:] - meant
            sumnum = sumnum + (hdiff*tdiff)
            sumden1 = sumden1 + hdiff*hdiff
            sumden2 = sumden2 + tdiff*tdiff

        cpaoutput[bnum] = (sumnum / np.sqrt(sumden1*sumden2))

        plt.figure(figsize=(20, 5))
        plt.plot(cpaoutput[bnum], color="grey", zorder=1)
        plt.axvline(x=5000, color="blue", lw=3, zorder=2, label="AES-decrypt, Start")
        plt.title(f"{num_traces} traces, CPA against cyphertext[{bnum}]")
        plt.savefig(f"./leakage_cypher_byte_{bnum}_{num_traces}{args.note}.png", dpi=600)

        bar.next()

    bar.finish()

plt.figure(figsize=(20, 5))  # Width is 12, height is 5
for i in range(BNUM):
    plt.plot(cpaoutput[i], zorder=1)
plt.axvline(x=5000, color="blue", lw=3, zorder=2, label="AES-decrypt, Start")
plt.savefig(f"./leakage_cypher_all_bytes_{num_traces}{args.note}.png", dpi=600)

