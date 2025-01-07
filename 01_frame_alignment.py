import matplotlib.pyplot as plt
import numpy as np

import argparse
import cpa_utils

parser = argparse.ArgumentParser()
parser.add_argument("--traces", help="Path to traces", required=True)
parser.add_argument("--num", help="Number of traces to use", type=int, default=1000)
parser.add_argument("--start", help="Sample where we start the analysis", type=int, default=5500)
parser.add_argument("--count", help="Number of sample to use", type=int, default=5000)
parser.add_argument("--sa", help="Index of the sample used for the start of the alignment window", type=int, default=100)
parser.add_argument("--ea", help="Index of the sample used for the end of the alignment window", type=int, default=200)
args = parser.parse_args()

start_point_for_align = args.sa
end_point_for_align   = args.ea
num_traces            = args.num
sample_start          = args.start
sample_count          = args.count

cyphers, traces = cpa_utils.load_npz_traces(num_traces, args.traces, sample_start, sample_count)

num_traces, num_samples = traces.shape

for i in range(num_traces):
    plt.plot(traces[i][start_point_for_align:end_point_for_align], alpha=0.03, color="black")
plt.title("Unaligned traces")
plt.show()

def align_trace(reference, trace, start, end):
    subtrace = trace[start:end]
    correlation = np.correlate(subtrace, reference, mode='full')
    shift = np.argmax(correlation) - (len(subtrace) - 1)
    aligned_trace = np.roll(trace, -shift)
    return aligned_trace

print("Align traces...")

def average_trace(traces, start, end):
    sliced_traces = traces[:, start:end]
    return np.mean(sliced_traces, axis=0)

# We use the first 200 traces to create the reference trace for alignment
averaged_trace  = average_trace(traces[:200], start_point_for_align, end_point_for_align)
# The first trace is arbitrary choosen as reference
reference_trace = traces[0][start_point_for_align:end_point_for_align]

aligned_traces_0   = np.array([align_trace(reference_trace, trace, start_point_for_align, end_point_for_align) for trace in traces])
aligned_traces_avg = np.array([align_trace(averaged_trace, trace, start_point_for_align, end_point_for_align) for trace in traces])

# Create a figure with two subplots
fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

for i in range(num_traces):
    ax1.plot(traces[i][start_point_for_align:end_point_for_align], alpha=0.03, color="black")
    ax1.title.set_text("Raw traces")

    ax2.plot(aligned_traces_0[i][start_point_for_align:end_point_for_align], alpha=0.03, color="black")
    ax2.plot(reference_trace, linewidth=2, color="blue", label="Reference")
    ax2.title.set_text("Aligned traces (trace[0] as ref.)")

    ax3.plot(aligned_traces_avg[i][start_point_for_align:end_point_for_align], alpha=0.03, color="black")
    ax3.plot(averaged_trace, linewidth=2, color="red", label="Reference")
    ax3.title.set_text("Aligned traces (averaged ref.)")

plt.show()

for i in range(num_traces):
    plt.plot(aligned_traces_0[i][4800:5200], alpha=0.03, color="black")
plt.axvline(x=200, color="blue", lw=3, zorder=2, label="AES-decrypt, Start")
plt.legend()
plt.show()