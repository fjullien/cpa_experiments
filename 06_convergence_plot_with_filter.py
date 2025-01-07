import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from rich.table import Table
from rich.console import Console
import sys

from scipy.signal import savgol_filter

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
parser.add_argument("--bnum", help="Key byte to target", type=int, default=10)
args = parser.parse_args()

start_point_for_align = args.sa
end_point_for_align   = args.ea
num_traces            = args.num
sample_start          = args.start
sample_count          = args.count

console = Console(highlight=False)

KNOWN_ROUND_10_KEY = ["EE","BD","E8","B1","17","F0","5A","5C","66","0B","84","36","77","04","D0","B3"]

plaintexts, traces = cpa_utils.load_npz_traces(num_traces, args.traces, sample_start, sample_count)

num_traces, num_samples = traces.shape

print("Align traces...")

reference_trace = cpa_utils.average_trace(traces[:200], start_point_for_align, end_point_for_align)
aligned_traces  = np.array([cpa_utils.align_trace(reference_trace, trace, start_point_for_align, end_point_for_align) for trace in traces])

print("Filter traces...")

filtered_traces = np.empty_like(traces)
window_length = 17 # Choose a suitable window length (must be odd)
polyorder = 4  # Choose the order of the polynomial fit
for i in range(num_traces):
    filtered_traces[i] = savgol_filter(aligned_traces[i], window_length, polyorder)

filtered_traces_11 = np.empty_like(traces)
window_length = 11 # Choose a suitable window length (must be odd)
polyorder = 4  # Choose the order of the polynomial fit
for i in range(num_traces):
    filtered_traces_11[i] = savgol_filter(aligned_traces[i], window_length, polyorder)

del traces

# Generate the HW values of the T-table
t_table_hw_dec = cpa_utils.hw_t_table_decrypt()

#-----------------------------------------------------------------------
# This is our model. We return the hamming weight of the result of:
# - cypher xor keyguess (which is AddRoundKey)
# - output of the table with the input equal to the above result
#-----------------------------------------------------------------------
def leakage_model(cyphertext_byte, keyguess):
    return t_table_hw_dec[cyphertext_byte ^ keyguess]

# Key bytes we want to attack
bnum = args.bnum

cpa_tests = {
    "Non filtered"             : (aligned_traces,     [0]*256, [0]*256, [0]*256, "blue"),
    "Filtered, WL=17, Order=4" : (filtered_traces,    [0]*256, [0]*256, [0]*256, "green"),
    "Filtered, WL=11, Order=4" : (filtered_traces_11, [0]*256, [0]*256, [0]*256, "red"),
}

plt.figure(figsize=(20, 5))

key_guess_list = [0x84, 0xcb, 0xd6, 0x0e]

# For each guess of a key byte, we compute the coefficients
for traces_type in cpa_tests:
    traces, cpaoutput, maxcpa, cpa_evol, color = cpa_tests[traces_type]
    with Bar(f"Attacking key byte {bnum}", max=len(key_guess_list)) as bar:
        for kguess in key_guess_list:
            cpaoutput[kguess], maxcpa[kguess], cpa_evol[kguess] = cpa_utils.compute_coeff_with_convergence(bnum, kguess, plaintexts, leakage_model, traces)
            bar.next()

    # Sort the guesses by their coefficient (only the first 32)
    best_guesses = np.argsort(maxcpa)[-32:][::-1]
    result = " ".join(f"{i:02X}" for i in best_guesses)

    # Print the six best key guesses and their coefficient. Highlight the known key byte if present
    table = Table(title=f"Best guesses for key byte {bnum}, {traces_type}")
    table.add_column("Guess", justify="center", no_wrap=True)
    table.add_column("Coefficient", justify="center", no_wrap=True)
    table.add_column("Difference", justify="center", no_wrap=True)
    for i in range(len(key_guess_list)):
        style = None
        coeff = max(abs(cpaoutput[best_guesses[i]]))
        if i == 0:
            diff = "-"
        else:
            diff = (coeff - max(abs(cpaoutput[best_guesses[i-1]])))*100
        if best_guesses[i] == int(KNOWN_ROUND_10_KEY[bnum], 16):
            style = "bold green"
        table.add_row(f"{best_guesses[i]:02X}", str(coeff), str(diff), style=style)
    console.print(table)

    # Print the 32 best key guesses
    highlighted_text = result.replace(KNOWN_ROUND_10_KEY[bnum], "[green]["+KNOWN_ROUND_10_KEY[bnum]+"][/green]")
    console.print("Ranking: " + highlighted_text)

    print("\n\n")

    for kguess in key_guess_list:
        if kguess == int(KNOWN_ROUND_10_KEY[bnum], 16):
            plt.plot(cpa_evol[kguess][5000:], color=color, zorder=2, label=traces_type)
        else:
            plt.plot(cpa_evol[kguess][5000:], alpha=0.5, color=color, zorder=1)

plt.legend()
plt.savefig(f"cpa_convergence_filtered_for_key_{bnum}.png", dpi=600)
