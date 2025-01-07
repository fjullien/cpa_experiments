import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from rich.table import Table
from rich.console import Console
import sys

from scipy.signal import find_peaks

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

console = Console(highlight=False)

KNOWN_ROUND_10_KEY = ["EE","BD","E8","B1","17","F0","5A","5C","66","0B","84","36","77","04","D0","B3"]

plaintexts, traces = cpa_utils.load_npz_traces(num_traces, args.traces, sample_start, sample_count)

num_traces, num_samples = traces.shape

print("Align traces...")

reference_trace = cpa_utils.average_trace(traces[:200], start_point_for_align, end_point_for_align)
aligned_traces  = np.array([cpa_utils.align_trace(reference_trace, trace, start_point_for_align, end_point_for_align) for trace in traces])

del traces

def find_lowest_value_index(array_list):
    # Flatten all arrays in the list and find the minimum value and its index
    lowest_value = float('inf')
    lowest_index = -1
    for idx, array in enumerate(array_list):
        if np.min(array) < lowest_value:
            lowest_value = np.min(array)
            lowest_index = idx
    return lowest_index


t_table_hw_dec = cpa_utils.hw_t_table_decrypt()

#-----------------------------------------------------------------------
# This is our model. We return the hamming weight of the result of:
# - cypher xor keyguess (which is AddRoundKey)
# - output of the table with the input equal to the above result
#-----------------------------------------------------------------------
def leakage_model(cyphertext_byte, keyguess):
    return t_table_hw_dec[cyphertext_byte ^ keyguess]

bestguess = [0]*16
bestguess_improved = [0]*16

# Number of key bytes we want to attack
BNUM = 16

for bnum in range(0, BNUM):

    cpaoutput = [0]*256
    maxcpa = [0]*256

    with Bar(f"Attacking key byte {bnum}", max=256) as bar:
        # For each guess of a key byte, we compute the coefficients
        for kguess in range(0, 256):
            cpaoutput[kguess], maxcpa[kguess] = cpa_utils.compute_coeff(bnum, kguess, plaintexts, leakage_model, aligned_traces)
            bar.next()
        print("\n\n")

        # Sort the guesses by their coefficient (only the first 32)
        best_guesses = np.argsort(maxcpa)[-32:][::-1]
        result = " ".join(f"{i:02X}" for i in best_guesses)

        # Print the six best key guesses and their coefficient. Highlight the known key byte if present
        table = Table(title=f"Best guesses for key byte {bnum}")
        table.add_column("Guess", justify="center", no_wrap=True)
        table.add_column("Coefficient", justify="center", no_wrap=True)
        for i in range(6):
            style = None
            if best_guesses[i] == int(KNOWN_ROUND_10_KEY[bnum], 16):
                style = "bold green"
            table.add_row(f"{best_guesses[i]:02X}", str(max(abs(cpaoutput[best_guesses[i]]))), style=style)
        console.print(table)

        # Print the 32 best key guesses
        highlighted_text = result.replace(KNOWN_ROUND_10_KEY[bnum], "[green]["+KNOWN_ROUND_10_KEY[bnum]+"][/green]")
        console.print("Ranking: " + highlighted_text)
        bestguess[bnum] = np.argmax(maxcpa)
        bar.finish()

        list_of_candidates = []
        for i in range(6):
            list_of_candidates.append(cpaoutput[best_guesses[i]])

        # Create a trace with lower values for each best_candidates
        merged_trace = np.minimum.reduce(list_of_candidates)

        # Get the minimum value from the first 500 points, that will be our noise floor
        min_value    = min(merged_trace[0:500])

        # Plot the best candidates and the noise level
        plt.figure(figsize=(20, 5))
        for i in range(6):
            plt.plot(list_of_candidates[i], label=f"{best_guesses[i]:02X}")#, alpha=0.3)
        plt.axhline(y=min_value, color="blue", lw=2, zorder=2, label="Noise level", linestyle='--')

        # Find (negative) peaks in the merged trace. We want to analyze the first peak.
        # We set the peak detection to be 1.5 times the level of the noise.
        peaks, _ = find_peaks(-merged_trace, prominence=-min_value*1.5)
        plt.plot(peaks, merged_trace[peaks], "xr", zorder=10, lw=2)

        # By default, the improved guess is the same as the "normal" one
        improved = bestguess[bnum]

        # If we find a peak, use the first one and plot the new window
        if len(peaks) > 0:
            plt.axvline(x=peaks[0]-10, color="black", lw=2, zorder=2, label="Refined window", linestyle='--')
            plt.axvline(x=peaks[0]+10, color="black", lw=2, zorder=2, label="Refined window", linestyle='--')

            # The new list of candidates has only samples from the refined window
            list_of_candidates = []
            for i in range(6):
                list_of_candidates.append(cpaoutput[best_guesses[i]][peaks[0]-10:peaks[0]+10])

            # In these traces, select the one with the lowest coefficient
            improved = best_guesses[find_lowest_value_index(list_of_candidates)]
            print("After windows refined: ", end="")
            style = None
            if int(KNOWN_ROUND_10_KEY[bnum], 16) == improved:
                style = "bold green"
            console.print(f"{improved:02X}", style=style)
            bestguess_improved[bnum] = improved
        else:
            console.print("No peak found", style="bold red")

        plt.legend()
        plt.title(f"{args.traces} - {num_traces} traces, Key index {bnum}")
        plt.figtext(0.5, 0, " ".join(sys.argv), ha="center")
        plt.savefig(f"key_guess_{bnum}.png", dpi=600, bbox_inches = "tight")

        # Plot a zoomed version
        if len(peaks) > 0:
            plt.xlim([peaks[0]-200, peaks[0]+200])
            plt.savefig(f"key_guess_zoomed_{bnum}.png", dpi=600, bbox_inches = "tight")

# Print complete guessed key
print("Guessed key         : ", end="")
for i, b in enumerate(bestguess):
    style = None
    if int(KNOWN_ROUND_10_KEY[i], 16) == b:
        style = "bold green"
    console.print(f"{b:02X} ", end="", style=style)
print("\n")

# Print complete guessed key
print("Guessed key improved: ", end="")
for i, b in enumerate(bestguess_improved):
    style = None
    if int(KNOWN_ROUND_10_KEY[i], 16) == b:
        style = "bold green"
    console.print(f"{b:02X} ", end="", style=style)
print("\n")
