import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from rich.table import Table
from rich.console import Console
import sys

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

# Generate the HW values of the T-table
t_table_hw_dec = cpa_utils.hw_t_table_decrypt()

#-----------------------------------------------------------------------
# This is our model. We return the hamming weight of the result of:
# - cypher xor keyguess (which is AddRoundKey)
# - output of the table with the input equal to the above result
#-----------------------------------------------------------------------
def leakage_model(cyphertext_byte, keyguess):
    return t_table_hw_dec[cyphertext_byte ^ keyguess]

bestguess = [0]*16

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

        plt.figure(figsize=(20, 5))
        for i in range(6):
            plt.plot(cpaoutput[best_guesses[i]], label=f"{best_guesses[i]:02X}")
        plt.legend()
        plt.title(f"{args.traces} - {num_traces} traces, Key index {bnum}")
        plt.figtext(0.5, 0, " ".join(sys.argv), ha="center")
        plt.savefig(f"key_guess_{bnum}.png", dpi=600, bbox_inches = "tight")

# Print complete guessed key
print("Guessed key: ", end="")
for i, b in enumerate(bestguess):
    style = None
    if int(KNOWN_ROUND_10_KEY[i], 16) == b:
        style = "bold green"
    console.print(f"{b:02X} ", end="", style=style)
print("\n")
