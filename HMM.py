import random
import argparse
import codecs
import os
import numpy as np

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        # Load transitions
        self.transitions = {}
        with open(f"{basename}.trans", "r") as trans_file:
            for line in trans_file:
                parts = line.strip().split()
                state = parts[0]
                if state not in self.transitions:
                    self.transitions[state] = {}
                for i in range(1, len(parts), 2):
                    next_state = parts[i]
                    probability = parts[i + 1]
                    self.transitions[state][next_state] = probability  # keep as string if needed

        # Load emissions
        self.emissions = {}
        with open(f"{basename}.emit", "r") as emit_file:
            for line in emit_file:
                parts = line.strip().split()
                state = parts[0]
                if state not in self.emissions:
                    self.emissions[state] = {}
                for i in range(1, len(parts), 2):
                    output = parts[i]
                    probability = parts[i + 1]
                    self.emissions[state][output] = probability  # keep as string if needed

    # Generate a random sequence
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        state = '#'
        states = []
        emissions = []
        for i in range(n):
            next_state = np.random.choice(list(self.transitions[state].keys()), p=[float(x) for x in self.transitions[state].values()])
            states.append(next_state)
            emission = np.random.choice(list(self.emissions[next_state].keys()), p=[float(x) for x in self.emissions[next_state].values()])
            emissions.append(emission)
            state = next_state
        return Sequence(states, emissions)

    ## The forward algorithm:
    def forward(self, sequence):
        """return the most likely state sequence for the given sequence of observations."""
        # Number of states and sequence length
        states = list(self.transitions.keys())
        num_states = len(states)
        num_observations = len(sequence)

        # Initialize matrix to store forward probabilities
        M = np.zeros((num_states, num_observations))

        # Set start state probability to observation 1
        M[0, 0] = 1.0

        # Calculate probabilities for each state and observation
        for s in states:
            if s == '#': # Skip the start state
                continue
            # Check if the state is in the transitions and emissions
            if s in self.transitions[states[0]] and sequence[1] in self.emissions[s]:
                M[states.index(s), 1] = float(self.transitions[states[0]][s]) * float(self.emissions[s][sequence[1]])
            else:
                M[states.index(s), 1] = 0.0 # Set to 0 if not in transitions or emissions

        # Propagate forward
        for i in range(2, num_observations):
            for s in states:
                if s == '#': # Skip the start state
                    continue
                sum_prob = 0
                for s2 in states:
                    t = self.transitions[s2][s] if s in self.transitions[s2] else 0.0
                    e = self.emissions[s][sequence[i]] if sequence[i] in self.emissions[s] else 0.0
                    sum_prob += M[states.index(s2), i - 1] * float(t) * float(e)
                M[states.index(s), i] = sum_prob

        # Return the state with the highest possible value in the last column
        high_prob = np.argmax(M[:, num_observations - 1])
        return states[high_prob]

    ##  The Viterbi algorithm:
    def viterbi(self, sequence):
        """return the most likely state sequence for hidden states."""
        sequence.insert(0, '-') # Add a dummy state to the beginning of the sequence (fix for off-by-one error)
        # Number of states and sequence length
        states = list(self.transitions.keys())
        num_states = len(states)
        num_observations = len(sequence)

        # Initialize matrix to store forward probabilities
        M = np.zeros((num_states, num_observations))
        Backpointers = np.zeros((num_states, num_observations))

        # Set start state probability to observation 1
        M[0, 0] = 1.0

        # Calculate probabilities for each state and observation
        for s in states:
            if s == '#':
                continue
            if s in self.transitions[states[0]] and sequence[1] in self.emissions[s]:
                M[states.index(s), 1] = float(self.transitions[states[0]][s]) * float(self.emissions[s][sequence[1]])
            else:
                M[states.index(s), 1] = 0.0

        # Propagate forward
        for i in range(2, num_observations):
            for s in states:
                if s == '#':
                    continue
                max_prob = 0 # Initialize the maximum probability
                max_state = 0 # Initialize the state with the highest probability
                for s2 in states:
                    if s2 == '#':
                        continue
                    t = self.transitions[s2][s] if s in self.transitions[s2] else 0.0
                    e = self.emissions[s][sequence[i]] if sequence[i] in self.emissions[s] else 0.0
                    prob = M[states.index(s2), i - 1] * float(t) * float(e)
                    # Check if the probability is higher than the current maximum
                    if prob > max_prob:
                        max_prob = prob
                        max_state = states.index(s2) # Save the index of the state with the highest probability
                M[states.index(s), i] = max_prob
                Backpointers[states.index(s), i] = max_state

        most_likely = []
        # Find the most likely state
        most_likely.append(np.argmax(M[:, num_observations - 1]))
        # Find the most likely state for the rest of the observations
        for i in range(num_observations - 1, 1, -1):
            most_likely.append(int(Backpointers[most_likely[-1], i]))

        # Reverse the list to get the most likely state in the right order
        most_likely.reverse()
        for i in range(len(most_likely)):
            # Convert the index to the actual state
            most_likely[i] = states[most_likely[i]]

        return most_likely

if __name__ == '__main__':
    # Parse command line arguments
    # Let's user do sequence and more from the command line
    parser = argparse.ArgumentParser(description="HMM")
    parser.add_argument("model", help="Basename of the transition/emission files (e.g., 'cat' for 'cat.trans' and 'cat.emit')")
    parser.add_argument("--generate", type=int, help="Generate a random sequence of given length")
    parser.add_argument("--forward", type=str, help="Compute the forward probability of a given sequence")
    parser.add_argument("--viterbi", type=str, help="Compute the most likely sequence of hidden states for a given sequence of observations")

    args = parser.parse_args()
    hmm = HMM()
    hmm.load(args.model)

    # Generate a random sequence
    if args.generate:
        sequence = hmm.generate(args.generate)
        print("Generated sequence:\n",sequence)

        # e.g. cat_sequence.obs, lander_sequence.obs, etc.
        file_name = f"{args.model}_sequence.obs"

        # Write the sequence to a file
        with open(file_name, 'w') as obs_file:
            obs_file.write('\n'.join(sequence.outputseq))

    # Compute the forward probability
    if args.forward:
        with open(args.forward, 'r') as obs_file:
            observations = obs_file.read().strip().split()
            most_likely = hmm.forward(observations)
            if args.model == 'lander':
                # Safe places to land for the rover
                if most_likely in ['2,5', '3,4', '4,3', '4,4', '5,5']:
                    print("Forward: Safe to land!")
                else:
                    print("Forward: Not safe to land :(")

    # Compute the most likely sequence of hidden states
    if args.viterbi:
        with open(args.viterbi, 'r') as obs_file:
            observations = obs_file.read().strip().split()
            most_likely = hmm.viterbi(observations)
            print("Most likely hidden states:", most_likely)

    # else:
    #     print("HMM loaded with transitions and emissions:")
    #     print("Transitions:", hmm.transitions)
    #     print("Emissions:", hmm.emissions)