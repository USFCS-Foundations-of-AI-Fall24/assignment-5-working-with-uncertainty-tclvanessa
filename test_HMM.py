from unittest import TestCase
from HMM import *

class Test(TestCase):
    # Unit test for Load
    def test_load(self):
        h = HMM()
        h.load("cat")
        self.assertEqual({'#': {'happy': '0.5', 'grumpy': '0.5', 'hungry': '0'},
                                           'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'},
                                           'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'},
                                           'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}}, h.transitions)
        self.assertEqual({'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
                                         'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
                                         'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}, h.emissions)

    def test_generate(self):
        hmm = HMM()
        hmm.load("hmm")
        sequence = hmm.generate(10)
        self.assertEqual(len(sequence), 10)

    def test_forward(self):
        hmm = HMM()
        hmm.load("hmm")
        sequence = ['meow', 'meow', 'meow', 'meow', 'meow']
        self.assertEqual(hmm.forward(sequence), 0.0)

    def test_viterbi(self):
        hmm = HMM()
        hmm.load("hmm")
        sequence = ['meow', 'meow', 'meow', 'meow', 'meow']
        self.assertEqual(hmm.viterbi(sequence), ['happy', 'happy', 'happy', 'happy', 'happy'])
        sequence = ['meow', 'meow', 'meow', 'meow', 'meow', 'meow']