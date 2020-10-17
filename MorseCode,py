import random
import numpy as np
import matplotlib.pyplot as plt
# construct the Morse dictionary
alphabet = " ".join("abcdefghijklmnopqrstuvwxyz").split()
values = ['.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', '.---', '-.-', '.-..', '--', '-.','---', '.--.', '--.-', 
'.-.', '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..']
morse_dict = dict(zip(alphabet, values))
def morse_encode(word):
    return "*".join([dict_morse_encode[i] for i in " ".join(word).split()])

