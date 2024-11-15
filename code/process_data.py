import numpy as np
import pandas as pd
import re

def LoadLogicData(mode = 'w'):
    """
    This functions pulls the dataset from hugging face API. 
    It returns a numpy array of propositional proofs of the form:
    [Premise 1, ... Premise n1, Derived Step 1, ... Derived Step n2, Conclusion]
    if mode = w, it will split the proof into words, otherwise it will split the proof by proof lines (mode = l). 
    """
    assert(mode in ['w', 'l'])


    #we are only interested in the proof column

    logic_data = pd.read_json("hf://datasets/ergotts/propositional-logic/output_proofs_divided.json")['propositional_logic_proof'].to_numpy()

    delimiters = '\n' # we will split the proofs up by lines


    for index, proof in enumerate(logic_data):
        #This is O(n^2) but the data set is not large so it's still very quick
        proof = proof.replace('¬', '¬ ') #we have to do this to be able to consider negation as a seperate word
        proof = proof.replace('c→', 'c →') #there are a few typos where there are no spaces between arrows and letters. 
        proof = re.split(delimiters, proof)
        proof  = [s.lower() for s in proof]
        
        if mode == 'w':

            proof  = "".join([s[s.find('.')+ 1 : s.find('(from')-1]  if s.find('(from') != -1 else s[s.find('.')+1:] for s in proof])[1:].split()

            proof = [s[:-1] if s[-1] == ":" else s for s in proof]
            proof = [s.replace("(", '') for s in proof]
            proof = [s.replace(")", '') for s in proof]

        else:
            #in order to split up by lines rather than words
            proof =  [s[s.find('.')+ 2 : s.find('(from')-1]  if s.find('(from') != -1 else s[s.find('.')+2:] for s in proof]

        if proof[0] == "ere's": #this is just to catch one badly formatted proof
            logic_data[index] = proof[9:]
        else:
            logic_data[index] = proof
            
    return logic_data



def generate_token_mapping(proofs):
    """ 
    map words to unique integers. 
    returns a word->int map and and inverse int->word map
    """
    import itertools

    tokens = list(itertools.chain.from_iterable(proofs)) #get all tokens appearing in the entire dataset as a single list
    # note that this is too inneficient for a lot of input, but it works out alright given our relatively small dataset
    values = set(tokens)
    keys = range(1, len(set(tokens))+1)

    int_to_word = dict(zip(keys, values)) #assign an integer to each token
    word_to_int =  dict(zip(values, keys))  #it's also important to know the inverse mapping

    return word_to_int, int_to_word

def tokenize(proof):
    """
    return n-gram sequence from the proof
    """
    output = []

    for i in range(2,len(proof)):
        output.append(proof[0:i])
    return output


def generate_sequences(proof_dataset):
    """
    create a list of sequences of mapped tokens from the whole dataset.
    """
    tokenized_proofs = []
    word_to_int, int_to_word = generate_token_mapping(proof_dataset)
    f = lambda word : word_to_int[word]

    for proof in proof_dataset:
        tokenized_proofs.append([f(word) for word in proof])

    return word_to_int, int_to_word, tokenized_proofs




def makeXy(sequenced_proofs):
    """
    Make the sequences into input sequences and labels by setting the label as the next word in the sequence
    """
    X, y = [],[]

    for s in sequenced_proofs:
        X.append(s[0:-1])
        y.append(s[1:])

    return X, y


def pad(sequences, value):
    """
    Pad the input sequences with a dumy value so that they all have the same length
    """
    #the length of the longest sequence n the set 
    max_length = max([len(s) for s in sequences])

    for s in sequences:
        
        s += [value for i in range(0, max_length-len(s))]

    return sequences





