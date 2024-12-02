import matplotlib.pyplot as plt
import model
import process_data
import torch as t
import torch.utils.data  as data
import numpy as np


#load the data from hugging face mode = 'w' means that we are tokenizing words rather than characters or sentences. 
proofs_dataset = process_data.LoadLogicData(mode = 'w') 

#format the proofs: essentially just mapping words to integers and then creating n-gram sequences
word_to_int, int_to_word, sequenced_proofs = process_data.generate_sequences(proofs_dataset)

#split data into input and label by setting label equal to next word.
#sequence length is the length of eqch sequence, this allows us to pack them during training. 
X, sequence_lengths,y = process_data.makeXy(sequenced_proofs)

percentages = np.arange(0.1, 1, .1)


accuracies = []
model_sizes = []
training_sizes = []


for p in percentages:

    X_train, X_val, train_lengths, val_lengths, y_train, y_val = process_data.makeSplit(X,sequence_lengths, y, p)

    #train data
    X_train = t.tensor(X_train, dtype = t.int64)
    train_lengths =  t.tensor(train_lengths, dtype = t.int64)
    y_train = t.tensor(y_train, dtype = t.int64)

    train_loader = data.DataLoader(data.TensorDataset(X_train,train_lengths, y_train),
                                batch_size = 100) #model expects training data to be batched a dataLoader

    #validation data
    X_val = t.tensor(X_val, dtype = t.int64)
    val_lengths =  t.tensor(val_lengths, dtype = t.int64)
    y_val = t.tensor(y_val, dtype = t.int64)

    validation_data = (X_val, val_lengths, y_val) # this model will expect validation data to be a tuple


    vocab_size = len(word_to_int)
    hidden_size = int(8*p + 1 )
    num_layers = 3
    epochs = 10
    loss_function = t.nn.CrossEntropyLoss(reduction = "mean")
    seq_length = len(X[0])

    lstm = model.LSTM(seq_length = seq_length, 
                hidden_size = hidden_size, 
                num_layers = num_layers, 
                vocab_size = vocab_size)

    optimizer = t.optim.Adam(params = lstm.parameters(), lr = .01)
    _, _, validation_accuracies, _ = lstm.Train(train_loader, 
                                                epochs, 
                                                loss_function, 
                                                optimizer, 
                                                validation_data)
    
    accuracies.append(validation_accuracies[-1])
    model_sizes.append(hidden_size)
    training_sizes.append(len(X_train))


a = accuracies
S = model_sizes
h = training_sizes

plt.scatter(S, h, alpha = np.array(a), color = 'red', s = 50)
plt.title("Validation Accuracy")
plt.xlabel("Training Size (in untis of Hidden Dimensions)")
plt.ylabel("hidden Size  Factor")

Β, Β_0 = np.polyfit(S, h, 1)

# adding the regression line to the scatter plot
plt.plot(S, Β*np.array(S) + Β_0, color = 'red')

plt.grid()