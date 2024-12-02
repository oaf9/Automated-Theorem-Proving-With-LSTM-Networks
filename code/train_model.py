import torch as t
import torch.utils.data  as data
import matplotlib.pyplot as plt

import model
import evaluate
import process_data



#load the data from hugging face mode = 'w' means that we are tokenizing words rather than characters or sentences. 
proofs_dataset = process_data.LoadLogicData(mode = 'w') 

#format the proofs: essentially just mapping words to integers and then creating n-gram sequences
word_to_int, int_to_word, sequenced_proofs = process_data.generate_sequences(proofs_dataset)

#split data into input and label by setting label equal to next word.
#sequence length is the length of eqch sequence, this allows us to pack them during training. 
X, sequence_lengths,y = process_data.makeXy(sequenced_proofs)

X_train, X_val, train_lengths, val_lengths, y_train, y_val = process_data.makeSplit(X,sequence_lengths, y, .95)


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

#set Your params as desired here


vocab_size = len(word_to_int)
hidden_size = 8
num_layers = 3
epochs = 40
loss_function = t.nn.CrossEntropyLoss(reduction = "mean")
seq_length = len(X[0])


lstm = model.LSTM(seq_length = seq_length, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            vocab_size = vocab_size)

optimizer = t.optim.Adam(params = lstm.parameters(), lr = .01)
training_accuracies, training_losses, validation_accuracies, validation_losses = lstm.Train(train_loader, 
                                                                                            epochs, 
                                                                                            loss_function, 
                                                                                            optimizer, 
                                                                                            validation_data)


fig, axs = evaluate.plotResults(epochs, training_accuracies, 
                                validation_accuracies, 
                                training_losses, validation_losses)
plt.show()


#loads some unseen premises from hugging face
test_data = process_data.load_test_data()
test_sequence_lengths, test_data = process_data.process_test_data(test_data, word_to_int)
test_data = process_data.pad(test_data, 0, seq_length)

#If you want to save the proofs to your folder, you can specify a path and set save = true in generate_proofs
#generated proofs returns a list of coleted proofs, it will also print the first value as a sample

path = ""
generated_proofs = evaluate.generate_proofs(lstm, test_data, test_sequence_lengths, int_to_word, save = False, path = path)