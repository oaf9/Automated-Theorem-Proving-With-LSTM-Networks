try:
    import matplotlib.pyplot as plt
    import model
    import process_data
    import torch as t
    import torch.utils.data  as data
    import numpy as np
except: 
    print("""You Do not have the required Modules to Run this Script.\n
          Required Modules: matplotlib, torch, numpy, huggingface_hub\n""")
    print("Run pip install huggingface_hub")
    exit(1)

"""
The ethos of this script is as follows:
We consider small training data sets, and iteratively increase
both the set of training data, and the size of the model.
This demonstrates a significant upward trend in performance with respect to both values
As such, this suggests that, with mroe data and larger models, high performance can be achieved. 
"""

#load the data from hugging face mode = 'w' means that we are tokenizing words rather than characters or sentences. 
proofs_dataset = process_data.LoadLogicData(mode = 'w') 

#format the proofs: essentially just mapping words to integers and then creating n-gram sequences
word_to_int, int_to_word, sequenced_proofs = process_data.generate_sequences(proofs_dataset)

#split data into input and label by setting label equal to next word.
#sequence length is the length of eqch sequence, this allows us to pack them during training. 
X, sequence_lengths,y = process_data.makeXy(sequenced_proofs)

percentages = np.arange(0.1, 1, .1)


accuracies_traindata = []
accuracies_size = []
model_sizes = []
training_sizes = []





for p in percentages:
    # for each percentage we train a model on a larger split of data.

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
    hidden_size = 8 #increase the size of the mdoel at each turn
    num_layers = 2
    epochs = 20
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
    
    accuracies_traindata.append(validation_accuracies[-1])
    training_sizes.append(len(X_train))







for i in range(2, 10):
    # for each percentage we train a model on a larger split of data.

    X_train, X_val, train_lengths, val_lengths, y_train, y_val = process_data.makeSplit(X,sequence_lengths, y, .9)

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
    hidden_size = i//2 #increase the size of the mdoel at each turn
    num_layers = 2
    epochs = 20
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
    
    accuracies_size.append(validation_accuracies[-1])
    model_sizes.append(i)

fig, axs = plt.subplots(1,2, figsize=(20, 6))

fig.suptitle("Performance Accuracy With Respect to Size of Data and Number of Paramaters.")
fig.subplots_adjust(hspace = .5, wspace=.2)

axs[0].scatter(model_sizes, accuracies_size, color = 'teal', label = "Train", s = 60)
axs[0].set_title("Accuracy and Size of Hidden Dimention")
axs[0].set_xlabel("Size")
axs[0].set_ylabel("Token Prediction Accuracy")
axs[0].grid()

axs[1].scatter(training_sizes, accuracies_traindata, color = 'darkred', label = "Train", s = 70)
axs[1].set_title("Accuracy and Size of Training Data")
axs[1].set_xlabel("Number of Samples")
axs[1].set_ylabel("Token Prediction Accuracy")
axs[1].grid()

plt.show()