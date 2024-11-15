import torch 
from torch import nn
from torch.nn import functional as F

class LSTM(nn.Module):
    #constructor that inherits from nn.Module
    def __init__(self, seq_length,  hidden_size, num_layers,vocab_size):

        super(LSTM, self).__init__()
        #should probably initilize the hidden states
        self.seq_length = seq_length
        self.hidden_dim = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        #we need to embed the words, a rule of thumb is that the 
        # embedding has the fourth root of the size of the vocabulary
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        #initilize an lstm layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        
        #output hidden layer
        self.fc = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, previous_words):
        """ forward pass through network"""
        embeddings = self.embedding(previous_words)
        output, (H,C) = self.lstm(embeddings)

        fc_out = self.fc(output)


        return F.log_softmax(fc_out, dim = -1)


    def train(self, train_loader, epochs, 
              loss_function, optimizer):

        for epoch in range(epochs): # for each epoch

            epoch_loss = 0
            correct_count = 0
            prediction_count = 0

            for index, data in enumerate(train_loader): #one pass over the training data

                X,y = data

                optimizer.zero_grad() # zero gradients to avoid blowup
                output = self.forward(X)

                # forward has shape[B,L,V] and targets has shape [B,L] which aren't the dimensions that loss fucntions except.
                # we can flatten the arrays to get something typical: [B*L, V] and [B*L]
                output = output.view(-1, self.vocab_size)
                y = y.view(-1)

                loss = loss_function(output, y)

                loss.backward()
                #gradient clipping helps avoid blowup, which was a problem with training
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=4)  
                optimizer.step()
   
                #update metrics
                epoch_loss += loss.item()
                _, y_hat = torch.max(output, dim = 1)
                correct_count += (y_hat == y).sum().item()
                prediction_count += y.size(0)


            #print metrics
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{epochs}], Accuracy: {correct_count/prediction_count:.4f}')
            print('   ')


