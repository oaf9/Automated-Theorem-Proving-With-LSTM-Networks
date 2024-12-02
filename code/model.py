import torch 
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first = True)
        
        #output hidden layer
        self.fc = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, X, sequence_lengths):
        """ forward pass through network"""

        X = self.embedding(X)

        X = pack_padded_sequence(X, sequence_lengths, 
                                 batch_first = True, 
                                 enforce_sorted = False)


        X, (H,C) = self.lstm(X)
        X, _ = pad_packed_sequence(X, batch_first = True)

        fc_out = self.fc(X)

        return F.log_softmax(fc_out, dim = -1)
    

    def predict(self, validation_data, loss_function):
        """
        This should be a test set with one batch
        mainly just calls forward but with no gradient updates:
        validation data is a tuple: (X_val, val_lengths, y_val)
        """

        X, l, y = validation_data

        predictions = self.forward(X, l)

        predictions = predictions[range(len(X)), l-1]
        predictions = predictions.view(-1, self.vocab_size)

        loss = loss_function(predictions, y)
        _, y_hat = torch.max(predictions, dim = 1)

        accuracy = (y_hat == y).sum().item()/len(y)

        return y_hat, accuracy, loss


    def Train(self, train_loader, epochs, 
              loss_function, optimizer, validation_data = None):

        validation_accuracies = []
        validation_losses = []
        training_accuracies = []
        training_losses = []

        for epoch in range(epochs): # for each epoch

            self.train() #put the model into training mode

            epoch_loss = 0
            correct_count = 0
            prediction_count = 0

            for index, data in enumerate(train_loader): #one pass over the training data

                X, sequence_lengths, y = data

                optimizer.zero_grad() # zero gradients to avoid blowup
                output = self.forward(X, sequence_lengths)

                output = output[range(len(X)), sequence_lengths-1]

                output = output.view(-1, self.vocab_size)

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



            #print metrics for the traning data

            training_loss = epoch_loss/len(train_loader)
            training_accuracy = correct_count/prediction_count
            training_losses.append(training_loss)
            training_accuracies.append(training_accuracy)

            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss/len(train_loader):.4f}')
            print(f'Epoch [{epoch+1}/{epochs}], Training Accuracy: {correct_count/prediction_count:.4f}')


            #after each epoch, we check the validation performance:

            if not validation_data == None:
                self.eval()
                with torch.no_grad():
                    _, validation_accuracy, validation_loss = self.predict(validation_data, loss_function)
                    validation_accuracies.append(validation_accuracy)
                    validation_losses.append(validation_loss.tolist())
                    
                    #print metrics for the validation data
                    print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {validation_loss:.4f}')
                    print(f'Epoch [{epoch+1}/{epochs}], Validation Accuracy: {validation_accuracy:.4f}')

            print('   ')

        self.eval()

        return training_accuracies, training_losses, validation_accuracies, validation_losses


    
    def generateToken(self, x,l):
        "predict the next token from a sequence"

        with torch.no_grad():
            #this will be the last item in the prediciton vector
            predicted_sequence = self.forward(x,l)
            predicted_sequence = predicted_sequence[:,-1,:]
            return torch.argmax(predicted_sequence, dim = 1)[0] #return argmax{log(p_i) w.r.t i}
        
    def generateProof(self, premises, l):

        with torch.no_grad():

            while True:
                
                next_token = self.generateToken(premises,l)
                premises[0][l] = next_token
                l += 1
                #break condition ... we won't won't predict anything longer than
                #in the training
                if l == len(premises[0]) or next_token == 1:
                    break
                
            return premises
