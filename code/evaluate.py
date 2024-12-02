import matplotlib.pyplot as plt

def plotResults(num_epochs, training_accuracy, 
                validation_accuracy, training_loss, validation_loss):

    plt.rcParams.update({'font.size': 16})


    fig, axs = plt.subplots(1,2, figsize=(20, 6))

    fig.suptitle("Performance Charts")
    fig.subplots_adjust(hspace = .5, wspace=.2)

    axs[0].plot(range(1,num_epochs+1), training_accuracy, color = 'teal', label = "Train")
    axs[0].plot(range(1,num_epochs+1), validation_accuracy, color = 'darkred', label = "Validation")
    axs[0].set_title("Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Token Classification Accuracy ")
    axs[0].legend(loc = 'lower right')
    axs[0].grid()


    axs[1].plot(range(1,num_epochs+1), training_loss, color = 'teal', label = "Train")
    axs[1].plot(range(1,num_epochs+1), validation_loss, color = 'darkred', label = "Validation")
    axs[1].set_title("Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc = 'upper right')
    axs[1].grid()

    return fig, axs


def generate_proofs(lstm, test_data, test_sequence_lengths,  
                    int_to_word, save = False, path = None):
    import torch as t
    import numpy as np
    generated_proofs = []

    sample, length = test_data[0], test_sequence_lengths[0]
    length
    sample = t.tensor([sample], dtype = t.int64)
    length = t.tensor([length], dtype = t.int64)

    lstm.generateProof(sample, length)

    f = lambda int : int_to_word[int]
    generated_proofs = []

    for i in range(len(test_data)):

        sample, length = test_data[i], test_sequence_lengths[i]
        sample = t.tensor([sample], dtype = t.int64)
        length = t.tensor([length], dtype = t.int64)
        proof = lstm.generateProof(sample, length)[0].tolist()
        proof = [f(t) for t in proof]
        proof = ''.join(proof)
        generated_proofs.append(proof.replace("<pad>", ''))

    generated_proofs[0].replace("<pad>", '')
    print("Sample Output")
    print(" ")
    print(generated_proofs[0])

    if save:
        try: 
            np.savetxt(path + '.csv', np.array( ["Proof"] +generated_proofs), delimiter = ',', fmt='%s' )
        except:
            print(f"Unable to save file to path: {path}")

    return generated_proofs