import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np
from sklearn.model_selection import train_test_split
from nn import nn, io, preprocess

def main():
    # digits = sklearn.datasets.load_digits()
    # X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=.1, random_state=27)
    # net = nn.NeuralNetwork(
    #     nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'sigmoid'},
    #                {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
    #     lr=1e-1,
    #     seed=0,
    #     batch_size=600,
    #     epochs=1000,
    #     loss_function="mse")
    #
    # print(net._param_dict)
    # train_loss = net.fit(X_train, X_train, X_test, X_test)
    #
    # pred = net.predict(digits.data[0])
    # plt.imshow(pred.reshape(8,8), vmin=4)
    # plt.colorbar()
    # plt.show()

    net = nn.NeuralNetwork(
        nn_arch = [{'input_dim': 5, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=1e-3,
        seed=0,
        batch_size=1,
        epochs=10,
        loss_function="mse")
    print(net._param_dict)
# def main():
#     positives = io.read_text_file("./data/rap1-lieb-positives.txt")
#     negatives = io.read_fasta_file("./data/yeast-upstream-1k-negative.fa")
#     pos_length = len(positives[0])
#
#
#     negs_shortened = []
#     for seq in negatives:
#         negs = [seq[i * pos_length:(i + 1) * pos_length] for i in range(len(seq) // pos_length)]
#         negs_shortened.extend(negs)
#
#     neg_short_arr = np.array(negs_shortened)[np.random.randint(len(negs_shortened), size=10000)]
#     one_hot_pos = preprocess.one_hot_encode_seqs(positives)
#     one_hot_neg = preprocess.one_hot_encode_seqs(neg_short_arr)
#
#     pos_with_labels = np.hstack((one_hot_pos, np.ones((one_hot_pos.shape[0], 1))))
#     neg_with_labels = np.hstack((one_hot_neg, np.zeros((one_hot_neg.shape[0], 1))))
#
#     if len(pos_with_labels) > len(neg_with_labels):
#         boosted = neg_with_labels[np.random.randint(neg_with_labels.shape[0], size=len(pos_with_labels)), :]
#         dataset = np.concatenate([boosted, neg_with_labels], axis=0)
#     elif len(pos_with_labels) < len(neg_with_labels):
#         boosted = pos_with_labels[np.random.randint(pos_with_labels.shape[0], size=len(neg_with_labels)), :]
#         dataset = np.concatenate([boosted, neg_with_labels], axis=0)
#
#     X_train_tfc, X_test_tfc, y_train_tfc, y_test_tfc = train_test_split(dataset[:, :-1], dataset[:, -1][:, np.newaxis],
#                                                                         test_size=.1, random_state=27)
#
#     clf = nn.NeuralNetwork(
#         nn_arch=[{'input_dim': 68, 'output_dim': 30, 'activation': 'relu'},
#                  {'input_dim': 30, 'output_dim': 1, 'activation': 'sigmoid'}],
#         lr=1e-3,
#         seed=27,
#         batch_size=10,
#         epochs=10,
#         loss_function="bce")
#
#     train_losses_tfc, val_losses_tfc = clf.fit(X_train_tfc, y_train_tfc, X_test_tfc, y_test_tfc)

if __name__ == '__main__':
    main()