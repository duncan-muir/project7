import sklearn.datasets
from sklearn.model_selection import train_test_split
from nn import nn

def main():
    digits = sklearn.datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=.1, random_state=27)
    net = nn.NeuralNetwork(
        nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                   {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
        lr = 1e-3,
        seed = 27,
        batch_size=10,
        epochs=10,
        loss_function="mse")
    for key, val in net._param_dict.items():
        print(key,val.shape)
    net.fit(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()