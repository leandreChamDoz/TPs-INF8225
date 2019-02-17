import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X = digits.data

y = digits.target
y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1  # one  hot  target  or  shape NxK


def softmax(x):
    return np.exp(x - max(x)) / np.exp(x - max(x)).sum()


def get_accuracy(X, y, W):
    n_correctly_classified = 0
    n_samples = X.shape[0]
    for i in range(0, n_samples):
        arg_max_pred = np.argmax(softmax(W.dot(X[i])))
        arg_max_corr = np.argmax(y[i])
        if arg_max_pred == arg_max_corr:
            n_correctly_classified = n_correctly_classified + 1

    return n_correctly_classified / n_samples


def get_grads(y, y_pred, X):
    return np.matmul(y.reshape(y.shape[0], 1), X.reshape(1, X.shape[0])) - \
           np.matmul(y_pred.reshape(y_pred.shape[0], 1), X.reshape(1, X.shape[0]))


def get_loss(y, y_pred, l1=0, l2=0, W=None):
    if W is None:
        return np.mean((-1 * y * np.log(y_pred)))
    else:
        if type(np.mean((l1 * (W * W)))) is not np.float64:
            print(type(np.mean((l1 * (W * W)))))
        return np.mean(-1*y*np.log(y_pred)) + np.mean((l1 * (W * W))) + np.mean((l2 * (np.abs(W))))


def train(lr, minibatch_size, is_ela_net=False, l1=0, l2=0):
    if not is_ela_net:
        l1, l2 = 0, 0

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]))  # weights  of  shape KxL
    best_W = None
    best_accuracy = 0
    nb_epochs = 50

    losses = []
    accuracies = []

    if is_ela_net:
        min_val, max_val = np.min(X), np.max(X)
        X_new_data = np.asarray([np.append(i, np.random.uniform(min_val, max_val, 8)) for i in X])
        X_train, X_test, y_train, y_test = train_test_split(X_new_data, y_one_hot, test_size=0.3, random_state=42)
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        W = np.random.normal(0, 0.01, (len(np.unique(y)), X_new_data.shape[1]))  # weights  of  shape KxL

    for epoch in range(nb_epochs):
        loss_test = 0
        loss_val = 0
        loss_train = 0
        accuracy = 0
        for i in range(0, X_train.shape[0], minibatch_size):
            grads = []
            step = minibatch_size if (i + minibatch_size) < X_train.shape[0] else X_train.shape[0] - i

            for j in range(0, step):
                y_pred = softmax(W.dot(X_train[i + j]))
                grads.append(get_grads(y_train[i + j], y_pred, X_train[i + j]))

            grad_sum = np.zeros((len(grads[0]), len(grads[0][0])))
            for grad in grads:
                grad_sum = grad_sum + grad
            reg = minibatch_size/X_train.shape[0] * (-l1 * 2 * W - l2 * (W / np.sqrt(W ** 2)))
            W = W + lr * (grad_sum / len(grads) + reg)

        for i in range(0, X_train.shape[0]):
            loss_train += get_loss(y_train[i], softmax(W.dot(X_train[i])), l1, l2, W)
        loss_train = loss_train / (X_train.shape[0])

        for i in range(0, X_test.shape[0]):
            loss_test += get_loss(y_test[i], softmax(W.dot(X_test[i])), l1, l2, W)
        loss_test = loss_test / (X_test.shape[0])

        for i in range(0, X_validation.shape[0]):
            loss_val += get_loss(y_validation[i], softmax(W.dot(X_validation[i])), l1, l2, W)
        loss_val = loss_val / (X_validation.shape[0])

        l1 = loss_val * l1
        l2 = loss_val * l2

        losses.append([loss_val, loss_test, loss_train])  # compute  the  l o s s  on  the  train  set
        accuracy = get_accuracy(X_validation, y_validation, W)
        accuracies.append(accuracy)  # compute  the  accuracy  on  the  validation  set

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_W = W

    accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W)
    print(accuracy_on_unseen_data)  # 0.897506925208

    if is_ela_net:
        mean_random = np.mean(W[:, 64:73])
        mean_not_random = np.mean(W[:, 0:64])
        var_random = np.var(W[:, 64:73])
        var_not_random = np.var(W[:, 0:64])

        print('moyenne des poids aléatoires : ', mean_random)
        print('moyenne des poids non aléatoires : ', mean_not_random)
        print('variance des poids aléatoires : ', var_random)
        print('variance des poids non aléatoires : ', var_not_random)

    else:
        mean = np.mean(W[:, 0:64])
        var = np.var(W[:, 0:64])
        print('moyenne des poids non aléatoires : ', mean)
        print('variance des poids non aléatoires : ', var)

    fig = plt.figure()

    plt.plot([i[0] for i in losses], label='validation')
    plt.plot([i[1] for i in losses], label='test')
    plt.plot([i[2] for i in losses], label='train')

    plt.xlabel('Epoch')
    plt.xlim(0, nb_epochs)

    plt.ylabel('Average negative log loss')

    plt.title('Average negative log-loss with lr = ' + str(lr) + ' and mini-batch size = ' + str(minibatch_size))
    plt.legend(loc='best')

    plt.show()

    #fig.savefig('./Graphs/Graph' + str(lr) + '_' + str(minibatch_size) + '.png', format='png')


multiple_tests = False

if multiple_tests:
    lrs = [0.1, 0.01, 0.001]
    batch_sizes = [1, 20, 200, 1000]
    for i in lrs:
        for j in batch_sizes:
            train(i, j, True)

else:
    train(0.001, 1, False)
