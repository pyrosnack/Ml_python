"""
nielsen_network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation. Based on the online book by Nielsen.
"""

import random
import numpy as np
from time import clock

################################### Misc #######################################
verbose = True
start_time = 0

def start_timer():
    global start_time
    start_time = clock()

def get_time():
    return clock() - start_time

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    sz = sigmoid(z)
    return sz * (1.0 - sz)

######################## Scoring ###############################################

def percentage_right(ypred, ytruth):
    res = sum(int(x==y) for (x, y) in zip(ypred, ytruth))
    return (res, str(res)) 

def precision(ypred, ytruth):
    product = float(np.dot(ypred, ytruth))
    if sum(ypred) > 0:
        res = product / sum(ypred)
    else:
        res = 0
    return 100 * res

def recall(ypred, ytruth):
    product = float(np.dot(ypred, ytruth))
    res = product / sum(ytruth)
    return 100 * res

def demog_score(ypred, ytruth):
    product = float(np.dot(ypred, ytruth))
    prec = precision(ypred, ytruth)
    rec = recall(ypred, ytruth)
    score = min(prec, rec)
    res_s = "%8.2f %8.2f %8.2f" % (prec, rec, score)
    if product == 0:
        res_s = res_s + " mais y a aucun positif"
    return (score, res_s)

################################## Main Object #################################

class Network(object):
    def __init__(self, sizes, cost_function="quadratic", dropout=True):
        cost_fun, delta = cost_functions[cost_function]
        self.cost_fun = cost_fun
        self.cost_fun_delta = delta
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.treshold = 0.5
        self.best_treshold = 0.5
        self.best_score = 0
        self.best_weights = self.weights
        self.best_biases = self.biases
        self.dropout = dropout
        self.neurons_keep = [list(range(s)) for s in sizes]
        for s_hidden in sizes[1:-1]:
            if s_hidden == 1:
                self.dropout = False
    
    def update_weights(self, delta, coeff):
        self.weights = [coeff * self.weights[i] + delta[i]
                        for i in range(len(self.weights))]

    def update_biases(self, delta):
        self.biases= [self.biases[i] + delta[i]
                      for i in range(len(self.biases))]
        
    def do_dropout(self):
        self.back_up_weights = self.weights[:]
        self.back_up_biases = self.biases[:]
        neurons_keep = [list(range(s)) for s in self.sizes]
        for i in range(len(neurons_keep) - 2):
            random.shuffle(neurons_keep[i+1])
            if len(neurons_keep[i+1]) > 1:
                neurons_keep[i+1] = neurons_keep[i+1][:len(neurons_keep[i+1])/2]
        self.neurons_keep = neurons_keep
        for i, w in enumerate(self.weights):
            new_w = []
            new_b = []
            for k in neurons_keep[i+1]:
                new_line = []
                for j in neurons_keep[i]:
                    new_line.append(w[k][j])
                new_w.append(new_line)
                new_b.append(self.biases[i][k])
            new_w = np.array(new_w)
            new_b = np.reshape(np.array(new_b), (len(new_b), 1))
            self.weights[i] = new_w
            self.biases[i] = new_b

    def restore_dropout(self):
        for i, w in enumerate(self.weights):
            for k, line in enumerate(w):
                initial_k = self.neurons_keep[i+1][k]
                self.back_up_biases[i][initial_k] = self.biases[i][k]
                for j, value in enumerate(line):
                    inital_j = self.neurons_keep[i][j]
                    self.back_up_weights[i][initial_k][inital_j] = value
        self.weights = self.back_up_weights
        self.biases = self.back_up_biases
                

def feed_forward(nn, a):
    res = a
    for b, w in zip(nn.biases, nn.weights):
        if nn.dropout:
            res = sigmoid(0.5 * np.dot(w, res) + b)
        else:
            res = sigmoid(np.dot(w, res) + b)
    return res

def update_batch(nn, batch, alpha, lmbda, len_training):
    if nn.dropout:
        nn.do_dropout()
    batch_size = len(batch)
    nabla_b = [np.zeros(b.shape) for b in nn.biases]
    nabla_w = [np.zeros(w.shape) for w in nn.weights]
    for x, y in batch:
        delta_b, delta_w = back_prop(nn, x, y)
        nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
        nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
    delta_weight = [(- alpha / batch_size) * nw for nw in nabla_w]
    delta_biases = [(- alpha / batch_size) * nb for nb in nabla_b]
    nn.update_weights(delta_weight, (1 - alpha * (lmbda / len_training)))
    nn.update_biases(delta_biases)
    if nn.dropout:
        nn.restore_dropout()

def apply_all_treshold(nn, test_set, treshold_possibles, treshold_last_values,
                       scoring_print):
    if len(treshold_last_values[0]) == 10:
        treshold_last_values = [lv[1:] for lv in treshold_last_values]
    for i, treshold in enumerate(treshold_possibles):
        res, res_string = evaluate_single(nn, test_set,
                                          score_function = scoring_print,
                                          treshold = treshold)
        treshold_last_values[i].append(res)
    tresh_total = [sum(lv) / len(lv) for lv in treshold_last_values]
    chosen_i = np.argmax(tresh_total)
    chosen_tresh = treshold_possibles[chosen_i]
    res_string = "                %4.2f          %8.2f" % (chosen_tresh, tresh_total[chosen_i])
    return chosen_tresh, treshold_last_values, res_string

def split_for_cva(training_set):
    res_training_set = []
    res_test_set = []
    for l in training_set:
        if random.randint(0, 3) == 0:
            res_test_set.append(l)
        else:
            res_training_set.append(l)
    return res_training_set, res_test_set

def sgd(nn, training_set, epochs, batch_size, alpha,
        lmbda = 0.0,
        test_set=None,
        scoring_print=percentage_right,
        treshold=0.5,
        find_best_treshold=True,
        time_limit=None):
    if not test_set:
        training_set, test_set = split_for_cva(training_set)
    n = len(training_set)
    if treshold:
        nn.treshold = treshold
    treshold_possibles = np.linspace(0.65, 0.85, 20)
    treshold_last_values = [[] for t in treshold_possibles]
    if time_limit:
        epochs = 100000
    for epoch in range(epochs):
        if time_limit and get_time() > time_limit * 0.95:
            print "time limit"
            break
        random.shuffle(training_set)
        batchs = [training_set[offset: offset + batch_size]
                  for offset in xrange(0, n, batch_size)]
        for batch in batchs:
            update_batch(nn, batch, alpha, lmbda, n)
        if nn.sizes[-1] == 1:
            test_result, test_r_string = evaluate_single(nn, test_set,
                    score_function=scoring_print,
                    treshold = nn.treshold)
        else:
            test_result, test_r_string = evaluate(nn, test_set)
        if test_result > nn.best_score:
            nn.best_biases = nn.biases[:]
            nn.best_weights = nn.weights[:]
            nn.best_treshold = nn.treshold
            nn.best_score = test_result
        elapsed_time = str(get_time())
        if verbose:
            print ("Epoch %4i: " % epoch) + test_r_string + " || " + \
                elapsed_time + "s"
        if nn.sizes[-1] == 1 and find_best_treshold:
            avt, tlv, s = apply_all_treshold(nn, test_set,
                                             treshold_possibles,
                                             treshold_last_values,
                                             scoring_print)
            nn.treshold = avt
            treshold_last_values = tlv
            if verbose:
                print s
    print "After training, on cva best score is %4.2f" % nn.best_score
    nn.biases = nn.best_biases[:]
    nn.weights = nn.best_weights[:]
    nn.treshold = nn.best_treshold

def back_prop(nn, x, y):
    nabla_b = [np.zeros(b.shape) for b in nn.biases]
    nabla_w = [np.zeros(w.shape) for w in nn.weights]
    activation = x
    activations = [x]
    zs = []

    for i in range(len(nn.biases)):
        w = nn.weights[i]
        b = nn.biases[i]
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    delta = nn.cost_fun_delta(zs[-1], activations[-1], y)
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].T)
    for l in xrange(2, nn.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(nn.weights[1 - l].T, delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-1 - l].T)
            
    return (nabla_b, nabla_w)

def predict(nn, test_set):
    results = [np.argmax(feed_forward(nn, x)) for x in test_set]
    return results

def proba_single(nn, test_set):
    results = [(feed_forward(nn, x)) for x in test_set]
    return results

def predict_single(nn, test_set, treshold=None):
    if not treshold:
        treshold = nn.treshold
    results = proba_single(nn, test_set)
    results = [ 1 if x > treshold else 0 for x in results]
    return results

def evaluate_single(nn, test_set, treshold=0.5, score_function=percentage_right):
    test_x = [x for (x,y) in test_set]
    test_y = [y for (x,y) in test_set]
    results = predict_single(nn, test_x, treshold=treshold)
    return score_function(results, test_y)

def evaluate(nn, test_set):
    results = [(np.argmax(feed_forward(nn, x)), y) for (x, y) in test_set]
    res = sum(int(x==y) for (x, y) in results)
    res_s = str(res) + (" / %i . " % len(test_set))
    return (res, res_s)

#################### Cost function and derivative ###############################

cost_functions = dict()

def quadratic(a, y):
    return 0.5 * np.linalg.norm(a - y)**2

def quadratic_delta(z, a, y):
    return (a - y) * sigmoid_prime(z)

def cross_entropy(a, y):
    return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

def cross_entropy_delta(z, a ,y):
    return a - y

cost_functions["quadratic"] = (quadratic, quadratic_delta)
cost_functions["cross_entropy"] = (cross_entropy, cross_entropy_delta)

