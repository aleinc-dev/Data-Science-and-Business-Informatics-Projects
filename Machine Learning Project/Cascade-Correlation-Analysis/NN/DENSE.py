import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto
from NN import common

class LayerInfo(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return count
    LAYER_LAYER_NUMBER =             auto()
    LAYER_UNITS_NUMBER =             auto()
    LAYER_INPUT_NUMBER =             auto()
    LAYER_ACTIVATION_FUNCTION =      auto()
    LAYER_ACTIVATION_DERIVATIVE =    auto()
    LAYER_WEIGHTS =                  auto()
    LAYER_WEIGHTS_LAST_UPDATE =      auto()
    LAYER_WEIGHTS_CURRENT_UPDATE =   auto()
    LAYER_BIASES =                   auto()
    LAYER_BIASES_LAST_UPDATE =       auto()
    LAYER_BIASES_CURRENT_UPDATE =    auto()
    LAYER_INPUT =                    auto()
    LAYER_NET =                      auto()
    LAYER_OUTPUT =                   auto()
LAYER_INFO_SIZE = len(LayerInfo)

class DataCollector(Enum):
    NO_COLLECTING = 0
    TRAIN_ONLY = 1
    TRAIN_VAL = 2

class Optimizer(Enum):
    Egd = 0

class DENSE(object):
    #   Dense Multi layer perceptron con un numero variabile di layer nascosti
    #   Tutti i parametri necessari al funzionamento sono passati al momento della creazione
    #   per fare in modo che sia possibile utilizzare il modello in pipeline
    def __init__(self,
                 input_units,           # il numero di unità di ingresso che è essere uguale al numero di features del dataset
                 output_units,          # il numero di unità in uscita che è uguale al numero di previsioni del modello
                 hidden_layers = [(5, common.ActivationFunction.SIGMOID)], # lista di coppie (nodi del layer, funzione di attivazione del layer)
                 output_function = common.ActivationFunction.LINEAR,  # funzione di attivazione dello strato di uscita
                 gradient_descent_algorithm = Optimizer.Egd,
                 loss_function = common.LossFunction.MSE,
                 val_X=None,              # eventuale dataset di validazione (input)
                 val_y=None,              # eventuale dataset di validazione (output)
                 minibatch_size=None,          # è il numero di pattern che si devono processare prima di aggiornare i pesi
                 epochs=5,              # iterazioni richieste
                 learning_rate=5e-3,              # eta
                 learning_rate_decay_tau=0,
                 learning_rate_min_factor=100,
                 momentum=0,            # alpha
                 regularization=0,             # lambda
                 val_min_error=0,
                 val_patience=0,
                 val_min_improvement=None,
                 train_min_error=0,
                 train_patience=0,
                 train_min_improvement=None,
                 train_min_step=None, # delta
                 weights_amplitude = 1,# alcuni dataset (MONK) sono sensibili alla grandezza iniziale dei pesi, questo parametro serve a ridurre l'ampiezza dell'intervallo [-weights_amplitude/2,weights_amplitude/2]
                 score_function = None,
                 seed=None,
                 verbose = False,
                 log_weight_norm=False):# seed per la generazione dei pesi iniziali, None per random
        self.verbose = verbose
        self.log_weight_norm = log_weight_norm
        self.early_stopped = False
        self.rng = np.random.RandomState(seed)
        if input_units <= 0:
            raise ValueError("Number of input units must be positive")
        if output_units <= 0:
            raise ValueError("Number of output units must be positive")
        self.input_units = input_units
        self.output_units = output_units
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        self.eta = learning_rate
        self.learning_rate_max = learning_rate
        if learning_rate_decay_tau < 0:
            raise ValueError("Tau for decay must be non-negative")
        self.learning_rate_decay_tau = learning_rate_decay_tau
        if learning_rate_min_factor < 1:
            raise ValueError("Factor for min learning rate must be greater than or equal to 1")
        self.leanring_rate_min_factor = learning_rate_min_factor
        self.learning_rate_min = learning_rate / learning_rate_min_factor
        if gradient_descent_algorithm == Optimizer.Egd:
            self.gradient_descent_algorithm = self.easy_gradient_descend
        else:
            raise ValueError("Unsupported gradient descent algorithm")
        self.loss_function = common.loss_functions[loss_function]
        self.loss_function_name = loss_function
        self.loss_function_derivative = common.loss_functions_derivatives[loss_function]
        if momentum < 0:
            raise ValueError("Momentum must be non-negative")
        self.alpha = momentum
        if regularization < 0:
            raise ValueError("Regularization must be non-negative")
        self.lambda_ = regularization
        if epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        self.epochs = epochs
        self.output_function = output_function
        if minibatch_size is not None and minibatch_size <= 0:
            raise ValueError("Minibatch size must be positive")
        self.minibatch_size = minibatch_size
        self.score_function = score_function
        self.val_X = val_X
        self.val_y = val_y
        self.provided_val_set = val_X is not None and val_y is not None
        self.train_error = []
        self.train_score = []
        self.val_error = []
        self.val_score = []
        if self.provided_val_set:
            if val_X.shape[0] != val_y.shape[0]:
                raise ValueError("val_X and val_y must have the same number of rows")
            if val_X.shape[1] != self.input_units:
                raise ValueError("val_X must have {} columns, as many as the input units".format(self.input_units))
            if val_y.shape[1] != self.output_units:
                raise ValueError("val_y must have {} columns, as many as the output units".format(self.output_units))
        self.number_of_layers = len(hidden_layers) + 1
        self.layers_info = []
        self.early_stopper = common.EarlyStopping(self._save_best_model, val_min_error, val_patience, val_min_improvement, train_min_step, train_min_error, train_patience, train_min_improvement)
        # le informazioni di ciascuno strato della rete sono memorizzate in una lista
        # dove al  posto 0 mettiamo il numero di unità dello strato, al posto 1 il numero di input che lo strato riceve (eccetera)
        # le define (in stile C) LAYER_UNITS_NUMBER, LAYER_INPUT_NUMBER, ... servono per accedere ai valori
        # il numero di input che riceve il primo strato è uguale al numero di input
        # per gli strati successivi è uguale al numero di unità contenute nello strato precedente eccetera
        layer_inputs_number = input_units
        if weights_amplitude <= 0:
            raise ValueError("Weights amplitude must be positive")
        self.L2weights = []
        for i in range(self.number_of_layers):
            self.L2weights.append([])
        for layer_index, (hidden_units, hidden_function) in enumerate(hidden_layers):
            self.layers_info.append(
                self. layer_init(layer_index, hidden_units, layer_inputs_number,
                                 common.activation_functions[hidden_function], common.activation_functions_derivatives[hidden_function], weights_amplitude))
            layer_inputs_number = hidden_units
        self.layers_info.append(
            self. layer_init(len(hidden_layers),output_units, layer_inputs_number,
                             common.activation_functions[output_function], common.activation_functions_derivatives[output_function], weights_amplitude))

    def get_error_curve(self):
        return self.train_error, self.val_error

    def get_score_curve(self):
        return self.train_score, self.val_score
    
    def layer_init(self,layer_index,layer_units_number,layer_input_number,layer_activation_function, layer_derivative_function,weights_amplitude):
        layer_info = [None] * LAYER_INFO_SIZE
        # l'ordine con cui si aggiungono i campi alla lista deve essere coerente con i valori di LAYER_UNITS_NUMBER, LAYER_INPUT_NUMBER, ...
        layer_info[LayerInfo.LAYER_LAYER_NUMBER.value] = (layer_index)
        layer_info[LayerInfo.LAYER_UNITS_NUMBER.value] = (layer_units_number)
        layer_info[LayerInfo.LAYER_INPUT_NUMBER.value] = (layer_input_number)
        layer_info[LayerInfo.LAYER_ACTIVATION_FUNCTION.value] = (layer_activation_function)
        layer_info[LayerInfo.LAYER_ACTIVATION_DERIVATIVE.value] = (layer_derivative_function)
        # i pesi delle unità di uno strato vengono memorizzati in una matrice,
        # numero di righe uguale al numero di input dello strato e
        # numero di colonne uguale al numero di unità dello strato
        layer_info[LayerInfo.LAYER_WEIGHTS.value] = ((self.rng.rand(layer_input_number, layer_units_number) - 0.5) * weights_amplitude)
        layer_info[LayerInfo.LAYER_WEIGHTS_LAST_UPDATE.value] = (np.zeros((layer_input_number, layer_units_number)))
        layer_info[LayerInfo.LAYER_WEIGHTS_CURRENT_UPDATE.value] = (np.zeros((layer_input_number, layer_units_number)))
        # i bias delle unità di uno strato vengono memorizati in una matrice di una riga
        # con un numero di colonne uguale al numero di unità dello strato
        layer_info[LayerInfo.LAYER_BIASES.value] = ((self.rng.rand(1, layer_units_number) - 0.5) * weights_amplitude)
        layer_info[LayerInfo.LAYER_BIASES_LAST_UPDATE.value] = (np.zeros((1, layer_units_number)))
        layer_info[LayerInfo.LAYER_BIASES_CURRENT_UPDATE.value] = (np.zeros((1, layer_units_number)))
        layer_info[LayerInfo.LAYER_INPUT.value] = (np.empty((1,1)))
        layer_info[LayerInfo.LAYER_NET.value] = (np.empty((1,1)))
        layer_info[LayerInfo.LAYER_OUTPUT.value] = (np.empty((1,1)))
        return(layer_info)
    
    def forward(self, X):
        # durante la fase di training, la funzione forward viene chiamata passando come parametro il dataset che si vuole processare
        # in questa fase non servono i valori di target che saranno invece usati nella fase di backward
        # per ogni layer si moltipliacno gli input per i pesi, si aggiunge il bias e si applica la sunzione di attivazione
        for layer_index in range(self.number_of_layers):
            self.layers_info[layer_index][LayerInfo.LAYER_INPUT.value] = X
            layer_weights = self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS.value]
            layer_biases = self.layers_info[layer_index][LayerInfo.LAYER_BIASES.value]
            intermediate_y = X @ layer_weights + layer_biases
            self.layers_info[layer_index][LayerInfo.LAYER_NET.value] = intermediate_y
            activated_y = self.layers_info[layer_index][LayerInfo.LAYER_ACTIVATION_FUNCTION.value](intermediate_y)
            # l'output di questo layer diventa l'input per il layer successivo
            X = activated_y
            self.layers_info[layer_index][LayerInfo.LAYER_OUTPUT.value] = X
        # l'array activated_y, a questo punto, è esattamente l'output del modello
        return

    def _intpredict(self, X):
        for layer_index in range(self.number_of_layers):
            layer_weights = self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS.value]
            layer_biases = self.layers_info[layer_index][LayerInfo.LAYER_BIASES.value]
            intermediate_y = X @ layer_weights + layer_biases
            activated_y = self.layers_info[layer_index][LayerInfo.LAYER_ACTIVATION_FUNCTION.value](intermediate_y)
            X = activated_y
        return activated_y

    def _log_error_and_score(self, X, y):
        activated_y = self._intpredict(X)
        self.train_error.append(self.loss_function(activated_y, y))
        if self.score_function is not None:
            self.train_score.append(self.score_function(activated_y, y))
        if self.provided_val_set:
            activated_y = self._intpredict(self.val_X)
            self.val_error.append(self.loss_function(activated_y, self.val_y))
            if self.score_function is not None:
                self.val_score.append(self.score_function(activated_y, self.val_y))

    def predict(self, X):
        if X.shape[1] != self.input_units:
            raise ValueError("X must have {} columns".format(self.input_units))
        return self._intpredict(X)

    def backward(self, batch_y):
        # output layer
        output = self.layers_info[self.number_of_layers - 1][LayerInfo.LAYER_OUTPUT.value]
        net = self.layers_info[self.number_of_layers - 1][LayerInfo.LAYER_NET.value]
        input = self.layers_info[self.number_of_layers - 1][LayerInfo.LAYER_INPUT.value]
        small_delta = -self.loss_function_derivative(output, batch_y) * self.layers_info[self.number_of_layers - 1][LayerInfo.LAYER_ACTIVATION_DERIVATIVE.value](net)
        self.layers_info[self.number_of_layers - 1][LayerInfo.LAYER_WEIGHTS_CURRENT_UPDATE.value] = input.T @ small_delta
        self.layers_info[self.number_of_layers - 1][LayerInfo.LAYER_BIASES_CURRENT_UPDATE.value] = small_delta.sum(axis=0)
        # hidden layers
        for layer_index in range(self.number_of_layers - 2, -1, -1):
            output = self.layers_info[layer_index][LayerInfo.LAYER_OUTPUT.value]
            net = self.layers_info[layer_index][LayerInfo.LAYER_NET.value]
            input = self.layers_info[layer_index][LayerInfo.LAYER_INPUT.value]
            weights = self.layers_info[layer_index + 1][LayerInfo.LAYER_WEIGHTS.value]
            small_delta = (small_delta @ weights.T) * self.layers_info[layer_index][LayerInfo.LAYER_ACTIVATION_DERIVATIVE.value](net)
            self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS_CURRENT_UPDATE.value] = input.T @ small_delta
            self.layers_info[layer_index][LayerInfo.LAYER_BIASES_CURRENT_UPDATE.value] = small_delta.sum(axis=0)
        return

    def easy_gradient_descend(self,layer_index,epoch,num_pattern):
        weights_last_update = self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS_LAST_UPDATE.value]
        weights_current_update = self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS_CURRENT_UPDATE.value]

        weights_update = weights_last_update * self.alpha + weights_current_update * self.eta / num_pattern
        return weights_update

    def update_biases(self,layer_index,num_pattern):
        biases_last_update = self.layers_info[layer_index][LayerInfo.LAYER_BIASES_LAST_UPDATE.value]
        biases_current_update = self.layers_info[layer_index][LayerInfo.LAYER_BIASES_CURRENT_UPDATE.value]
        biases_values = self.layers_info[layer_index][LayerInfo.LAYER_BIASES.value]

        biases_last_update = biases_last_update * self.alpha + biases_current_update * self.eta / num_pattern
        biases_new_values = biases_values + biases_last_update

        self.layers_info[layer_index][LayerInfo.LAYER_BIASES.value] = biases_new_values
        self.layers_info[layer_index][LayerInfo.LAYER_BIASES_LAST_UPDATE.value] = biases_last_update
        return

    def update(self,num_pattern,batch_size,epoch):
        for layer_index in range(self.number_of_layers):
            # prelevo i dati che mi servono dalla lista del layer
            weights_values = self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS.value]

            # calcolo i nuovi valori a seconda dell'aògoritmo selezionato
            weights_update = self.gradient_descent_algorithm(layer_index,epoch,num_pattern)

            # applico la regolarizzazione e aggiorno i valori
            final_delta = weights_update - weights_values * self.lambda_ * batch_size / num_pattern
            self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS.value] = weights_values + final_delta
            self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS_LAST_UPDATE.value] = final_delta

            # aggiorno i biases
            self.update_biases(layer_index, num_pattern)
        return
    
    def _get_largest_delta(self):
        max_delta = 0
        for layer_index in range(self.number_of_layers):
            weights_update = self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS_LAST_UPDATE.value]
            max_delta = max(max_delta, np.max(np.abs(weights_update)))
        return max_delta

    def check_early_stop_conditions(self, epoch, delta):
        # check first to save best weights
        if self.early_stopper.check_convergence(self.val_error[-1] if self.provided_val_set else None, self.train_error[-1], delta):
            if self.verbose:
                self.early_stopper.print_convergence_reason()
            self.early_stopped = True
            return True
        if epoch >= self.epochs:
            if self.verbose:
                print("Reached max number of epochs")
            return True
        return False
    
    def collect_weight_data(self):
        for i in range(self.number_of_layers):
            self.L2weights[i].append(np.linalg.norm(self.layers_info[i][LayerInfo.LAYER_WEIGHTS.value]))
    
    def triggered_early_stopping(self):
        return self.early_stopped

    def _save_best_model(self):
        self.best_model = []
        self.best_model_bias = []
        for layer_index in range(self.number_of_layers):
            self.best_model.append(self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS.value].copy())
            self.best_model_bias.append(self.layers_info[layer_index][LayerInfo.LAYER_BIASES.value].copy())

    def _restore_best_model(self):
        for layer_index in range(self.number_of_layers):
            self.layers_info[layer_index][LayerInfo.LAYER_WEIGHTS.value] = self.best_model[layer_index]
            self.layers_info[layer_index][LayerInfo.LAYER_BIASES.value] = self.best_model_bias[layer_index]
        self.best_model = None
        self.best_model_bias = None

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[1] != self.input_units:
            raise ValueError("X must have {} columns".format(self.input_units))
        if y.shape[1] != self.output_units:
            raise ValueError("y must have {} columns".format(self.output_units))
        self.early_stopped = False
        self.eta = self.learning_rate_max
        minibatch_size = self.minibatch_size if self.minibatch_size is not None else X.shape[0]
        self.X, self.y = X, y
        size_of_X = X.shape[0]
        ind = np.arange(size_of_X)
        self.rng.shuffle(ind)
        XX = X[ind]
        yy = y[ind]
        self._log_error_and_score(X, y)
        epoch = 0
        delta = np.inf
        if self.log_weight_norm:
            self.collect_weight_data()
        if self.provided_val_set:
            self._save_best_model()
        while not self.check_early_stop_conditions(epoch, delta):
            delta = 0
            for i in range(0, size_of_X, minibatch_size):
                batch_X = XX[i:i+minibatch_size]
                batch_y = yy[i:i+minibatch_size]
                self.forward(batch_X)
                self.backward(batch_y)
                self.update(X.shape[0], batch_X.shape[0],epoch)
                delta = max(delta, self._get_largest_delta())
            self._log_error_and_score(X, y)
            epoch += 1
            alpha = 1
            if self.learning_rate_decay_tau != 0:
                alpha = min(1, epoch / self.learning_rate_decay_tau)
            self.eta = self.learning_rate_max * alpha + self.learning_rate_min * (1 - alpha)
            if self.log_weight_norm:
                self.collect_weight_data()
        if self.provided_val_set:
            self._restore_best_model()
        return

