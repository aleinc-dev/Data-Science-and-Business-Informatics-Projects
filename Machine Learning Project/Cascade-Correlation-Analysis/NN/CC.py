
import numpy as np
from NN import common

# Class of a Neural Network obtained through Cascade Correlation
class CC:
    # Constructor
    def __init__(self, 
                 input_units, # Number of input features
                 output_units, # Number of output features
                 seed=None, # Seed for random number generator, None for random seed
                 output_activation_function=common.ActivationFunction.LINEAR, # Activation function of the output layer
                 learning_rate_decay_tau=0, # Tau for exponential decay of BOTH learning rates, 0 to disable
                 learning_rate_min_factor = 100, # fraction of the original leaning rates reached in tau epochs
                 learning_rate=0.1, # Eta
                 learning_rate_hidden=0.1, # Eta for correlation maximization
                 momentum=0, # Alpha
                 momentum_hidden=0, # Alpha for correlation maximization
                 regularization=0, # Lambda
                 regularization_hidden=0, # Lambda for correlation maximization
                 max_hidden_nodes=10, # Maximum number of hidden nodes, affects maximum number of iterations of the cycle train output add hidden node
                 max_intra_step_iterations=100, # Maximum number of iterations of a step (train output or add hidden node)
                 rescaling=False, # Min-Max scaling of the input data and the output of the hidden nodes
                 rescaling_min=-1, # Min value of the rescaling
                 rescaling_max=1, # Max value of the rescaling
                 hidden_unit_activation_choices=[common.ActivationFunction.SIGMOID, common.ActivationFunction.RELU, common.ActivationFunction.LINEAR], # Activation functions to choose from when adding a hidden unit
                 hidden_unit_pool_size=1, # Number of different hidden unit per activation function to test when adding a hidden unit
                 minibatch_size=None, # Size of the minibatch, None to use batch
                 loss_function=common.LossFunction.MEE, # Loss function
                 # Stopping criterion (3 different sets: outer, inner output training, inner hidden addition)
                 # General stopping criterion
                 val_X=None, # Validation set, used for early stopping and scores
                 val_y=None, # Validation set target, used for early stopping and scores
                 val_min_error=0, # Stop when the validation error is below this threshold
                 train_min_error=0, # Stop when the training error is below this threshold
                 # Outer cycle stopping criterion
                 nodewise_val_patience=0, # Like patience, but considering improvement across multiple additions of hidden nodes over validation set
                 nodewise_val_min_improvement=None, # Stop when the improvement is below this threshold(considering patience), will be used only if a validation set is provided
                 nodewise_train_patience=0, # Like patience, but considering improvement across multiple additions of hidden nodes over training set
                 nodewise_train_min_improvement=None, # Stop when the improvement of training error is below this threshold(considering patience)
                 # Stopping for output layer training phases
                 out_val_patience=0, 
                 out_val_min_improvement=None, # Stop when the improvement is below this threshold(considering patience), will be used only if a validation set is provided
                 out_train_min_step=None, # Stop when the update step is below this threshold(considering patience)
                 out_train_patience=0,
                 out_train_min_improvement=None,
                 # Stopping for hidden node addition phases
                 hid_min_step=None, # Stop when the update step is below this threshold
                 hid_patience=0,
                 hid_min_improvement=None,
                 weight_amplitude=1, # Amplitude of the initial weights [-weight_amplitude/2, weight_amplitude/2]
                 add_internal_error=True, # Add error informations also after each iteration of output training
                 output_weigths_reset=False, # Reset output weights after each hidden node addition
                 score_function=None, # Function to compute the score of the model(in addition to logging the loss), will receive output of output layer and target as parameters(reasoning nodewise)
                 verbose=False, # Print additional information
                 function_choice_scale=None # Scale the output of the pool before computing the correlation during choice of best node, if not set will be set to True if there are more than one activation function in the pool
                 ):
        # Scaling of output nodes during choice of best node
        self.function_choice_scale = function_choice_scale
        if function_choice_scale is None:
            self.function_choice_scale = len(hidden_unit_activation_choices) > 1
        # Verbose
        self.verbose = verbose
        # Early stopped
        self.early_stopped = False
        # Random number generator
        self.rng = np.random.RandomState(seed)
        # Input size
        if input_units < 1:
            raise ValueError("Input size must be at least 1")
        self.input_size = input_units
        # Output size
        if output_units < 1:
            raise ValueError("Output size must be at least 1")
        self.output_size = output_units
        # Output layer activation function
        self.fK = common.activation_functions[output_activation_function]
        self.fK_derivative = common.activation_functions_derivatives[output_activation_function]
        # Hidden layer activation functions (list of ActivationFunction enum values)
        self.fJ = [] # No hidden layer yet
        # Hidden layer weights
        self.WJ = [] # No hidden layer yet
        # Output layer weights (k x (j+i+1) matrix)
        # Initialize the first column with the bias (=1), the rest with random values
        if weight_amplitude <= 0:
            raise ValueError("Weight amplitude must be positive")
        self.weight_amplitude = weight_amplitude
        self.WK = np.empty((output_units, input_units + 1))
        self.WK = (self.rng.rand(output_units, input_units + 1) - 0.5) * weight_amplitude
        #--------------------
        # Parameters
        self.weight_reset = output_weigths_reset
        if learning_rate <= 0 or learning_rate_hidden <= 0:
            raise ValueError("Learning rate must be positive")
        self.learning_rate = learning_rate
        self.learning_rate_hidden = learning_rate_hidden
        self.learning_rate_max = learning_rate
        self.learning_rate_hidden_max = learning_rate_hidden
        if learning_rate_decay_tau < 0:
            raise ValueError("Tau for decay must be non-negative")
        self.learning_rate_decay_tau = learning_rate_decay_tau
        if learning_rate_min_factor < 1:
            raise ValueError("Factor for minimum learning rate must be at least 1")
        self.learning_rate_min_factor = learning_rate_min_factor
        self.learning_rate_min = learning_rate / learning_rate_min_factor
        self.learning_rate_hidden_min = learning_rate_hidden / learning_rate_min_factor
        if momentum < 0 or momentum_hidden < 0:
            raise ValueError("Momentum must be non-negative")
        self.momentum = momentum
        self.momentum_hidden = momentum_hidden
        if regularization < 0 or regularization_hidden < 0:
            raise ValueError("Regularization must be non-negative")
        self.regularization = regularization
        self.regularization_hidden = regularization_hidden
        if max_hidden_nodes < 0:
            raise ValueError("Maximum number of hidden nodes must be non negative")
        self.max_hidden_nodes = max_hidden_nodes
        if max_intra_step_iterations < 1: # Enforces to always have a stopping criterion that will be met
            raise ValueError("Maximum number of iterations of a step must be at least 1")
        self.max_intra_step_iterations = max_intra_step_iterations
        self.rescaling = rescaling
        if rescaling_min >= rescaling_max:
            raise ValueError("Min value of the rescaling must be less than max value")
        self.rescaling_min = rescaling_min
        self.rescaling_max = rescaling_max
        # Hidden unit activation functions
        self.hidden_unit_activation_choices = hidden_unit_activation_choices
        # Number of hidden units per activation function
        if hidden_unit_pool_size < 1:
            raise ValueError("Number of hidden units per activation function must be at least 1")
        self.hidden_unit_pool_size = hidden_unit_pool_size
        # Minibatch size
        if minibatch_size is not None and minibatch_size < 1:
            raise ValueError("Minibatch size must be at least 1 or None to use batch")
        self.minibatch_size = minibatch_size
        # Loss function
        self.loss_function = common.loss_functions[loss_function]
        self.loss_function_derivative = common.loss_functions_derivatives[loss_function]
        #--------------------
        # Stopping criterion
        self.outer_stop = common.EarlyStopping(self._save_best_state_outer, 
                                               val_min_error=val_min_error, 
                                               val_patience=nodewise_val_patience,
                                               val_min_improvement=nodewise_val_min_improvement,
                                               train_min_error=train_min_error,
                                               train_patience=nodewise_train_patience,
                                               train_min_improvement=nodewise_train_min_improvement)
        self.inner_output_stop = common.EarlyStopping(self._save_best_state_inner,
                                                      val_min_error=val_min_error,
                                                      val_patience=out_val_patience,
                                                      val_min_improvement=out_val_min_improvement,
                                                      delta_thresh=out_train_min_step,
                                                      train_min_error=train_min_error,
                                                      train_patience=out_train_patience,
                                                      train_min_improvement=out_train_min_improvement)
        self.inner_hidden_stop = common.EarlyStopping(None, # to simulate maximization, in calls the errors are negated
                                                      delta_thresh=hid_min_step,
                                                      train_patience=hid_patience,
                                                      train_min_improvement=hid_min_improvement,
                                                      train_min_error=-np.inf)# since non normalized correlation is unbounded
        #--------------------
        # error over time
        self.tr_error = []
        self.val_error = []
        self.score_function = score_function
        self.tr_score = []
        self.val_score = []
        self.add_internal_error = add_internal_error
        self.tr_internal_error = []
        self.val_internal_error = []
        self.tr_internal_hid_error = []
        # Check existence of validation set
        self.provided_val_set = val_X is not None and val_y is not None
        # Check consistency of validation set
        if self.provided_val_set:
            self.val_X, self.val_y = self.setup_dataset(val_X, val_y)
        else:
            self.val_X = None
            self.val_y = None


    def _rescale_columns(self, X):
        # avoid division by zero
        den = np.max(X, axis=0) - np.min(X, axis=0)
        num = X - np.min(X, axis=0)
        den[den == 0] = 1
        return num / den * (self.rescaling_max - self.rescaling_min) + self.rescaling_min

    def setup_dataset(self, X, y):
        # Check input size
        if X.shape[1] != self.input_size:
            raise ValueError("Input size is not correct")
        # Check output size
        if y.shape[1] != self.output_size:
            raise ValueError("Output size is not correct")
        # Check consistency of num of labels wrt num of observations
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of observations in input and output is not consistent")
        if self.rescaling:
            # Rescale input
            X = self._rescale_columns(X)
        # Add bias to input
        X = np.c_[np.ones(X.shape[0]), X]
        return X, y

    def get_delta(self, X_enriched, net, out, target, num_pattern):
        temp = self.loss_function_derivative(out, target) / num_pattern * self.fK_derivative(net)# divisione per num_pattern per eta tweaking
        return temp.T @ X_enriched

    def get_error_curve(self):
        return self.tr_error, self.val_error
    
    def get_internal_error_curve(self):
        return self.tr_internal_error, self.val_internal_error, self.tr_internal_hid_error

    def get_score_curve(self):
        return self.tr_score, self.val_score

    def hidden_nodes_forward(self, X):
        in_k = X
        # Compute the output of the hidden nodes, prepare input to the output layer)
        # For each hidden node
        for weitght_j, fun_j in zip(self.WJ, self.fJ):
            # Compute the output of that node and "enrich" the data with the additional "feature" column
            to_add = common.activation_functions[fun_j](in_k @ weitght_j)
            if self.rescaling:
                # Rescale output of hidden node
                to_add = self._rescale_columns(to_add[:, np.newaxis])
            in_k = np.c_[in_k, to_add]
        return in_k

    def output_layer_forward(self, X):
        # Compute the output of the output layer
        return self.fK(X @ self.WK.T)
    
    def _get_output_error(self, X_enriched, target):
        # Compute output of output layer
        out = self.output_layer_forward(X_enriched)
        # Compute the error
        return self.loss_function(out, target)

    def predict(self, X):
        if X.shape[1] != self.input_size:
            raise ValueError("Input size is not correct")
        if self.rescaling:
            # Rescale input
            X = self._rescale_columns(X)
        # Compute the output of the network
        return self.output_layer_forward(self.hidden_nodes_forward(np.c_[np.ones(X.shape[0]), X]))
    
    # reduces space occupancy removing temporary data (does not affect the model performance)
    def clean(self):
        self.best_WK = None
        self.best_inner_WK = None
        if self.provided_val_set:
            # Keep only bias and input
            self.val_X = self.val_X[:, :self.input_size + 1]

    def _log_error_and_score(self, X_enriched, y):
        # Compute the output of the output layer
        output = self.output_layer_forward(X_enriched)
        # Compute the error
        tr_error = self.loss_function(output, y)
        # Add the error to the list
        self.tr_error.append(tr_error)
        if self.provided_val_set:
            # Compute the output of the output layer
            val_output = self.output_layer_forward(self.val_X)
            # Compute the error
            val_error = self.loss_function(val_output, self.val_y)
            # Add the error to the list
            self.val_error.append(val_error)
        if self.score_function is not None:
            # Compute the score
            tr_score = self.score_function(output, y)
            # Add the score to the list
            self.tr_score.append(tr_score)
            if self.provided_val_set:
                # Compute the score
                val_score = self.score_function(val_output, self.val_y)
                # Add the score to the list
                self.val_score.append(val_score)

    def triggered_early_stopping(self):
        return self.early_stopped

    # Returns True iff a stopping criterion for the whole training process is met and print the reason
    def outer_stop_check(self):
        # check first to save best weights
        # check if the outer stopping criterion is met
        if self.outer_stop.check_convergence(self.val_error[-1] if self.provided_val_set else None, self.tr_error[-1], None):
            if self.verbose:
                print("Nodewise epochs early stop")
                self.outer_stop.print_convergence_reason()
            self.early_stopped = True
            return True
        # check if the maximum number of hidden nodes has been reached
        if len(self.WJ) >= self.max_hidden_nodes:
            if self.verbose:
                print("Maximum number of hidden nodes reached")
            return True
        return False
    
    def inner_output_stop_check(self, iteration, val_error, train_error, delta):
        # check first to save best weights
        # check if the inner stopping criterion is met
        return self.inner_output_stop.check_convergence(val_error, train_error, delta) or iteration >= self.max_intra_step_iterations

    def inner_hidden_stop_check(self, iteration, mean_correlation, delta):
        # check if the inner stopping criterion is met
        return iteration >= self.max_intra_step_iterations or self.inner_hidden_stop.check_train_convergence(-mean_correlation, delta)

    def output_train(self, X_enriched, y, minibatch):
        iteration = 0
        max_delta = np.inf # maximum delta over the whole epoch to be used as stopping criterion
        previous_delta = 0 # previous step size, used for momentum
        self.learning_rate = self.learning_rate_max
        train_error_series = []
        val_error_series = []
        # Compute output error
        error = self._get_output_error(X_enriched, y)
        # Add the error to the list
        train_error_series.append(error)
        if self.provided_val_set:
            # Compute output error
            val_error = self._get_output_error(self.val_X, self.val_y)
            # Add the error to the list
            val_error_series.append(val_error)
            # Save best weigths
            self._save_best_state_inner()
        while (not self.inner_output_stop_check(iteration, val_error if self.provided_val_set else None, error, max_delta)):
            max_delta = 0
            # Shuffle the data
            permutation =  self.rng.permutation(X_enriched.shape[0])
            X_enriched = X_enriched[permutation]
            y = y[permutation]
            for i in range(0, X_enriched.shape[0], minibatch):
                # Compute output net
                out_net = X_enriched[i:i+minibatch] @ self.WK.T
                # Activation of the output layer
                out_act = self.fK(out_net)
                # Compute the delta
                delta = self.momentum * previous_delta - self.learning_rate * self.get_delta(X_enriched[i:i+minibatch], out_net, out_act, y[i:i+minibatch], X_enriched.shape[0])
                weight_decay = self.regularization * self.WK[:, 1:] * out_net.shape[0] / X_enriched.shape[0] # out_net.shape[0] is the effective number of patterns in the minibatch
                delta[:, 1:] = delta[:, 1:] - weight_decay
                # Update the weights
                self.WK += delta
                # Update previous delta
                previous_delta = delta
                # Update max delta
                if self.inner_output_stop.is_delta_thresh_set():
                    max_delta = max(max_delta, np.max(np.abs(delta)))
            # Compute output error
            error = self._get_output_error(X_enriched, y)
            # Add the error to the list
            train_error_series.append(error)
            if self.provided_val_set:
                # Compute output error
                val_error = self._get_output_error(self.val_X, self.val_y)
                # Add the error to the list
                val_error_series.append(val_error)
            # tweak learning rate
            iteration += 1
            if self.learning_rate_decay_tau > 0:
                alpha = min(1, iteration / self.learning_rate_decay_tau)
                self.learning_rate = self.learning_rate_max * alpha + self.learning_rate_min * (1 - alpha)
        if self.add_internal_error:
            # Add the error series
            self.tr_internal_error.append(train_error_series)
        if self.provided_val_set:
            # Restore the best weights
            self._restore_best_state_inner()
            if self.add_internal_error:
                # Add the error series
                self.val_internal_error.append(val_error_series)

        # Cleanup
        self.inner_output_stop.reset()#reset state of the stopping criterion

    def _get_hid_delta(self, X_enriched, weights, pool_total_size, error_bar):
        # Compute the output of the hidden node
        hidden_node_net, hidden_node_output = self._compute_hid_train_output(X_enriched, weights, pool_total_size)
        # Compute output minus mean output over patterns
        hidden_node_output_delta = hidden_node_output - np.mean(hidden_node_output, axis=0)
        # Compute the correlation matrix
        correlation_matrix = error_bar.T @ hidden_node_output_delta
        # Compute sign of the correlation matrix
        sign_correlation_matrix = np.sign(correlation_matrix)
        # Compute the derivative of the activation function over the net
        derivative_over_net = np.empty((X_enriched.shape[0], pool_total_size))
        for j in range(pool_total_size):
            derivative_over_net[:, j] = common.activation_functions_derivatives[self.hidden_unit_activation_choices[j % len(self.hidden_unit_activation_choices)]](hidden_node_net[:, j])
        # multiply each element of derivative_over_net by a row of X_enriched (f'(net) * x))
        temp = np.repeat(derivative_over_net[:, np.newaxis, :], X_enriched.shape[1], axis=1) * X_enriched[:, :, np.newaxis]
        # Subtract the mean over patterns
        temp -= np.mean(temp, axis=0)
        # Compute the delta
        temp = np.tensordot(temp, error_bar, axes=([0], [0]))
        delta = np.empty((X_enriched.shape[1], pool_total_size))
        for j in range(pool_total_size):
            delta[:, j] = temp[:, j, :] @ sign_correlation_matrix[:, j]
        return delta
    
    def _compute_hid_train_output(self, X_enriched, weights, pool_total_size):
        # Compute the net of the hidden node
        hidden_node_net = X_enriched @ weights
        # Compute the output of the hidden node
        hidden_node_output = np.empty((X_enriched.shape[0], pool_total_size))
        for i in range(pool_total_size):
            hidden_node_output[:, i] = common.activation_functions[self.hidden_unit_activation_choices[i % len(self.hidden_unit_activation_choices)]](hidden_node_net[:, i])
        return hidden_node_net, hidden_node_output

    def _compute_correlation(self, X_enriched, weights, error_bar, pool_total_size, scale=False):
        # Compute output of hidden nodes
        _, hidden_node_output = self._compute_hid_train_output(X_enriched, weights, pool_total_size)
        # Compute output minus mean output over patterns
        hidden_node_output_delta = hidden_node_output - np.mean(hidden_node_output, axis=0)
        if scale:
            # Scale the output
            if hidden_node_output_delta.ndim == 1:
                hidden_node_output_delta = self._rescale_columns(hidden_node_output_delta[:, np.newaxis])
            else:
                hidden_node_output_delta = self._rescale_columns(hidden_node_output_delta)
        # Compute the correlation matrix
        correlation_matrix = error_bar.T @ hidden_node_output_delta
        # Add the error to the list
        correlation_matrix = np.abs(correlation_matrix)
        return [correlation_matrix[:, i].sum() for i in range(pool_total_size)]

    def _add_correlation_data(self, correlation_data, X_enriched, weights, error_bar, pool_total_size):
        # Compute the correlation
        correlation = self._compute_correlation(X_enriched, weights, error_bar, pool_total_size)
        # Add the correlation to the list
        for i in range(pool_total_size):
            correlation_data[i].append(correlation[i])
    
    def add_hidden_node(self, X_enriched, y, minibatch):
        max_delta = np.inf # maximum delta over the whole epoch to be used as stopping criterion
        previous_delta = 0 # used for momentum
        self.learning_rate_hidden = self.learning_rate_hidden_max
        mean_correlation = -np.inf # mean correlation over the whole epoch to be used as stopping criterion (minus inf to simulate maximization)
        # Compute the output of the output layer
        output = self.output_layer_forward(X_enriched)
        # Compute the error
        error = output - y
        # Compute error minus mean error over patterns
        error -= np.mean(error, axis=0)
        # Initialize the new weights
        n_to_test = self.hidden_unit_pool_size * len(self.hidden_unit_activation_choices)
        new_node_weights = (self.rng.randn(X_enriched.shape[1], n_to_test) - 0.5) * self.weight_amplitude
        if self.add_internal_error:
            correlation_series = []
            for i in range(n_to_test):
                correlation_series.append([])
            # Add correlation of the initial weights
            self._add_correlation_data(correlation_series, X_enriched, new_node_weights, error, n_to_test)
        iteration = 0
        # Starting permutation
        permutation = np.array([i for i in range(X_enriched.shape[0])])
        while (not self.inner_hidden_stop_check(iteration, mean_correlation, max_delta)):
            max_delta = 0
            # Shuffle the data
            next_permutation =  self.rng.permutation(X_enriched.shape[0])
            X_enriched = X_enriched[next_permutation]
            y = y[next_permutation]
            error = error[next_permutation]
            permutation = permutation[next_permutation]
            for i in range(0, X_enriched.shape[0], minibatch):
                effective_minibatch_size = min(minibatch, X_enriched.shape[0] - i)
                # Compute the delta
                delta = self._get_hid_delta(X_enriched[i:i+minibatch], new_node_weights, n_to_test, error[i:i+minibatch])
                # Update the weights
                weight_decay = self.regularization_hidden * new_node_weights[1:, :] * effective_minibatch_size / X_enriched.shape[0]
                delta = self.learning_rate_hidden * delta / X_enriched.shape[0] + self.momentum_hidden * previous_delta
                delta[1:, :] = delta[1:, :] - weight_decay
                new_node_weights += delta
                # Update max delta
                if self.inner_hidden_stop.is_delta_thresh_set():
                    max_delta = max(max_delta, np.max(np.abs(delta)))
                # Update previous delta
                previous_delta = delta
            # Prepare for next iteration
            iteration += 1
            # tweak learning rate hidden
            if self.learning_rate_decay_tau > 0:
                alpha = min(1, iteration / self.learning_rate_decay_tau)
                self.learning_rate_hidden = self.learning_rate_hidden_max * alpha + self.learning_rate_hidden_min * (1 - alpha)
            if self.add_internal_error:
                # Add correlation of the new weights
                self._add_correlation_data(correlation_series, X_enriched, new_node_weights, error, n_to_test)
            # Update min correlation
            if self.inner_hidden_stop.is_train_min_improvement_set():
                if self.add_internal_error:
                    mean_correlation = np.mean([correlation_series[i][-1] for i in range(n_to_test)])
                else:
                    mean_correlation = np.mean(self._compute_correlation(X_enriched, new_node_weights, error, n_to_test))
        # Get the node with the highest correlation
        correlations = self._compute_correlation(X_enriched, new_node_weights, error, n_to_test, scale=self.function_choice_scale)
        max_correlation_index = np.argmax(correlations)
        new_node_weights = new_node_weights[:, max_correlation_index]
        # Add the new node
        self.WJ.append(new_node_weights)
        self.fJ.append(self.hidden_unit_activation_choices[max_correlation_index % len(self.hidden_unit_activation_choices)])
        # Output of the hidden node to be returned
        best_node_output = common.activation_functions[self.fJ[-1]](X_enriched @ new_node_weights)
        best_node_output = best_node_output[np.argsort(permutation)]
        if self.rescaling:
            best_node_output = self._rescale_columns(best_node_output[:, np.newaxis])
        # Add output of the hidden node related to validation set
        if self.provided_val_set:
            out_val = common.activation_functions[self.fJ[-1]](self.val_X @ new_node_weights)
            if self.rescaling:
                out_val = self._rescale_columns(out_val[:, np.newaxis])
            self.val_X = np.c_[self.val_X, out_val]
        if self.weight_reset:
            # Reset output weights
            self.WK = (self.rng.rand(self.output_size, self.WK.shape[1] + 1) - 0.5) * self.weight_amplitude
        else:
            # Add a column of random values to the output layer weights
            self.WK = np.c_[self.WK, (self.rng.rand(self.output_size) - 0.5) * self.weight_amplitude]
        if self.add_internal_error:
            # Add the error series
            self.tr_internal_hid_error.append(correlation_series[max_correlation_index])
        # Cleanup
        self.inner_hidden_stop.reset()#reset state of the stopping criterion
        return best_node_output

    def _save_best_state_outer(self):
        self.best_WK = self.WK

    def _restore_best_state_outer(self):
        # Restore the best weights
        self.WK = self.best_WK
        # Keep only the hidden units at the time of the best validation error
        n_units = self.best_WK.shape[1] - self.input_size - 1
        self.WJ = self.WJ[:n_units]
        self.fJ = self.fJ[:n_units]

    def _save_best_state_inner(self):
        self.best_inner_WK = self.WK

    def _restore_best_state_inner(self):
        self.WK = self.best_inner_WK

    def fit(self, X, y):
        # --- Preparation of data ---
        minibatch = self.minibatch_size if self.minibatch_size is not None else X.shape[0]
        # Prepare the dataset (initially will contain only input and bias)
        X_enriched, y = self.setup_dataset(X, y)
        # Compute the output of the hidden nodes, prepare input to the output layer
        X_enriched = self.hidden_nodes_forward(X_enriched)
        if self.provided_val_set:
            # Compute the output of the output layer for validation set (to have consistency if multiple fits are performed)
            self.val_X = self.hidden_nodes_forward(self.val_X)
            # Save best weights(just to be more robust to code changes)
            self._save_best_state_outer()
        self.early_stopped = False
        # ---------------------------
        # Train output layer
        self.output_train(X_enriched, y, minibatch)
        # Compute the error on training and validation set (if provided) and the score
        self._log_error_and_score(X_enriched, y)
        while (not self.outer_stop_check()):
            # Add a hidden node and add its output to the input of the output layer
            X_enriched = np.c_[X_enriched, self.add_hidden_node(X_enriched, y, minibatch)]
            # Train output layer
            self.output_train(X_enriched, y, minibatch)
            # Compute the error and the score
            self._log_error_and_score(X_enriched, y)
        if self.provided_val_set:
            # Restore the best weights
            self._restore_best_state_outer()
        # Reset the stopping criterion
        self.outer_stop.reset()
        # Clean temporary data
        self.clean()