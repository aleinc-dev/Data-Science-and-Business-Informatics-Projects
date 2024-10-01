import numpy as np
from enum import Enum, auto
import itertools
import pickle
import tqdm.notebook

class ActivationFunction(Enum):
    def _generate_next_value_(name, start, count, last_values):# To start from 0
        return count
    SIGMOID   =  auto()
    TANH      =  auto()
    SOFTPLUS  =  auto()
    LINEAR    =  auto()
    RELU      =  auto()
    SOFTMAX   =  auto()

class LossFunction(Enum):
    def _generate_next_value_(name, start, count, last_values):# To start from 0
        return count
    MSE       =  auto() # Mean Squared Error
    MEE       =  auto() # Mean Euclidean Error

# ------------------------------------
# Activation functions implementations
# ------------------------------------

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Tanh(x):
    return np.tanh(x)

def SoftPlus(x):
    return np.log(1 + np.exp(x))

def Linear(x):
    return x

def ReLU(x):
    return np.maximum(0, x)

def Softmax(x):# stable version
    if x.ndim > 1:
        temp = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
        return temp / np.sum(temp, axis=1)[:, np.newaxis]
    else:
        return np.ones(x.shape[0])
    
activation_functions = {
    ActivationFunction.SIGMOID: Sigmoid,
    ActivationFunction.TANH: Tanh,
    ActivationFunction.SOFTPLUS: SoftPlus,
    ActivationFunction.LINEAR: Linear,
    ActivationFunction.RELU: ReLU,
    ActivationFunction.SOFTMAX: Softmax
}

# ------------------------------------------------
# Activation functions derivatives implementations
# ------------------------------------------------

def SigmoidDerivative(x):
    temp = Sigmoid(x)
    return temp * (1 - temp)

def TanhDerivative(x):
    return 1 - np.tanh(x)**2

def SoftPlusDerivative(x):
    return 1 / (1 + np.exp(-x))

def LinearDerivative(x):
    return 1

def ReLUDerivative(x):
    return (x > 0).astype(int)

def SoftmaxDerivative(x):
    if x.ndim <= 1:
        return np.zeros(x.shape[0])
    f = Softmax(x)
    return f * (1 - f)


activation_functions_derivatives = {
    ActivationFunction.SIGMOID: SigmoidDerivative,
    ActivationFunction.TANH: TanhDerivative,
    ActivationFunction.SOFTPLUS: SoftPlusDerivative,
    ActivationFunction.LINEAR: LinearDerivative,
    ActivationFunction.RELU: ReLUDerivative,
    ActivationFunction.SOFTMAX: SoftmaxDerivative
}

# ------------------------------------
# Loss functions implementations
# ------------------------------------

def MSE(out, target):
    return np.sum((out - target)**2) / out.shape[0]

def MEE(out, target):
    return np.sum(np.sqrt(np.sum((out - target)**2, axis=1))) / out.shape[0]

loss_functions = {
    LossFunction.MSE: MSE,
    LossFunction.MEE: MEE
}

# ------------------------------------------
# Loss functions derivatives implementations
# ------------------------------------------
# Excludes the 1/N factor, added during eta tweaking

def MSEDerivative(out, target):
    return 2 * (out - target)

def MEEDerivative(out, target):
    return (out - target) / np.sum(np.sqrt(np.sum((out - target)**2, axis=1)))

loss_functions_derivatives = {
    LossFunction.MSE: MSEDerivative,
    LossFunction.MEE: MEEDerivative
}

# --------------
# Early stopping
# --------------

# Patience will consider improvement with respecto to the known best value, not the previous one
class EarlyStopping:
    def __init__(self, 
                 save_state, # function to save the state of the model
                 val_min_error=0, # minimum error to consider the validation converged
                 val_patience=0, # number of non-improving(small improvement or increase) epochs to consider the validation converged
                 val_min_improvement=None,
                 delta_thresh=None, 
                 train_min_error=0, 
                 train_patience=0, 
                 train_min_improvement=None):
        self.save_state = save_state
        if val_patience < 0 or train_patience < 0:
            raise ValueError("Patience must be non negative")
        self.val_min_err = val_min_error
        self.val_patience = val_patience
        self.val_min_improvement = val_min_improvement
        self.delta_thresh = delta_thresh
        self.train_min_err = train_min_error
        self.train_patience = train_patience
        self.train_min_improvement = train_min_improvement
        self.best_val_err = np.inf
        self.best_train_err = np.inf
        self.val_patience_counter = 0
        self.train_patience_counter = 0 # also for delta decrease

    def is_delta_thresh_set(self):
        return self.delta_thresh is not None

    def is_train_min_improvement_set(self):
        return self.train_min_improvement is not None

    def check_val_convergence(self, val_err):
        if val_err is None:
            return False
        prev_best = self.best_val_err
        if val_err < self.best_val_err:
            self.best_val_err = val_err
            self.save_state()
        if val_err <= self.val_min_err:
            return True
        if self.val_min_improvement is None:
            return False
        if prev_best - val_err < self.val_min_improvement:
            self.val_patience_counter += 1
        else:
            self.val_patience_counter = 0
        return self.val_patience_counter > self.val_patience
    
    def check_train_convergence(self, train_err, delta):
        prev_best = self.best_train_err
        if train_err < self.best_train_err:
            self.best_train_err = train_err
        if train_err <= self.train_min_err:
            return True
        if self.delta_thresh is None and self.train_min_improvement is None:
            return False
        if  (self.train_min_improvement is None or prev_best - train_err < self.train_min_improvement) and (self.delta_thresh is None or (np.abs(delta) < self.delta_thresh).all()):
            self.train_patience_counter += 1
        else:
            self.train_patience_counter = 0
        return self.train_patience_counter > self.train_patience

    # val_err can be None
    def check_convergence(self, val_err, train_err, delta):
        return self.check_val_convergence(val_err) or self.check_train_convergence(train_err, delta)
    
    def reset(self):
        self.best_val_err = np.inf
        self.best_train_err = np.inf
        self.val_patience_counter = 0
        self.train_patience_counter = 0

    def print_convergence_reason(self):
        if self.best_train_err < self.train_min_err:
            print("Training error reached minimum")
        elif self.train_patience_counter > self.train_patience:
            print("Training error did not improve and delta was small for {} epochs".format(self.train_patience))
        if self.best_val_err < self.val_min_err:
            print("Validation error reached minimum")
        elif self.val_patience_counter > self.val_patience:
            print("Validation error did not improve for {} epochs".format(self.val_patience))


# ------------------
# Scores computation
# ------------------

class ClassificationPostprocessing(Enum):
    def _generate_next_value_(name, start, count, last_values):# To start from 0
        return count
    ONE_HOT   = auto()
    ORDERED   = auto()
    THRESHOLD = auto()
    

def PostprocessingOneHot(out, target):
    return np.argmax(out, axis=1), np.argmax(target, axis=1)

def PostprocessingOrdered(out, target):
    return np.rint(out), target

def PostprocessingThreshold(out, target, threshold, positive_check_value=1, negative_check_value=-1):
    return np.where(out > threshold, positive_check_value, negative_check_value), target



OUTPUT_POSTPROCESSORS = {
    ClassificationPostprocessing.ONE_HOT: PostprocessingOneHot,
    ClassificationPostprocessing.ORDERED: PostprocessingOrdered,
    ClassificationPostprocessing.THRESHOLD: PostprocessingThreshold
}

def accuracy(out, target, postprocessor=None, **kwargs):
    if postprocessor is not None:
        out, target = OUTPUT_POSTPROCESSORS[postprocessor](out, target, **kwargs)
    return np.mean(out == target)

# ---------------------------
# Data collection for testing
# ---------------------------

# Data recovery allows to load previous results and continue from there, but it will not check for parameters changes
class ModelDataCollector:
    def __init__(self,
                 model_class, #model to test
                 X, #training data
                 y, # target data
                 param_dict, #dict of parameters (name: list of values)
                 show_progress=True,
                 recovery_file = None # file to recover data from
                 ):
        self.show_progress = show_progress
        self.model = None # will keep the last model trained(to check)
        self.model_class = model_class
        self.X = X
        self.y = y
        self.param_dict = param_dict
        self.tr_error_list,self.val_error_list,self.tr_score_list, self.val_score_list = None, None, None, None
        self.run_metadata = []
        self.recovery_file = recovery_file

    def __iter__(self):
        if self.tr_error_list is None:
            self.gather_data()
        self.iterator = itertools.product(*self.param_dict.values())
        self.index = -1
        return self
    
    def __next__(self):
        self.index += 1
        if self.index >= len(self.tr_error_list):
            raise StopIteration
        res_metadata = self.run_metadata[self.index] if len(self.run_metadata) > 0 else None
        res_tr_error = self.tr_error_list[self.index] if len(self.tr_error_list) > 0 else []
        res_val_error = self.val_error_list[self.index] if len(self.val_error_list) > 0 else []
        res_tr_score = self.tr_score_list[self.index] if len(self.tr_score_list) > 0 else []
        res_val_score = self.val_score_list[self.index] if len(self.val_score_list) > 0 else []
        return dict(zip(self.param_dict.keys(), next(self.iterator))), res_metadata, res_tr_error, res_val_error, res_tr_score, res_val_score
        
    def get_run_amount(self):
        length = 1
        for value_list in self.param_dict.values():
            length *= len(value_list)
        return length

    def _save_data(self, recovery_file):
        if recovery_file is None:
            return
        try:
            with open(recovery_file, "wb") as f:
                pickle.dump(self.tr_error_list, f)
                pickle.dump(self.val_error_list, f)
                pickle.dump(self.tr_score_list, f)
                pickle.dump(self.val_score_list, f)
                pickle.dump(self.run_metadata, f)
        except:
            pass

    def _recover_data(self, recovery_file):
        if recovery_file is None:
            return
        try:
            with open(recovery_file, "rb") as f:
                self.tr_error_list = pickle.load(f)
                self.val_error_list = pickle.load(f)
                self.tr_score_list = pickle.load(f)
                self.val_score_list = pickle.load(f)
                self.run_metadata = pickle.load(f)
        except:
            pass


    # if show_progress is set, it will override the show_progress parameter passed to the constructor
    def gather_data(self, metadata_callback=None, show_progress = None, recovery_file = None):#i k fold in cui suddividere il training (K-FOLD), deve essere maggiore di 1
        if show_progress is None:
            show_progress = self.show_progress
        if recovery_file is None:
            recovery_file = self.recovery_file
        self.tr_error_list,self.val_error_list,self.tr_score_list, self.val_score_list = [],[],[],[]
        self._recover_data(recovery_file)
        starting_index = len(self.tr_error_list)
        runs = self.get_run_amount()
        if starting_index >= runs:
            return
        if starting_index > 0:
            print("Resuming from run " + str(starting_index))
        iterator = itertools.product(*self.param_dict.values())
        for _ in range(starting_index):
            next(iterator)
        try :
            for combinazione in tqdm.notebook.tqdm(iterator, 
                                total=runs - starting_index, 
                                desc="Gathering data",
                                disable=not self.show_progress):
                theta = dict(zip(self.param_dict.keys(), combinazione))
                tr_error, val_error, tr_score, val_score = self._eval_model(self.model_class, theta, self.X, self.y)
                if len(tr_error) > 0:
                    self.tr_error_list.append(tr_error)
                if len(val_error) > 0:
                    self.val_error_list.append(val_error)
                if len(tr_score) > 0:
                    self.tr_score_list.append(tr_score)
                if len(val_score) > 0:
                    self.val_score_list.append(val_score)
                if metadata_callback is not None:
                    self.run_metadata.append(metadata_callback(theta, tr_error, val_error, tr_score, val_score))
        except KeyboardInterrupt:
            if recovery_file is not None:
                print("Interrupted, progress saved in " + recovery_file)
                # make data consistent (remove last run if not present in all lists)
                max_index = min(len(self.tr_error_list), len(self.val_error_list), len(self.tr_score_list), len(self.val_score_list))
                self.tr_error_list = self.tr_error_list[:max_index]
                self.val_error_list = self.val_error_list[:max_index]
                self.tr_score_list = self.tr_score_list[:max_index]
                self.val_score_list = self.val_score_list[:max_index]
                self._save_data(recovery_file)
                raise KeyboardInterrupt
        self._save_data(recovery_file)    
        return 
    
    def sensitivity_analysis(self, params_to_check):
        # Check existence of params
        for param_name in params_to_check:
            if param_name not in self.param_dict.keys():
                raise ValueError(param_name + " must be in param_dict.keys()")
        if self.tr_error_list is None:
            # Gather data
            self.gather_data()
        # Get length of each dimension
        dimensions = [len(self.param_dict[param]) for param in self.param_dict.keys()]
        # Take last error and score for each run
        last_tr_error = [self.tr_error_list[i][-1] for i in range(len(self.tr_error_list))]
        last_val_error = [self.val_error_list[i][-1] for i in range(len(self.val_error_list))]
        last_tr_score = [self.tr_score_list[i][-1] for i in range(len(self.tr_score_list))]
        last_val_score = [self.val_score_list[i][-1] for i in range(len(self.val_score_list))]
        # Reshape to matrix
        reduction_axis = tuple(i for i, param_name in enumerate(self.param_dict.keys()) if param_name not in params_to_check)
        last_tr_error = np.array(last_tr_error).reshape(dimensions)
        last_tr_error_mean = np.mean(last_tr_error, axis=reduction_axis)
        last_tr_error_std = np.std(last_tr_error, axis=reduction_axis)
        global_tr_error_mean = np.mean(last_tr_error_mean)
        parameter_tr_error_std = np.std(last_tr_error_mean)
        global_tr_error_std = np.std(last_tr_error)
        # Compute mean and std
        last_val_error_mean, last_val_error_std, global_val_error_mean, global_val_error_std, parameter_val_error_std = None, None, None, None, None
        if last_val_error:
            last_val_error = np.array(last_val_error).reshape(dimensions)
            last_val_error_mean = np.mean(last_val_error, axis=reduction_axis)
            last_val_error_std = np.std(last_val_error, axis=reduction_axis)
            global_val_error_mean = np.mean(last_val_error_mean)
            parameter_val_error_std = np.std(last_val_error_mean)
            global_val_error_std = np.std(last_val_error)
        # Compute mean and std
        last_tr_score_mean, last_tr_score_std, global_tr_score_mean, global_tr_score_std, parameter_tr_score_std = None, None, None, None, None
        if last_tr_score:
            last_tr_score = np.array(last_tr_score).reshape(dimensions)
            last_tr_score_mean = np.mean(last_tr_score, axis=reduction_axis)
            last_tr_score_std = np.std(last_tr_score, axis=reduction_axis)
            global_tr_score_mean = np.mean(last_tr_score_mean)
            parameter_tr_score_std = np.std(last_tr_score_mean)
            global_tr_score_std = np.std(last_tr_score)

        last_val_score_mean, last_val_score_std, global_val_score_mean, global_val_score_std, parameter_val_score_std = None, None, None, None, None
        if last_val_score:
            last_val_score = np.array(last_val_score).reshape(dimensions)
            last_val_score_mean = np.mean(last_val_score, axis=reduction_axis)
            last_val_score_std = np.std(last_val_score, axis=reduction_axis)
            global_val_score_mean = np.mean(last_val_score_mean)
            parameter_val_score_std = np.std(last_val_score_mean)
            global_val_score_std = np.std(last_val_score)
        # Return results
        return {
            "tr_error_mean": last_tr_error_mean,
            "val_error_mean": last_val_error_mean,
            "tr_score_mean": last_tr_score_mean,
            "val_score_mean": last_val_score_mean,
            "tr_error_std": last_tr_error_std,
            "val_error_std": last_val_error_std,
            "tr_score_std": last_tr_score_std,
            "val_score_std": last_val_score_std,
            "parameter_tr_error_std": parameter_tr_error_std,
            "parameter_val_error_std": parameter_val_error_std,
            "parameter_tr_score_std": parameter_tr_score_std,
            "parameter_val_score_std": parameter_val_score_std,
            "global_tr_error_mean": global_tr_error_mean,
            "global_val_error_mean": global_val_error_mean,
            "global_tr_score_mean": global_tr_score_mean,
            "global_val_score_mean": global_val_score_mean,
            "global_tr_error_std": global_tr_error_std,
            "global_val_error_std": global_val_error_std,
            "global_tr_score_std": global_tr_score_std,
            "global_val_score_std": global_val_score_std
        }

    def get_error_curve(self):
        return self.tr_error_list,self.val_error_list
    
    def get_score_curve(self):
        return self.tr_score_list, self.val_score_list
    
    def get_run_metadata(self):
        return self.run_metadata
    
    def _eval_model(self, model_class, params, X, y,):
        self.model = model_class(**params)
        self.model.fit(X, y)
        tr_error, val_error = self.model.get_error_curve()
        tr_score, val_score = self.model.get_score_curve()
        return tr_error, val_error, tr_score, val_score
