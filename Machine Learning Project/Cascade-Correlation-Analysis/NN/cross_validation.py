import numpy as np
import itertools
import tqdm.notebook
import pickle
import shutil
import os
from NN.ensemble import SimpleEnsemble

def DataSplit(data, target_data, test_size, seed=None):
    if test_size < 0 or test_size > 1:
        raise ValueError("val_size must be between 0 and 1")
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(data))
    data = data[permutation]
    target_data = target_data[permutation]
    last_val_index = int(len(data) * test_size)
    return data[last_val_index:], target_data[last_val_index:], data[:last_val_index], target_data[:last_val_index]
    
class CrossValidation:
    def __init__(self,
                 model_class, #Accuracy, precision, recall, f1-score, etc
                 metric_dict, #dict of metrics (name: function)
                 X, # X
                 y, # y
                 param_dict, #dict of parameters (name: list of values)
                 seed=None,
                 recovery_file = None, # file to recover data from
                 checkpoint_interval = 10 # number of parameters tested between each checkpoint
                 ):
        self.recovery_file = recovery_file
        self.model_class = model_class
        self.checkpoint_interval = checkpoint_interval
        self.X = X
        self.y = y
        self.metric_dict = metric_dict
        self.recovered_metric_names = list(metric_dict.keys())
        self.param_dict = param_dict
        if self.param_dict.pop("val_X", None) is not None:
            raise ValueError("val_X is a reserved parameter name")
        if self.param_dict.pop("val_y", None) is not None:
            raise ValueError("val_y is a reserved parameter name")
        self.recovered_param_dict = {el: [] for el in param_dict.keys()}
        self.cv_type = ""
        self.seed = seed
        self.id_combinazione = 0
        if not metric_dict.items():
            raise ValueError("At least one metric must be specified")
        # setup data
        self.expected_fits = self.get_run_amount(self.param_dict)
        self.model_train_data = {metric_name: [] for metric_name in self.metric_dict.keys()}
        self.model_val_data = {metric_name: [] for metric_name in self.metric_dict.keys()}
        self._setup_data(recovery_file)
        # shuffle data and define the splits (should be in init to avoid sampling a different split if interrupted)
        self.rng = np.random.RandomState(self.seed)
        permutation = self.rng.permutation(len(self.X))
        self.X = self.X[permutation]
        self.y = self.y[permutation]

    def __iter__(self):
        self.iterator = itertools.product(*self.param_dict.values())
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index >= len(self.model_val_data[list(self.metric_dict.keys())[0]]):
            raise StopIteration
        return {"theta" : dict(zip(self.param_dict.keys(), next(self.iterator))), 
                "train data" : {score: self.model_train_data[score][self.index] for score in self.model_train_data.keys()},
                "val data" : {score: self.model_val_data[score][self.index] for score in self.model_val_data.keys()}}

    def get_run_amount(self, param_dict):
        length = 1
        for value_list in param_dict.values():
            length *= len(value_list)
        return length

    def _get_expected_missing_fits(self):
        # count None in data
        count_none = 0
        for score in self.metric_dict.keys():
            count_none += self.model_val_data[score].count(None)
        return count_none

    def _setup_data(self, recovery_file):
        # recover data if possible and the seed used to reproduce the splits
        self._recover_data(recovery_file)
        # check if parameters have changed
        self._check_new_dict()
        # Build data structures that will hold values
        self._prepare_data_structures()
        
    def _prepare_data_structures(self):
        # Make sure that the data structures are consistent with the parameters
        if self.recovered_param_dict == self.param_dict:# if the parameters are the same as the last time, the recovery will read all the configurations and they will be consistent
            return
        fit_old = self.get_run_amount(self.recovered_param_dict)
        fit_new = self.get_run_amount(self.param_dict)
        if fit_old > fit_new:
            raise ValueError("Number of expected fits is lower than the previous one. Should not have happened")
        # if the parameters are different, the data may be inconsistent wrt the new iteration
        # shape old data to match the new parameters in a merge-like fashion
        iterator_old = itertools.product(*self.recovered_param_dict.values())
        iterator_new = itertools.product(*self.param_dict.values())
        old_idx = 0
        try:
            theta_old = dict(zip(self.recovered_param_dict.keys(), next(iterator_old)))
            for config in iterator_new:
                theta_new = dict(zip(self.param_dict.keys(), config))
                if theta_old != theta_new:
                    # add placeholder None in the position of the missing configuration
                    for score in self.metric_dict.keys():
                        self.model_train_data[score].insert(old_idx, None)
                        self.model_val_data[score].insert(old_idx, None)
                else:
                    theta_old = dict(zip(self.recovered_param_dict.keys(), next(iterator_old)))
                    old_idx += 1
        except StopIteration:
            # add placeholder None
            for config in iterator_new:
                for score in self.metric_dict.keys():
                    self.model_train_data[score].append(None)
                    self.model_val_data[score].append(None)

    # currentrly, only addition of new values is supported
    def _check_new_dict(self):
        # check if parameters have changed
        if self.recovered_param_dict.keys() != self.param_dict.keys():
            raise ValueError("Addition or removal of parameters is not supported")
        for param_name in self.param_dict.keys():
            for value in self.recovered_param_dict[param_name]:
                if value not in self.param_dict[param_name]:
                    raise ValueError("Deletion of values is not supported")
        # Addition of values is supported
        # check if metrics have changed
        if self.recovered_metric_names != list(self.metric_dict.keys()):
            raise ValueError("Addition or removal of metrics is not supported")

    def _make_consistent(self, folds, id_combinazione):
        # make data consistent (remove last run if not present in all lists)
        # train data
        consistente = True
        for score in self.metric_dict.keys():
            if self.model_train_data[score][id_combinazione] is None or len(self.model_train_data[score][id_combinazione]) != folds:
                consistente = False
                break
        if not consistente:
            for score in self.metric_dict.keys():
                self.model_train_data[score][id_combinazione] = None
        # val data
        consistente = True
        for score in self.metric_dict.keys():
            if self.model_val_data[score][id_combinazione] is None or len(self.model_val_data[score][id_combinazione]) != folds:
                consistente = False
                break
        if not consistente:
            for score in self.metric_dict.keys():
                self.model_val_data[score][id_combinazione] = None

    def _save_data(self, recovery_file, id_combinazione, folds=1):
        if recovery_file is None:
            return
        # remove validation set saved parameters
        pars_to_save = self.param_dict.copy()
        pars_to_save.pop("val_X", None)
        pars_to_save.pop("val_y", None)
        # backup old file
        if os.path.isfile(recovery_file):
            shutil.copyfile(recovery_file, recovery_file + ".backup")
        print("Saving data in " + recovery_file)
        # make data consistent
        self._make_consistent(folds, id_combinazione)
        try:
            with open(recovery_file, "wb") as f:
                # save data
                pickle.dump(self.model_train_data, f)
                pickle.dump(self.model_val_data, f)
                pickle.dump(self.seed, f)
                pickle.dump(pars_to_save, f)
                pickle.dump(list(self.metric_dict.keys()), f)
                pickle.dump(self.cv_type, f)
        except:
            pass

    def _recover_data(self, recovery_file):
        if recovery_file is None:
            return
        # check if file exists
        if not os.path.isfile(recovery_file):
            return
        try:
            with open(recovery_file, "rb") as f:
                # load data
                self.model_train_data = pickle.load(f)
                self.model_val_data = pickle.load(f)
                self.seed = pickle.load(f)
                self.recovered_param_dict = pickle.load(f)
                self.recovered_metric_names = pickle.load(f)
                self.cv_type = pickle.load(f)
        except:
            # read failed. raise exception
            raise ValueError("Unable to read recovery file " + recovery_file)
                
    def get_cv_type(self):
        return self.cv_type
    
    def get_saved_param_dict(self):
        return self.recovered_param_dict

    def KFoldCV(self, n_fold, show_progress=True):#i k fold in cui suddividere il training (K-FOLD), deve essere maggiore di 1
        # check consistency of executions
        if self.cv_type == "":
            self.cv_type = "KFoldCV" + str(n_fold)
        elif self.cv_type != "KFoldCV" + str(n_fold):
            raise ValueError("Cross validation type is not consistent with previous executions")
        # check parameters
        if n_fold < 2:
            raise ValueError("n_fold must be greater than 1")
        fold_size = len(self.X) // n_fold
        # restore iterator informations
        todo_fits = self._get_expected_missing_fits()
        if todo_fits <= 0:
            return
        iter_to_checkpoint = self.checkpoint_interval
        # gather data starting from the last run
        rand_metric = list(self.metric_dict.keys())[0]
        try:
            p_bar = tqdm.notebook.tqdm(total=todo_fits, desc="Testing parameters", disable=not show_progress)
            for self.id_combinazione, combinazione in enumerate(itertools.product(*self.param_dict.values())):
                theta = dict(zip(self.param_dict.keys(), combinazione))
                if self.model_val_data[rand_metric][self.id_combinazione] is None:
                    train_cur_fold_scores = {name: [] for name in self.metric_dict.keys()}
                    val_cur_fold_scores = {name: [] for name in self.metric_dict.keys()}
                    for i in range(0, len(self.X), fold_size):
                        # add validation set to theta
                        theta["val_X"] = self.X[i:i+fold_size]
                        theta["val_y"] = self.y[i:i+fold_size]
                        cur_train_scores, cur_val_scores = self._eval_model(self.model_class, theta, np.concatenate((self.X[:i], self.X[i+fold_size:])), np.concatenate((self.y[:i], self.y[i+fold_size:])), self.X[i:i+fold_size], self.y[i:i+fold_size])
                        for name, score in cur_train_scores.items():
                            train_cur_fold_scores[name].append(score)
                        for name, score in cur_val_scores.items():
                            val_cur_fold_scores[name].append(score)
                    for name in self.metric_dict.keys():
                        self.model_train_data[name][self.id_combinazione] = train_cur_fold_scores[name]
                        self.model_val_data[name][self.id_combinazione] = val_cur_fold_scores[name]
                    p_bar.update(1)
                    iter_to_checkpoint -= 1
                    if iter_to_checkpoint == 0:
                        self._save_data(self.recovery_file, self.id_combinazione, folds=n_fold)
                        iter_to_checkpoint = self.checkpoint_interval
        except KeyboardInterrupt:
            self._save_data(self.recovery_file, self.id_combinazione, folds=n_fold)
            p_bar.close()
            raise KeyboardInterrupt
        self._save_data(self.recovery_file, self.id_combinazione, folds=n_fold)
        p_bar.close()

    def get_best_info(self, metric, is_maximization=False):
        if metric not in self.metric_dict.keys():
            raise ValueError(metric + " must be in metric_dict.keys()")
        if self._get_expected_missing_fits() > 0:
            raise ValueError("Some data is missing. Cannot compute best model")
        # get index of best model
        best_index_fun = np.argmax if is_maximization else np.argmin
        best_index = best_index_fun(np.mean(self.model_val_data[metric], axis=1))
        # get best model
        best_model = {"theta" : dict(zip(self.param_dict.keys(), next(itertools.islice(itertools.product(*self.param_dict.values()), best_index, None)))), "train scores" : {}, "val scores" : {}}
        # add scores to info
        for score in self.model_val_data.keys():
            best_model["train scores"][score] = {}
            best_model["train scores"][score]["val"] = self.model_train_data[score][best_index]
            best_model["train scores"][score]["mean"] = np.mean(best_model["train scores"][score]["val"])
            best_model["train scores"][score]["std"] = np.std(best_model["train scores"][score]["val"])
            best_model["val scores"][score] = {}
            best_model["val scores"][score]["val"] = self.model_val_data[score][best_index]
            best_model["val scores"][score]["mean"] = np.mean(best_model["val scores"][score]["val"])
            best_model["val scores"][score]["std"] = np.std(best_model["val scores"][score]["val"])
        return best_model
    
    def get_model_data(self, tensor=False, mean_reduction=False):
        res_train = self.model_train_data
        res_val = self.model_val_data
        if mean_reduction:
            res_train = {score: np.mean(self.model_data[score], axis=1) for score in self.model_data.keys()}
            res_val = {score: np.mean(self.model_data[score], axis=1) for score in self.model_data.keys()}
        if tensor:
            dimensions = [len(self.param_dict[param]) for param in self.param_dict.keys()]
            for score in self.model_data.keys():
                res_train = np.array(self.model_data[score]).reshape(dimensions)
                res_val = np.array(self.model_data[score]).reshape(dimensions)
        return res_train, res_val

    def HoldOutCV(self, val_size, show_progress=True):#La percentuale da riservare al validation set (HOLD-OUT) (ex. 0.2)
        # check consistency of executions
        if self.cv_type == "":
            self.cv_type = "HoldOutCV" + str(val_size)
        elif self.cv_type != "HoldOutCV" + str(val_size):
            raise ValueError("Cross validation type is not consistent with previous executions")
        # check parameters
        if val_size < 0 or val_size > 1:
            raise ValueError("val_size must be between 0 and 1")
        last_val_index = int(len(self.X) * val_size)
        # restore iterator position
        todo_fits = self._get_expected_missing_fits()
        if todo_fits <= 0:
            return
        rand_metric = list(self.metric_dict.keys())[0]
        p_bar = tqdm.notebook.tqdm(total=todo_fits, desc="Testing parameters", disable=not show_progress)
        iter_to_checkpoint = self.checkpoint_interval
        # gather data
        try:
            for self.id_combinazione, combinazione in enumerate(itertools.product(*self.param_dict.values())):
                theta = dict(zip(self.param_dict.keys(), combinazione))
                if self.model_val_data[rand_metric][self.id_combinazione] is None:
                    # add validation set to theta
                    theta["val_X"] = self.X[:last_val_index]
                    theta["val_y"] = self.y[:last_val_index]
                    cur_train_scores, cur_val_scores = self._eval_model(self.model_class, theta, self.X[last_val_index:], self.y[last_val_index:], self.X[:last_val_index], self.y[:last_val_index])
                    for name in self.metric_dict.keys():
                        self.model_train_data[name][self.id_combinazione] = [cur_train_scores[name]]
                        self.model_val_data[name][self.id_combinazione] = [cur_val_scores[name]]
                    p_bar.update(1)
                    iter_to_checkpoint -= 1
                    if iter_to_checkpoint == 0:
                        self._save_data(self.recovery_file, self.id_combinazione)
                        iter_to_checkpoint = self.checkpoint_interval
        except KeyboardInterrupt:
            self._save_data(self.recovery_file, self.id_combinazione)
            p_bar.close()
            raise KeyboardInterrupt
        self._save_data(self.recovery_file, self.id_combinazione)
        p_bar.close()

    def _eval_model(self, model_class, params, train_data, train_target_data, test_data, test_target_data):
        model = model_class(**params)
        model.fit(train_data, train_target_data)
        train_predictions = model.predict(train_data)
        train_scores = {name: metric(train_predictions, train_target_data) for name, metric in self.metric_dict.items()}
        predictions = model.predict(test_data)
        val_scores = {name: metric(predictions, test_target_data) for name, metric in self.metric_dict.items()}
        return train_scores, val_scores

    def export_csv(self, file_name, metric, csv_line_fun=None, header_fun=None):
        if metric not in self.metric_dict.keys():
            raise ValueError(metric + " must be in metric_dict.keys()")
        if self._get_expected_missing_fits() > 0:
            raise ValueError("Some data is missing. Cannot export csv")
        # print all combinations of variable parameters and their scores
        var_params = [param_name for param_name, param_values in self.param_dict.items() if len(param_values) > 1]
        with open(file_name, "w") as f:
            # print header
            if header_fun is not None:
                f.write(header_fun(list(self.param_dict.keys()), var_params) + ",")
            else:
                f.write(",".join(var_params) + ",")
            f.write("Train" + metric + "," + "Val" + metric + "\n")
            # print data
            for i, combinazione in enumerate(itertools.product(*self.param_dict.values())):
                if self.model_val_data[metric][i] is None:
                    continue
                theta = dict(zip(self.param_dict.keys(), combinazione))
                if csv_line_fun is not None:
                    f.write(csv_line_fun(theta, var_params) + ",")
                else:
                    f.write(",".join([str(theta[param_name]) for param_name in var_params]) + ",")
                f.write(str(np.mean(self.model_train_data[metric][i])) + "," + str(np.mean(self.model_val_data[metric][i])) + "\n")

    def sensitivity_analysis(self, params_to_check, score):
        # Check existence of params
        for param_name in params_to_check:
            if param_name not in self.param_dict.keys():
                raise ValueError(param_name + " must be in param_dict.keys()")
        if not score in self.metric_dict.keys():
            raise ValueError(score + " must be in metric_dict.keys()")
        if not self.model_val_data[score]:
            raise ValueError("No data has been collected yet")
        # Get length of each dimension
        dimensions = [len(self.param_dict[param]) for param in self.param_dict.keys()]
        # Compute mean of data
        train_mean_scores_over_folds = np.mean(self.model_train_data[score], axis=1)
        val_mean_scores_over_folds = np.mean(self.model_val_data[score], axis=1)
        # Reshape to matrix
        reduction_axis = tuple(i for i, param_name in enumerate(self.param_dict.keys()) if param_name not in params_to_check)
        train_mean_scores_over_folds = train_mean_scores_over_folds.reshape(dimensions)
        mean_scores = np.mean(train_mean_scores_over_folds, axis=reduction_axis)
        std_scores = np.std(train_mean_scores_over_folds, axis=reduction_axis)
        val_mean_scores_over_folds = val_mean_scores_over_folds.reshape(dimensions)
        mean_scores_val = np.mean(val_mean_scores_over_folds, axis=reduction_axis)
        std_scores_val = np.std(val_mean_scores_over_folds, axis=reduction_axis)
        return {"mean": mean_scores, "std": std_scores}, {"mean": mean_scores_val, "std": std_scores_val}
    
class CVInternalHoldOut:
    def __init__(self,
                 model_class, #Accuracy, precision, recall, f1-score, etc
                 metric_dict, #dict of metrics (name: function)
                 X, # X
                 y, # y
                 param_dict, #dict of parameters (name: list of values)
                 seed=None,
                 recovery_file_prefix = None, # file to recover data from
                 checkpoint_interval = 10 # number of parameters tested between each checkpoint
                 ):
        self.recovery_file_prefix = recovery_file_prefix
        self.recovery_file = recovery_file_prefix + ".pkl" if recovery_file_prefix is not None else None
        self.model_class = model_class
        self.checkpoint_interval = checkpoint_interval
        self.X = X
        self.y = y
        self.cv_type = ""
        self.metric_dict = metric_dict
        self.param_dict = param_dict
        self.seed = seed
        if not metric_dict.items():
            raise ValueError("At least one metric must be specified")
        # shuffle data and define the splits (should be in init to avoid sampling a different split if interrupted)
        self.rng = np.random.RandomState(self.seed)
        permutation = self.rng.permutation(len(self.X))
        self.X = self.X[permutation]
        self.y = self.y[permutation]
        self.validations = []
        self._recover_data(recovery_file_prefix)

    #one model per fold, folds_to_compute = None -> all folds, otherwise list of folds to compute
    def select_models(self, n_folds, internal_val_size, show_progress=True, folds_to_compute=None):
        if n_folds < 2:
            raise ValueError("n_fold must be greater than 1")
        if internal_val_size < 0 or internal_val_size > 1:
            raise ValueError("internal_val_size must be between 0 and 1")
        if self.cv_type == "":
            self.cv_type = "InternalHoldOutCV" + str(n_folds) + "_" + str(internal_val_size)
        elif self.cv_type != "InternalHoldOutCV" + str(n_folds) + "_" + str(internal_val_size):
            raise ValueError("Cross validation type is not consistent with previous executions")
        self.validations = [None] * n_folds
        fold_size = len(self.X) // n_folds
        try:
            for fold_num, i in enumerate(range(0, len(self.X), fold_size)):
                # internal hold out cross validation
                # strip suffix from recovery file and add _fold_i
                if folds_to_compute is not None and fold_num not in folds_to_compute:
                    continue
                cur_recovery_file = None
                if self.recovery_file_prefix is not None:
                    cur_recovery_file = self.recovery_file_prefix + "_fold_" + str(fold_num) + ".pkl"
                hold_out_validation = CrossValidation(self.model_class, 
                                                        self.metric_dict, 
                                                        np.concatenate((self.X[:i], self.X[i+fold_size:])), 
                                                        np.concatenate((self.y[:i], self.y[i+fold_size:])), 
                                                        self.param_dict, 
                                                        seed=self.seed,
                                                        recovery_file=cur_recovery_file, 
                                                        checkpoint_interval=self.checkpoint_interval)
                self.validations[fold_num] = hold_out_validation
                hold_out_validation.HoldOutCV(internal_val_size, show_progress=show_progress)
        except KeyboardInterrupt:
            self._save_data(self.recovery_file)
            raise KeyboardInterrupt
        # save data
        self._save_data(self.recovery_file)

    def _save_data(self, recovery_file):
        if recovery_file is None:
            return
        # backup old file
        if os.path.isfile(recovery_file):
            shutil.copyfile(recovery_file, recovery_file + ".backup")
        print("Saving data in " + recovery_file)
        try:
            with open(recovery_file, "wb") as f:
                # save data
                pickle.dump(self.seed, f)
                pickle.dump(self.cv_type, f)
        except:
            pass

    def _recover_data(self, recovery_file):
        if recovery_file is None:
            return
        # check if file exists
        if not os.path.isfile(recovery_file):
            return
        try:
            with open(recovery_file, "rb") as f:
                # load data
                self.seed = pickle.load(f)
                self.cv_type = pickle.load(f)
        except:
            # read failed. raise exception
            raise ValueError("Unable to read recovery file " + recovery_file)
        
    def get_best_info(self, metric, is_maximization=False):
        if metric not in self.metric_dict.keys():
            raise ValueError(metric + " must be in metric_dict.keys()")
        return {"model " + str(i): validation.get_best_info(metric, is_maximization) for i, validation in enumerate(self.validations)}
    
    def model_assessment(self, metric, is_maximization=False, tweak_epochs_factor=10, epochs_param_name="epochs"):
        # re-fit each model on the design set and test it on the test set
        res = {name: [] for name in self.metric_dict.keys()}
        fold_size = len(self.X) // len(self.validations)
        for i, test_set in enumerate(range(0, len(self.X), fold_size)):
            best_model_info = self.validations[i].get_best_info(metric, is_maximization)
            best_params = best_model_info["theta"]
            # add train_min_error to parameters since the retraining is done without validation set
            best_params["train_min_error"] = best_model_info["train scores"][metric]["mean"]
            # tweak epochs to avoid underfitting
            if epochs_param_name in best_params.keys():
                best_params[epochs_param_name] = int(best_params[epochs_param_name] * tweak_epochs_factor)
            # fit model on design set
            model = self.model_class(**best_params)
            model.fit(np.concatenate((self.X[:test_set], self.X[test_set+fold_size:])), np.concatenate((self.y[:test_set], self.y[test_set+fold_size:])))
            # test model on test set
            predictions = model.predict(self.X[test_set:test_set+fold_size])
            scores = {name: metric(predictions, self.y[test_set:test_set+fold_size]) for name, metric in self.metric_dict.items()}
            for name, score in scores.items():
                res[name].append(score)
        # compute mean and std
        for name, score in res.items():
            res[name] = {"mean": np.mean(score), "std": np.std(score)}
        return res
    
    def export_csv(self, file_name_prefix, metric, csv_line_fun=None, csv_header_fun=None, folds_to_export=None):
        if folds_to_export is None:
            folds_to_export = range(len(self.validations))
        for fold in folds_to_export:
            self.validations[fold].export_csv(file_name_prefix + "_fold_" + str(fold) + ".csv", metric, csv_line_fun, csv_header_fun)

    def generate_ensemble(self, metric, is_maximization=False, tweak_epochs_factor=10, epochs_param_name="epochs"):
        # get best model for each fold
        best_models = self.get_best_info(metric, is_maximization)
        # get parameters list
        best_models_temp = [best_models["model " + str(i)]["theta"] for i in range(len(best_models))]
        # remove duplicates
        best_models_params = []
        best_models_tr_to_reach = []
        for i in range(len(best_models_temp)):
            if best_models_temp[i] not in best_models_params:
                best_models_params.append(best_models_temp[i])
                best_models_tr_to_reach.append([best_models["model " + str(i)]["train scores"][metric]["mean"]])
            else:
                index = best_models_params.index(best_models_temp[i])
                best_models_tr_to_reach[index].append(best_models["model " + str(i)]["train scores"][metric]["mean"])
        # compute mean of train scores
        for i in range(len(best_models_tr_to_reach)):
            best_models_tr_to_reach[i] = np.mean(best_models_tr_to_reach[i])
        # add train_min_error to parameters since the ensemble should be trained on the whole dataset and validation set is not available
        for params, tr_to_reach in zip(best_models_params, best_models_tr_to_reach):
            params["train_min_error"] = tr_to_reach
            # tweak epochs to avoid underfitting
            if epochs_param_name in params.keys():
                params[epochs_param_name] = int(params[epochs_param_name] * tweak_epochs_factor)
        # generate ensemble
        ensemble = SimpleEnsemble(best_models_params, [self.model_class for i in range(len(best_models_params))])
        # fit ensemble on the whole dataset
        ensemble.fit(self.X, self.y)
        return ensemble