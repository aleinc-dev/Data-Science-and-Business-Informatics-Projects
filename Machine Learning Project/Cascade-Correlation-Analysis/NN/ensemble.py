import numpy as np

class SimpleEnsemble:
    def __init__(self, model_param_list, model_class_list):
        self.model_param_list = model_param_list
        self.model_class_list = model_class_list
        self.models = []

    def fit(self, X, y):
        for params, model_class in zip(self.model_param_list, self.model_class_list):
            model = model_class(**params)
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        return np.mean([model.predict(X) for model in self.models], axis=0)