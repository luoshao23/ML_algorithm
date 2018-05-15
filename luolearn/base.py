import warnings
from collections import defaultdict
from inspect import signature


class BaseEstimator(object):

    @classmethod
    def _get_param_names(cls):

        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []

        init_signature = signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("Errro")
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        out = {}
        for key in self._get_param_names():
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    continue
            finally:
                warnings.filters.pop(0)

            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        if not params:
            return self

        valid_params = self.get_params(deep=True)

        neste_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid')

            if delim:
                neste_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        for key, sub_params in neste_params.items():
            valid_params[key].set_params(**sub_params)

        return self


class ClassifierMixin(object):

    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class RegressorMixin(object):
    def score(self, X, y, sample_weight=None):
        pass