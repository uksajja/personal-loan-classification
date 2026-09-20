from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

 


class ModelFactory:
    """Dynamically instantiates different model architectures with unique parameters."""
    def __init__(self):
        pass

    # registered model types
    _REGISTRY = {
        'logistic_regression': LogisticRegression,
        "decision_tree_classifier": DecisionTreeClassifier
    }
       
    @classmethod
    def get_model(cls, model_type: str, hyperparameters: dict | None):
        if model_type not in cls._REGISTRY:
            raise ValueError(f"Unknown model_type '{model_type}'. Registered: {list(cls._REGISTRY)}")
        hyperparameters = hyperparameters or {}
        resolved = {key: cls.resolve_param(value) for key, value in hyperparameters.items()}
        return cls._REGISTRY[model_type](**resolved)

    @classmethod
    def resolve_param(cls, value):
        """Recursively resolves nested model specs -- dicts shaped like
        {model_type, hyperparameters} -- into real estimator objects.
        Single source of truth, reused by both get_model() and
        get_model_search_space().
        """
        if isinstance(value, dict) and "model_type" in value:
            return cls.get_model(value["model_type"], value.get("hyperparameters", {}))
        if isinstance(value, list):
            return [cls.resolve_param(v) for v in value]
        return value

    @classmethod
    def get_model_search_space(cls, search_space: dict, step_name: str = "model") -> dict:
        """Prepares a fine_tuning search_space for GridSearchCV/RandomizedSearchCV
        against a Pipeline: resolves nested model specs (same rule as
        get_model) and namespaces bare keys with the pipeline step name
        (n_estimators -> model__n_estimators). Keys already namespaced
        (containing '__') pass through untouched, so a config can target a
        different step explicitly (e.g. sampler__k_neighbors).
        """
        return {
            (key if "__" in key else f"{step_name}__{key}"): cls.resolve_param(value)
            for key, value in search_space.items()
        }

    