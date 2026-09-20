# transformers/samplers.py
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class SamplerFactory:
    """
    Builds an imblearn sampler from config. Not a transformer itself —
    samplers implement fit_resample (not transform), and must only ever
    be used inside an imblearn.pipeline.Pipeline, never a plain sklearn
    Pipeline, and never applied at inference time (imblearn correctly
    no-ops resampling during .predict()/.transform() calls, only acting
    during .fit()/.fit_resample()).
    """

    @staticmethod
    def build(data_config: dict, random_state: int = 42):
        if data_config is None:
                    return None
        over_cfg = data_config.get("over_sampling")
        under_cfg = data_config.get("under_sampling")

        if over_cfg and under_cfg:
            raise ValueError("Configure only one of over_sampling / under_sampling per training_config entry.")

        if over_cfg:
            method = over_cfg.get("method", "SMOTE")
            if method == "SMOTE":
                return SMOTE(random_state=random_state)
            raise ValueError(f"Unsupported over_sampling method: {method}")

        if under_cfg:
            method = under_cfg.get("method", "RandomUnderSampling")
            if method == "RandomUnderSampling":
                return RandomUnderSampler(random_state=random_state)
            raise ValueError(f"Unsupported under_sampling method: {method}")

        return None