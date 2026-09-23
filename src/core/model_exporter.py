import os
from pathlib import Path
from typing import Any


class ModelExporter:
    """A helper class to save and load machine learning models."""

    def __init__(self):
        pass

    @staticmethod
    def save_model(model: Any, path: Path):
        suffix = path.suffix
        if suffix == '.joblib':
            ModelExporter.save_model_joblib(model, path)
        elif suffix == '.pkl':
            ModelExporter.save_model_pkl(model, path)
        elif suffix == '.onnx':
            ModelExporter.save_model_onnx(model, path)
        else:
            raise ValueError("Can't Save Model: Invalid Model Format")

    @staticmethod
    def load_model(model: Any, path: Path):
        suffix = path.suffix
        if suffix == '.joblib':
            return ModelExporter.load_model_joblib(path)
        elif suffix == '.pkl':
            return ModelExporter.load_model_pkl(path)
        elif suffix == '.onnx':
            return ModelExporter.load_model_onnx(path)
        else:
            raise ValueError("Can't Load Model: Invalid Model Format")   

   
    #---------------------------------------------------------------------------
    # Joblib Format
    #--------------------------------------------------------------
    #-------------------------- save_model_joblib -----------------------------
    @staticmethod
    def save_model_joblib(model, path: Path):
        import joblib
        joblib.dump(model, path)
    
    #-------------------------- load_model_joblib -----------------------------
    @staticmethod
    def load_model_joblib(path: Path):
        import joblib
        return joblib.load(path)
    
    #---------------------------------------------------------------------------
    # Pickle Format
    #---------------------------------------------------------------------------
    #--------------------------- save_model_pickle -----------------------------
    @staticmethod
    def save_model_pkl(model, path: Path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(model, f)
    #--------------------------- load_model_pickle -----------------------------
    @staticmethod
    def load_model_pkl(path: Path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    
    #---------------------------------------------------------------------------
    # ONNX Format
    #--------------------------------------------------------------
    #--------------------------- save_model_onnx ------------------------------
    @staticmethod
    def save_model_onnx(model, path: Path):
        import onnx
        import onnxruntime
        import skl2onnx
        import numpy as np
        from skl2onnx import convert_sklearn
        input_schema = np.array([[1,11]])
        onnx_model = convert_sklearn(model, initial_types=input_schema)

        with open(path, "wb") as f:
            f.write(model.SerializeToString())
    #--------------------------- load_model_onnx ------------------------------
    @staticmethod
    def load_model_onnx(path: Path):
        import onnx
        return onnx.load(path)
      
    
    #------------------------- get_model_signature -----------------------------
    @staticmethod
    def get_model_signature(model):
        # Placeholder for future implementation to extract model signature (e.g., input/output schema)
        return {"input_features": None, "output_labels": None}
    