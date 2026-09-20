import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri="http://localhost:5000")
client.delete_registered_model_alias("EasyVisa_Classification_Model", "champion")