from mlflow.tracking import MlflowClient

import mlflow

client = MlflowClient()

run_id = "8027495603bf4c3b929004b810d7a20f"

model_path = "mlflow-artifacts:/783572788091518455/8027495603bf4c3b929004b810d7a20f/artifacts/Best Model"

model_name = "water_potability_rf"


model_uri = f"runs:/{run_id}/{model_path}"


reg= mlflow.register_model(model_uri, model_name)


