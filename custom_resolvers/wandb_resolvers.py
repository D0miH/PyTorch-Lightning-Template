import wandb
import os


def wandb_artifact(artifact_path: str, file_name: str, artifact_type: str = None):
    api = wandb.Api()
    art = api.artifact(artifact_path, type=artifact_type)
    local_path = art.download()
    return os.path.join(local_path, file_name)
