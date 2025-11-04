import os
import glob

HF_LOCAL = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

def get_local_model(remote_name: str) -> str:

    model_dir = os.path.join(HF_LOCAL, "models--" + remote_name.replace("/", "--"))
    snapshots_dir = os.path.join(model_dir, "snapshots")

    if not os.path.exists(snapshots_dir):
        
        return remote_name


    snapshots = sorted(
        glob.glob(os.path.join(snapshots_dir, "*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not snapshots:
        return remote_name

    return snapshots[0]
