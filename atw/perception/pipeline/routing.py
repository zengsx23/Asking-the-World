from pathlib import Path


def resolve_sam3_checkpoint(checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint
    from huggingface_hub import hf_hub_download
    from scripts.prepare_sam3 import REPO_ID, REVISION

    return Path(hf_hub_download(
        repo_id=REPO_ID, filename="sam3.pt", revision=REVISION,
        local_files_only=True,
    ))
