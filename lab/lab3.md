# Lab 3 - Containerizing the model with Docker

## Question 1

The model was initially registered as `food11` version 1. A logged model artifact belongs to one specific training run and stores the model files produced by that run, while a registered model gives the model a stable name, independent versions, and aliases that can be used for deployment. A second version was later created from the same trained model so Docker could fetch its artifacts through the MLflow server.

## Question 2

MLflow aliases such as `champion` and `challenger` replace the old built-in `Staging` and `Production` stages. Separate model versions let us promote or roll back a deployment without changing the original training runs, and an alias is flexible because it can be moved to any version while serving code continues to use the same URI. The `champion` alias currently points to version 2.

## Question 3

Loading `models:/food11@champion` lets MLflow resolve the selected registered version and load its full model package, instead of making the API depend on a machine-specific `.pth` path and manually recreated preprocessing code. To serve a newer version, I only need to move the `champion` alias to that version and restart the API; the serving code and Docker image do not need to change.

## Question 4

Copying `pyproject.toml` and `uv.lock` before the source code lets Docker cache the expensive dependency-installation layer. If only `serve.py` changes, Docker reuses the existing virtual environment and rebuilds only the small source-copy layer and the layers after it, which makes rebuilding much faster.

## Question 5

The multi-stage `food11-api:latest` image is 533,718,082 bytes (about 534 MB), while the measured single-stage comparison image is 946,226,044 bytes (about 946 MB). The multi-stage build is therefore about 412.5 MB, or 43.6%, smaller. `docker history` showed that the main layer is the Python virtual environment: about 1.79 GB uncompressed in the multi-stage image, compared with a 3.25 GB uncompressed `uv sync` layer in the single-stage image. The single-stage image also keeps the approximately 63.9 MB `uv` installation layer, which the runtime stage does not need.

## Question 6

Without `.dockerignore`, Docker sends the datasets, Git history, local environment, MLflow database, and artifacts as build context, making transfers and builds much slower and potentially making the image larger if a broad `COPY` is used. The local Windows `.venv` is the most likely excluded folder to break a Linux image because its executables and paths are incompatible; large `data/` and `mlruns/` folders can also exhaust memory, disk, or build time even when they do not directly cause a code error.

## Question 7

Inside a container, `127.0.0.1` refers to the container itself, so it cannot reach the MLflow server listening on the Windows host at that address. On Docker Desktop, `host.docker.internal` resolves to the host through Docker's internal networking, allowing the container to use `http://host.docker.internal:5000`.

## Question 8

Yes. After stopping the first container and starting a new one from the same `food11-api:latest` image, `/health` returned `{"status":"ok"}` and `/predict` again returned a category and confidence without rebuilding the image. The API code and Python dependencies are baked into the image, while the model selected by `models:/food11@champion` is resolved and fetched from MLflow when the container starts.

## Question 9

The image still needs to be tagged and pushed to a container registry such as Docker Hub or GitHub Container Registry. A CI runner or Kubernetes deployment should reference an immutable image digest, with registry authentication and a build-and-push workflow, so another machine can pull the exact tested image instead of rebuilding an image that might differ.
