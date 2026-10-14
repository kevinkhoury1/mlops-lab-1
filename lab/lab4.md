# Lab 4 - Orchestrating the app with Docker Compose

## Question 1

Without a volume, MLflow writes its database and artifacts into the container's writable layer. Removing that container removes the data with it, so a new container made from the same image shows an empty MLflow UI.

## Question 2

A named volume keeps MLflow's state outside the container and lets Docker manage its location and lifecycle, which is portable across host operating systems and avoids adding runtime data to the repository. A bind mount would also preserve the data and can be convenient for inspecting files during local development, but it couples the setup to a particular host path and its permissions.

## Question 3

Docker Compose creates a private network and embedded DNS for the project. The service name `mlflow` therefore resolves to the MLflow container on that network, while `host.docker.internal` was needed in Lab 3 because MLflow was running on the Windows host outside the container network.

## Question 4

Reading `INFERENCE_URL` from the environment keeps the frontend image reusable. Compose can set it to `http://inference:8000`, while a standalone frontend can use the default host address or receive another URL without changing or rebuilding the image.

## Question 5

The inference service does not publish a host port because only the frontend needs to call it. The frontend reaches `http://inference:8000` through the private Compose network, while ports 5000 and 8501 are published because a user needs to open the MLflow UI and Streamlit page from the host.

## Question 6

With basic `depends_on`, inference can start before MLflow is ready, fail while resolving `models:/food11@champion`, and exit. Its logs would show the MLflow connection or model-loading error. This Compose file adds an MLflow health check and waits for `service_healthy`; `restart: on-failure` provides an additional recovery path.

## Question 7

`docker compose ps` showed host mappings for MLflow (`0.0.0.0:5000->5000/tcp`) and the frontend (`0.0.0.0:8501->8501/tcp`). Inference showed only `8000/tcp`, with no host mapping. This matches the Compose file: users open the two web interfaces from the host, while the frontend reaches inference privately by service name.

## Question 8

The API continues using the old model because `serve.py` loads the model only once at process startup. After moving `champion` to another registered version, `docker compose restart inference` starts a new API process and loads the version currently selected by the alias.

## Question 9

Restarting works without rebuilding because the inference image contains the API code and Python dependencies, while the registered model selected by `models:/food11@champion` is fetched from MLflow at container startup.

## Question 10

After `docker compose down` followed by `docker compose up`, the registered `food11` model and the `champion` alias pointing to version 3 were still present because the named volume remained. After `docker compose down -v`, the new MLflow server returned an empty model registry, and inference failed with `RESOURCE_DOES_NOT_EXIST: Registered Model with name=food11 not found`. The `-v` option deleted the named volume that held both the SQLite database and model artifacts. The saved volume data was restored after this test so the final stack is working again.

## Question 11

Docker Compose only orchestrates containers on one machine. Running replicas across machines with load balancing, rescheduling, health-based recovery, and highly available persistent storage requires a cluster orchestrator such as Kubernetes or Docker Swarm, plus an external database and shared object storage for MLflow.
