Question 1: `mlflow`, `torch`, `torchvision`, and `scikit-learn` were added to the dependencies in `pyproject.toml`. The `uv.lock` file was updated with the exact resolved versions of these packages and all their transitive dependencies so the environment can be reproduced.

Question 2: `--backend-store-uri sqlite:///mlflow.db` specifies the database for experiment and run metadata, including parameters, metrics, and run status. `--default-artifact-root ./mlruns` specifies where run artifacts, such as trained models and output files, are stored. Metadata describes a run; artifacts are files produced by it.

Question 3: They contain frequently changing, potentially large experiment outputs that MLflow already manages. Tracking them with Git or DVC would duplicate storage and complicate versioning. Git tracks the training code, DVC tracks the datasets, and MLflow manages the experiments.

Question 4: MLflow creates the experiment if it does not exist and selects it for subsequent runs. Calling mlflow.set_experiment("food11") creates a food11 experiment that should appear in the UI.

Question 5: mlflow.log_param records a fixed configuration chosen before training, such as the learning rate, batch size, or number of epochs. mlflow.log_metric records numerical results produced during training, such as loss and accuracy. Metrics use step to associate each value with a particular epoch so MLflow can display their evolution as charts. Parameters remain fixed, so they do not need a step.

Question 6: In the successful valuable-bird-20 run, the Parameters section contains settings such as lr=0.001, batch_size=32, and epochs=5. The Metrics section contains charts for train_loss, val_loss, and val_accuracy, plus the final test_accuracy. The logged ResNet18 model appears in the run’s model section. Its files are stored locally under:
[`mlruns/1/models/m-ee1acd0a184145f2862f2b6b911c27fd/artifacts`](C:/Users/user/Desktop/Semestre 9/Machine learning ops/mlops-lab-1/mlruns/1/models/m-ee1acd0a184145f2862f2b6b911c27fd/artifacts)
The model artifact is approximately 44.8 MB.

Question 7: 0.01 produced the highest validation accuracy. A higher learning rate is not always better in general, although 0.01 was best among these tested values.

Question 8: Learning rate had the clearest effect. 0.0001 learned too slowly, while 0.001 and 0.01 performed much better. At lr=0.001, batch size 32 performed slightly better than 64.

Question 9: The best run is mysterious-bat-439, with run ID 689a6e73e08f4e8bb2e1e3a21fbff804.
