Question 1: Running uv init created pyproject.toml, which contains the project’s metadata, dependencies, and build configuration; .python-version, which specifies Python 3.11; README.md, which is used for project documentation; and src/mlops_lab_1/__init__.py, which initializes the Python package and contains its starter code.

Question 2:Running dvc init created .dvc/config to store DVC settings (currently empty), .dvc/.gitignore to exclude local configuration, temporary files, and cache from Git, and .dvcignore to specify files DVC should ignore. It also created .dvc/tmp/ for internal temporary files. The files .dvc/config, .dvc/.gitignore, and .dvcignore should be pushed to Git and are already tracked in your folder; temporary files, cache, and local credentials should not be pushed.

Question 3: With --global, credentials are stored in your user-level DVC configuration, located on Windows at %LOCALAPPDATA%\iterative\dvc\config, and apply to all your DVC projects. Other scopes are --local, which stores settings in the Git-ignored .dvc/config.local; the default project scope, which uses .dvc/config; and --system, which applies to all users on the computer. Credentials should never be pushed to GitHub; only non-secret project configuration should be committed.

Question 4: After running dvc add data, DVC added /data to your project’s .gitignore, so Git ignores the entire data folder. The actual data is managed by DVC, while Git tracks the data.dvc pointer file and .gitignore, keeping the large dataset out of the Git repository.

Question 5: Yes, the folder contains data.dvc. It points to the data folder and records its MD5 hash (a3a457d03c51ff8b037a833440f6ad13.dir), total size of 1,188,442,712 bytes, and 16,643 files. This small metadata file lets DVC identify the dataset version while Git tracks the pointer instead of the actual data.

Question 6: On GitHub’s main branch, the DVC configuration files are present, but the Python code has not been pushed yet. The actual data is not stored in GitHub; instead, data.dvc points to the data folder and identifies its version. The three test images were successfully pushed to DagsHub, where the actual data is stored and can be browsed through its web interface.

Question 7: No, i don't see the data, and the command to get them is dvc pull.

Question 8: After running git checkout 6483c34 and dvc checkout, the food11_processed and food11_processed_mini folders disappear because that commit tracks only the raw dataset. Running git checkout main followed by dvc checkout restores both processed folders. This shows how Git and DVC work together to restore matching versions of the code and data.