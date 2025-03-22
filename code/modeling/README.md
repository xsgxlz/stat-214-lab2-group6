### Modeling

The **modeling** directory contains the code for the predictive modeling and post-hoc analysis of the models. The function of each file is as follows:

Autoencoder.py: Provides the Autoencoder code for training the model.

classification.py: Provides utility functions for classification tasks.

classification_*.ipynb: Search for the best hyperparameters for the corresponding classification model.

classification_eval_*.ipynb: Evaluate the corresponding classification model by using the best hyperparameters.

preprocessing.py: Provides utility functions for preprocessing the data.

preprocessing.ipynb: Preprocess the data and save the preprocessed data.

train_autoencoder.py: Train the Autoencoder model.

The train_autoencoder.sh script under the code directory is designed to run the train_autoencoder.py script with all the possible hyperparameters simultaneously.


rf_xgb_modeling_and_analysis.ipynb: Perform end-to-end modeling and post-hoc analysis for the random forest and XGBoost approaches, starting with the data and autoencoder, training RF/XGBoost on various feature sets, hyperparameter search, and cross-validation and test set performance. It also contains post-hoc analysis, which includes feature importance, prediction visualization, variability checks, stability analysis (through perturbation), etc.