import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from xgboost import *
import fasttreeshap
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder

np.random.seed(1)

# In[2]: CellPhenoX Class


class CellPhenoX:
    def __init__(self, X, y, CV_repeats, outer_num_splits, inner_num_splits):
        """_summary_

        Args:
            X (dataframe): cell by latent dimensions dataframe
            y (series): the target variable
            CV_repeats (int): number of times to repeat the cross-validation
            outer_num_splits (int): number of outer folds (stratified k fold)
            inner_num_splits (int): number of inner folds (for hyperparameter tuning)
        """
        # self.steps = ["hyperparameter tuning", "model training", "model prediction", "performance eval", "SHAP value calculation"]
        # self.CV_repeat_cumulative_times = [0,0,0,0,0]
        # self.CV_repeat_times = []
        self.X = X
        self.y = y
        # self.fc = fc
        # self.num_samp = num_samp
        self.total_time = 0
        self.CV_repeats = CV_repeats
        self.outer_num_splits = outer_num_splits
        self.inner_num_splits = inner_num_splits
        # Make a list of random integers between 0 and 10000 of length = CV_repeats to act as different data splits
        self.random_states = np.random.randint(10000, size=CV_repeats)
        self.param_grid_RF = [
            {
                "max_features": ["sqrt", "log2"],
                "max_depth": [10, 20, 30],
                "min_samples_leaf": [1, 2, 5],
                "min_samples_split": [2, 5, 10],
                "n_estimators": [100, 200, 800],
            }
        ]
        self.param_grid_LR = [{
                        'C': np.logspace(-4, 4, 20), # regularization space (1e-4=strong to 1e4=weak)
                        'penalty': ['l1', 'l2', 'elasticnet'],
                        'l1_ratio': [0.1, 0.5, 0.9]  # only used if penalty == 'elasticnet'
                        }]
        self.label_encoder = LabelEncoder()
        self.best_model_LR = None
        self.best_model_RF = None
        self.best_score_LR = float("-inf")
        self.best_score_RF= float("-inf")
        self.shap_values_per_cv_LR = dict()
        self.shap_values_per_cv_RF = dict()
        self.shap_df_LR = None
        self.shap_df_RF = None
        for sample in X.index:
            ## Create keys for each sample
            self.shap_values_per_cv_LR[sample] = {}
            self.shap_values_per_cv_RF[sample] = {}
            ## Then, keys for each CV fold within each sample
            for CV_repeat in range(self.CV_repeats):
                self.shap_values_per_cv_LR[sample][CV_repeat] = {}
                self.shap_values_per_cv_RF[sample][CV_repeat] = {}

    def split_data(self, train_outer_ix, test_outer_ix):
        X_train_outer, X_test_outer = (
            self.X.iloc[train_outer_ix, :],
            self.X.iloc[test_outer_ix, :],
        )
        y_train_outer, y_test_outer = (
            self.y.iloc[train_outer_ix],
            self.y.iloc[test_outer_ix],
        )

        # Create an additional inner split for validation
        X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
            X_train_outer,
            y_train_outer,
            test_size=0.2,
            random_state=42,
            stratify=y_train_outer,
        )
        y_train_inner = self.label_encoder.fit_transform(y_train_inner)
        y_train_outer = self.label_encoder.fit_transform(y_train_outer)
        y_test_outer = self.label_encoder.transform(y_test_outer)
        y_val_inner = self.label_encoder.transform(y_val_inner)

        return [
            X_train_outer,
            X_test_outer,
            X_train_inner,
            X_val_inner,
            y_train_inner,
            y_train_outer,
            y_test_outer,
            y_val_inner,
        ]

    def model_training_shap_val(self, outpath, num_cores, model_type, shap_type="Fast"):
        """Train the model using nested cross validation strategy and generate shap values for each fold/CV repeat

        Parameters:
        outpath (str): the path for the output folder
        num_cores (int): number of cores to use for inner training
        model_type (str): LogReg or RandomForest to choose the model to use
        shap_type (str): Options are OG, Approx, and Fast
        

        Returns:


        """

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        accuracy_list = []
        y_test_combined_list, y_prob_combined_list = [], []
        val_auc_combined_list, val_prc_combined_list = (
            [],
            [],
        )

        # save the models
        overal_model_list = []

        print("entering CV loop")
        for i, CV_repeat in enumerate(range(self.CV_repeats)):
            # Verbose
            print("\n------------ CV Repeat number:", CV_repeat + 1)
            # Establish CV scheme
            CV = StratifiedKFold(
                n_splits=self.outer_num_splits,
                shuffle=True,
                random_state=self.random_states[i],
            )

            ix_training, ix_test = [], []
            # Loop through each fold and append the training & test indices to the empty lists above
            for fold in CV.split(self.X, self.y):
                ix_training.append(fold[0]), ix_test.append(fold[1])

            y_prob_list = []
            y_test_list = []
            val_auc_list = []
            val_accuracy_list = []
            val_prc_list = []
            model_list = []

            ## Loop through each outer fold and extract SHAP values
            for j, (train_outer_ix, test_outer_ix) in enumerate(
                zip(ix_training, ix_test)
            ):
                # Verbose
                print("\n------ Fold Number:", j + 1)
                (
                    X_train_outer,
                    X_test_outer,
                    X_train_inner,
                    X_val_inner,
                    y_train_inner,
                    y_train_outer,
                    y_test_outer,
                    y_val_inner,
                ) = self.split_data(train_outer_ix, test_outer_ix)
                ## Establish inner CV for parameter optimization #-#-#
                cv_inner = StratifiedKFold(
                    n_splits=self.inner_num_splits, shuffle=True, random_state=1
                )  # -#-#

                # Search to optimize hyperparameters
                # hp_start = time.time()
                if model_type == "RandomForest":
                    model = RandomForestClassifier(random_state=10, class_weight="balanced")
                    search = RandomizedSearchCV(
                        model,
                        self.param_grid_RF,
                        scoring="balanced_accuracy",
                        cv=cv_inner,
                        refit=True,
                        n_jobs=num_cores,
                    )  # -#-#
                    model_use_quick = "RF"
                elif model_type == "LogReg":
                    model = LogisticRegression(solver='saga', max_iter=10000, class_weight="balanced")
                    search = RandomizedSearchCV(
                        model,
                        self.param_grid_LR,
                        scoring="balanced_accuracy",
                        cv=cv_inner,
                        refit=True,
                        n_jobs=num_cores,
                    )  # -#-#
                    model_use_quick = "LR"
                    
                else:
                    raise Exception("Model_Type must be either RandomForest or LogReg")
                
                # fit the results with the above search parameters using the inner training data
                result = search.fit(X_train_inner, y_train_inner)  # -#=#

                # Fit the best hyperameritized model on the FULL training data
                result.best_estimator_.fit(X_train_outer, y_train_outer)  # -#-#

                # Make predictions on the test set using the best model
                y_pred = result.best_estimator_.predict(X_test_outer)
                y_prob = result.best_estimator_.predict_proba(X_test_outer)  # [:, 1]
                y_prob_list.append(y_prob)
                y_test_list.append(y_test_outer)

                # Calculate accuracy
                accuracy = accuracy_score(y_test_outer, y_pred)
                print("--- Accuracy: ", accuracy)
                accuracy_list.append(accuracy)
                y_val_pred = result.best_estimator_.predict(X_val_inner)
                val_accuracy = accuracy_score(y_val_inner, y_val_pred)
                # val_accuracy_list.append(val_accuracy_list)
                val_accuracy_list.append(val_accuracy)

                # Calculate ROC curve and AUC
                num_classes = len(np.unique(y_train_outer))
                if num_classes == 2:
                    # fpr, tpr, thresholds = roc_curve(y_test_outer, y_pred)
                    # auc_value = auc(fpr, tpr)
                    # auc_list.append(auc_value)
                    # Validate on the validation set
                    y_prob_val = result.best_estimator_.predict_proba(X_val_inner)[:, 1]
                    fpr_val, tpr_val, _ = roc_curve(y_val_inner, y_prob_val)
                    val_auc = auc(fpr_val, tpr_val)
                    val_prc = average_precision_score(y_val_inner, y_prob_val)
                else:  # more than 2 classes
                    val_auc = []
                    val_prc = []
                    for k in range(num_classes):
                        # fpr, tpr, _ = roc_curve(y_test_outer == k, y_prob[:, k])
                        # auc_value = auc(fpr, tpr)
                        # auc_list.append(auc_value)
                        val_fpr, val_tpr, _ = roc_curve(
                            y_val_inner == k,
                            result.best_estimator_.predict_proba(X_val_inner)[:, k],
                        )
                        val_auc.append(auc(val_fpr, val_tpr))
                        val_prc.append(
                            average_precision_score(
                                y_val_inner == k,
                                result.best_estimator_.predict_proba(X_val_inner)[:, k],
                            )
                        )

                val_auc_list.append(val_auc)
                # print(len(val_auc_list))
                val_prc_list.append(val_prc)
                val_auc_combined_list.append(val_auc)
                val_prc_combined_list.append(val_prc)
                # val_accuracy_list.append(val_accuracy)
                print(
                    "--- Validation Accuracy: ",
                    val_accuracy,
                    " - Validation AUROC: ",
                    val_auc,
                    " - Val AUPRC: ",
                    val_prc,
                )
            
                if model_type == "RandomForest":
                    if shap_type == "OG":
                        start = time.time()
                        explainer = shap.TreeExplainer(result.best_estimator_)
                        shap_values = explainer.shap_values(X_test_outer)
                        tottime = time.time()-start
                        self.total_time = self.total_time + tottime
                    elif shap_type == "Approx":
                        start = time.time()
                        explainer = shap.TreeExplainer(result.best_estimator_)
                        shap_values = explainer.shap_values(X_test_outer, approximate=True)
                        tottime = time.time()-start
                        self.total_time = self.total_time + tottime
                    elif shap_type == "Fast":
                        start = time.time()
                        explainer = fasttreeshap.TreeExplainer(result.best_estimator_, algorithm="auto", n_jobs=num_cores) # n_jobs=-1 for parallel processing
                        shap_values = explainer(X_test_outer).values
                        tottime = time.time()-start
                    
                elif model_type == "LogReg":
                    n_sample_use = max(int(X_train_outer.shape[0] * 0.3), 1000)
                    start = time.time()
                    explainer = shap.LinearExplainer(result.best_estimator_, X_train_outer, #feature_perturbation='correlation_dependent', 
                    nsamples=n_sample_use)
                    shap_values = explainer.shap_values(X_test_outer)
                    tottime = time.time()-start
                print("Time for getting Shaps (min):", tottime/60)
                print("Shap values shape", shap_values.shape, "0 shape", shap_values[0].shape)

                # Extract SHAP information per fold per sample
                # Shap > 0.39 
                # General Explainer --> Shape (n_samples, n_features) for binary classification and  (n_samples, n_features, n_classes) for multiclass classification
                # Using shap.TreeExplainer, etc. --> Maybe Shape (n_classes) where each is (n_samples, n_feautres)
                if float(shap.__version__.split(".")[1]) > 39:
                    if num_classes == 2:
                        # if there are 3 entires for length then the last is the class
                        if shap_values.ndim == 3:
                            # Get the values for the positive class: 
                            shap_values_pos_class = shap_values[:, :, 1]
                            for k, test_index in enumerate(test_outer_ix):
                                test_index = self.X.index[test_index]
                                # Save the SHAP vector for that sample
                                getattr(self, f"shap_values_per_cv_{model_use_quick}")[test_index][CV_repeat] = shap_values_pos_class[k]
                        elif shap_values.ndim == 2:
                            for k, test_index in enumerate(test_outer_ix):
                                test_index = self.X.index[test_index]
                                getattr(self, f"shap_values_per_cv_{model_use_quick}")[test_index][CV_repeat] = shap_values[k]

                    else:
                        # UNTESTED
                        for k, test_index in enumerate(test_outer_ix):
                            predicted_class = y_pred[k]
                            # Get the values for the positive class: 
                            shap_values_pos_class = np.array([sv[:, predicted_class] for sv in shap_values])
                            test_index = self.X.index[test_index]
                            getattr(self, f"shap_values_per_cv_{model_use_quick}")[test_index][CV_repeat] = shap_values_pos_class[k]
                else:
                    for k, test_index in enumerate(test_outer_ix):
                        test_index = self.X.index[test_index]
                        if num_classes == 2:
                            getattr(self, f"shap_values_per_cv_{model_use_quick}")[test_index][CV_repeat] = shap_values[1][k]
                        else:
                            predicted_class = y_pred[k]
                            getattr(self, f"shap_values_per_cv_{model_use_quick}")[test_index][CV_repeat] = shap_values[
                            predicted_class][k]
                            
                # save best model
                model_list.append(result.best_estimator_)

            # one ROC curve for each repeat
            # y_prob_combined = np.concatenate(y_prob_list)
            y_prob_combined = (
                np.concatenate([prob[:, 1] for prob in y_prob_list])
                if num_classes == 2
                else np.concatenate(y_prob_list)
            )
            y_test_combined = np.concatenate(y_test_list)
            y_prob_combined_list.append(y_prob_combined)
            y_test_combined_list.append(y_test_combined)

            # val_accuracy_combined = val_accuracy_list
            # val_accuracy_combined_list.append(val_accuracy_combined)
            # val_prc_combined = val_prc_list
            # val_prc_combined_list.append(val_prc_combined)

            precision, recall, _ = precision_recall_curve(
                y_test_combined, y_prob_combined
            )
            prc_auc = average_precision_score(y_test_combined, y_prob_combined)

            # BINARY if statement here?
            val_auc_combined = val_auc_list
            val_auc_combined_list.append(val_auc_combined)
            # Compute ROC and PR curve for the combined data
            if num_classes == 2:
                fpr, tpr, _ = roc_curve(y_test_combined, y_prob_combined)
                # Compute AUC (Area Under the Curve)
                roc_auc = auc(fpr, tpr)
                # Plot the ROC and precision recall curve for the current fold size
                axes[0].plot(
                    fpr, tpr, lw=2, label=f"CV Repeat {i+1} (ROC = {roc_auc:.2f})"
                )
            else:
                # MULTICLASS else statement Here?
                for k in range(num_classes):
                    fpr, tpr, _ = roc_curve(
                        y_test_combined == k, y_prob_combined[:, k]
                    )  # one vs rest..
                    roc_auc = auc(fpr, tpr)
                    axes[0].plot(
                        fpr,
                        tpr,
                        lw=2,
                        label=f"CV Repeat {i+1} - Class {k} (ROC = {roc_auc:.2f})",
                    )

            # Precision Recall curve
            axes[1].plot(
                recall, precision, lw=2, label=f"CV Repeat{i+1} (PRC = {prc_auc:.2f})"
            )
            # save the best model for this repeat
            model_pr_pairs = list(
                zip(model_list, val_prc_combined_list)
            )  # val_prc_combined

            # Find the model with the highest precision-recall score
            best_model_repeat, best_score_repeat = max(
                model_pr_pairs, key=lambda x: x[1]
            )
            overal_model_list.append((best_model_repeat, best_score_repeat))

        # Add labels and show the ROC curve plot
        axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(
            'Receiver Operating Characteristic\naggregated over folds for each repeat'
        )
        axes[0].legend(loc="lower right")

        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel("Precision")
        axes[1].set_ylabel("Recall")
        axes[1].set_title(
            'Precision-Recall Curve \naggregated over folds for each repeat'
        )
        axes[1].legend(loc="lower right")

        y_prob_combined_repeat = np.concatenate(y_prob_combined_list)
        y_test_combined_repeat = np.concatenate(y_test_combined_list)

        predicted_positive = y_prob_combined_repeat[y_test_combined_repeat == 1]
        predicted_negative = y_prob_combined_repeat[y_test_combined_repeat == 0]
        axes[2].hist(
            predicted_negative,
            label="Negative Class",
            bins=30,
            color="#f09102",
            alpha=0.5,
        )
        axes[2].hist(
            predicted_positive,
            label="Positive Class",
            bins=30,
            color="#2d5269",
            alpha=0.5,
        )
        axes[2].set_xlabel("Predicted Probabilities")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title('Distribution of Predicted Probabilities using model')
        axes[2].legend(loc="upper right", labels=["Negative Class", "Positive Class"])
        plt.suptitle(f'Classification Model Performance Evaluation for model {model_type}')
        plt.tight_layout()
        plt.savefig(f"{outpath}modelperformance.pdf", format="pdf")

        # print(f"val_auc_combined_list length: {len(val_auc_combined_list)}")
        # print(f"first list of val_auc_combined_list length: {len(val_auc_combined_list[0])}")
        # print(f"val_prc_combined_list length: {len(val_prc_combined_list)}")
        # print(f"first list of val_prc_combined_list length: {len(val_prc_combined_list[0])}")
        # val_auc_combined_repeat = np.concatenate(val_auc_combined_list)
        # val_prc_combined_repeat = np.concatenate(val_prc_combined_list)

        # avg_val_auc = np.mean(val_auc_combined_list)
        # avg_val_prc = np.mean(val_prc_combined_list)
        # print(f"Average AUROC: {avg_val_auc} | Average AUPRC: {avg_val_prc}")

        # select the final model
        best_model, best_score = max(overal_model_list, key=lambda x: x[1])
        setattr(self, f"best_model_{model_use_quick}", best_model)
        setattr(self, f"best_score_{model_use_quick}", best_score)
        value = getattr(self, f"best_score_{model_use_quick}")
        print(f"best model precision-recall score = {value:.4f}")

        # now aggregate the shap values per CV
        self.get_shap_values(outpath, model_use_quick)
        # and calculate the interpretable score
        self.get_interpretable_score(model_use_quick)

    def get_shap_values_per_cv(self, model_use_quick):
        return getattr(self, f"shap_values_per_cv_{model_use_quick}")

    def get_best_score(self):
        return getattr(self, f"best_score_{model_use_quick}")

    def get_best_model(self):
        return getattr(self, f"best_model_{model_use_quick}")

    def get_shap_values(self, outpath, model_use_quick):
        average_shap_values = []
        for i in range(0, len(self.X)):  # len(NAM)
            id = self.X.index[i]  # NAM.index[i]
            # print(id)
            # print(self.shap_values_per_cv)
            df_per_obs = pd.DataFrame.from_dict(
                getattr(self, f"shap_values_per_cv_{model_use_quick}")[id][0]
            )  # Get all SHAP values for sample number i
            # Get relevant statistics for every sample
            average_shap_values.append(df_per_obs.mean(axis=1).values)
        df_add = pd.DataFrame(
            average_shap_values, columns=[f"{col}_shap" for col in self.X.columns]
        )
        setattr(self, f"shap_df_{model_use_quick}", df_add)
        setattr(self, f"shap_df_{model_use_quick}", getattr(self, f"shap_df_{model_use_quick}").set_index(self.X.index))
        # plot the SHAP summary plot?
        plt.figure()
        shap.summary_plot(np.array(average_shap_values), self.X, show=False)
        plt.title("Average SHAP values after nested cross-validation")
        plt.savefig(f"{outpath}SHAPsummary.png")

    def get_interpretable_score(self, model_use_quick):
        # Calculate the SHAP-adjusted probability score
        shap_df = getattr(self, f"shap_df_{model_use_quick}")
        interpretable_score = np.sum(shap_df, axis=1)
        # add t0 the shap_df
        shap_df["interpretable_score"] = interpretable_score
