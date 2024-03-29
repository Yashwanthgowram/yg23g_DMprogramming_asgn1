import numpy as np
from numpy.typing import NDArray
from typing import Any

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        
        # Enter code and return the `answer`` dictionary

        answer = {}
        counts_train = np.unique(ytrain, return_counts=True)
        counts_test = np.unique(ytest, return_counts=True)
        print("counts_train: ", counts_train)
        print("counts_test: ", counts_test)
        print()

        self.is_int = nu.check_labels(ytrain)
        self.is_int = nu.check_labels(ytrain)
        self.dist_dict = self.analyze_class_distribution(
            ytrain.astype(np.int32)
        )  
        # Train a Logistic regression Classifier
        clf = LogisticRegression(
                random_state=self.seed, multi_class="multinomial", max_iter=300
            )
        clf.fit(Xtrain, ytrain)
        
        # Predict probabilities for training and test sets
        ytrain_pred = clf.predict_proba(Xtrain)
        ytest_pred = clf.predict_proba(Xtest)

        # Calculate top-k accuracy scores
        topk = [k for k in range(1, 6)]
        plot_scores_test = []
        plot_scores_train = []

        for k in topk:
            topk_dict = {}
            nb_unique, counts = np.unique(ytest, return_counts=True)

            # Calculate top-k accuracy score for both training and test sets
            score_train = top_k_accuracy_score(ytrain, ytrain_pred, normalize=True, k=k)
            score_test = top_k_accuracy_score(ytest, ytest_pred, normalize=True, k=k)
            topk_dict["score_train"] = score_train
            topk_dict["score_test"] = score_test
            answer[k] = topk_dict
            plot_scores_test.append((k, score_test))
            plot_scores_train.append((k, score_train))

        # Store the trained classifier in the answer dictionary
        answer["clf"] = clf

        # Store the k vs. score plots in the answer dictionary
        answer["plot_k_vs_score_train"] = plot_scores_train
        answer["plot_k_vs_score_test"] = plot_scores_test

        #plt.plot(Xtrain, ytrain)
        #plt.plot(Xtest, ytest)
        #plt.show()

        # explanation
        answer["text_rate_accuracy_change"] = "The accuracy of the model on testing data consistently improves as k increases, reflecting its heightened proficiency in predicting the top-k classes."
        answer["text_is_topk_useful_and_why"] = "Beyond the scope of traditional accuracy, the top-k accuracy metric is valuable for assessing performance, indicating the model's effectiveness in capturing relevant patterns and making accurate predictions across a more diverse set of probable classes."
        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        answer = {}
        """
        seven_nine_idx = (y == 7) | (y == 9)
        X = X[seven_nine_idx, :]
        y = y[seven_nine_idx]

        frac_to_remove = 0.9
        X, y = nu.filter_9s_convert_to_01(X, y, frac=frac_to_remove)
        Xtest, ytest = nu.filter_9s_convert_to_01(
            Xtest, ytest, frac=frac_to_remove
        )
        """
        #X,y,Xtest,ytest = u.prepare_data()
        X,y = nu.filter_imbalanced_7_9s(X, y)
        Xtest,ytest = nu.filter_imbalanced_7_9s(Xtest, ytest)

        Xtrain_test = nu.scale_data(X)
        Xtest_test = nu.scale_data(Xtest)

        # Checking that the labels are integers
        ytrain_test = nu.scale_data_1(y)
        ytest_test = nu.scale_data_1(ytest)

        print("3(B) - Are elements in Xtrain a floating point number and scaled between 0 to 1:" +str(Xtrain_test))
        print("3(B) - Are elements in a floating point number and scaled between 0 to 1:" +str(Xtest_test))
        print("3(B) - Are elements in ytrian an integer:" +str(ytrain_test))
        print("3(B) - Are elements in ytest an integer:" +str(ytest_test))

        # Answer is a dictionary with the same keys as part 1.B
        answer["length_X"] = len(X)
        answer["length_Xtest"] = len(Xtest)
        answer["length_y"] = len(y)
        answer["length_ytest"] = len(ytest)
        answer["max_X"] = np.max(X)
        answer["max_Xtest"] = np.max(Xtest)
        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
    
        # Enter your code and fill the `answer` dictionary
        n_splits = 5
        clf = SVC(random_state=self.seed)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        def cross_validate_metric(score_type: str):
            score = ["accuracy", "recall", "precision", "f1"]
            cv_scores = cross_validate(
                clf, X, y, scoring=score, cv=cv, return_train_score=False
            )
            #cv_scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
            scores = {
                "mean_accuracy": cv_scores["test_accuracy"].mean(),
                "mean_recall": cv_scores["test_recall"].mean(),
                "mean_precision": cv_scores["test_precision"].mean(),
                "mean_f1": cv_scores["test_f1"].mean(),
                "std_accuracy": cv_scores["test_accuracy"].std(),
                "std_recall": cv_scores["test_recall"].std(),
                "std_precision": cv_scores["test_precision"].std(),
                "std_f1": cv_scores["test_f1"].std(),
            }
            return scores

        # scores_macro = cross_validate_metric(score_type="macro")
        scores = cross_validate_metric(score_type="macro")

        # Train on all the data
        clf.fit(X, y)

        ytrain_pred = clf.predict(X)
        ytest_pred = clf.predict(Xtest)
        conf_mat_train = confusion_matrix(y, ytrain_pred)
        conf_mat_test = confusion_matrix(ytest, ytest_pred)

        answer = {}
        answer["scores"] = scores
        answer["cv"] = cv
        answer["clf"] = clf
        answer["is_precision_higher_than_recall"] = (
            scores["mean_precision"] > scores["mean_recall"]
        )
        answer["explain_is_precision_higher_than_recall"] = "On average, the model's accuracy is higher in positive predictions, with fewer false positives, than in its ability to capture all positive instances, where a tendency for more false negatives is evident."
        answer["confusion_matrix_train"] = conf_mat_train  
        answer["confusion_matrix_test"] = conf_mat_test  

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        
        # Enter your code and fill the `answer` dictionary

        n_splits = 5
        clf = SVC(random_state=self.seed, class_weight="balanced")
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        def cross_validate_metric(score_type: str):
            score = ["accuracy", "recall", "precision", "f1"]
            cv_scores = cross_validate(
                clf, X, y, scoring=score, cv=cv, return_train_score=False
            )

            scores = {
                "mean_accuracy": cv_scores["test_accuracy"].mean(),
                "mean_recall": cv_scores["test_recall"].mean(),
                "mean_precision": cv_scores["test_precision"].mean(),
                "mean_f1": cv_scores["test_f1"].mean(),
                "std_accuracy": cv_scores["test_accuracy"].std(),
                "std_recall": cv_scores["test_recall"].std(),
                "std_precision": cv_scores["test_precision"].std(),
                "std_f1": cv_scores["test_f1"].std(),
            }
            return scores

        scores = cross_validate_metric(score_type="macro")

        # Train on all the data
        clf.fit(X, y)

        ytrain_pred = clf.predict(X)
        ytest_pred = clf.predict(Xtest)
        conf_mat_train = confusion_matrix(y, ytrain_pred)
        conf_mat_test = confusion_matrix(ytest, ytest_pred)

        answer = {}
        answer["scores"] = scores
        answer["cv"] = cv
        answer["clf"] = clf
        answer['class_weights']= compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        
        answer["confusion_matrix_train"] = conf_mat_train  
        answer["confusion_matrix_test"] = conf_mat_test  
        answer["explain_purpose_of_class_weights"] = "The employment of class weights is geared towards alleviating class imbalance by applying increased penalties to misclassifications of the minority class.."
        answer["explain_performance_difference"] = "The performance difference noted when employing class weights reflects the model's improved capacity to generalize to the minority class, leading to more balanced performance metrics across all classes."
        return answer
