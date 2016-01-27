from sklearn.learning_curve import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np

def draw_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                       train_sizes=np.linspace(.2,1.0,10), n_jobs=1):
    """Create a learning curve to help us determine if we are
    overfitting or underfitting.
    TODO: specify parameters and clean this up
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    # TODO: add argument allowing for % of train examples
    plt.xlabel("Number of training examples used")
    # TODO: needs to accept many metrics
    plt.ylabel("Accuracy")
    # curve plotting
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    # add grid and legend
    plt.grid()
    plt.legend(loc="best")
    return plt
