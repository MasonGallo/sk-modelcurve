from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import collections


def draw_learning_curve(estimator, X, y, ylim=None, cv=None, scoring=None,
                        train_sizes=np.linspace(.2, 1.0, 5),
                        train_axis='n_examples', estimator_titles=None,
                        n_jobs=1):
    """
    Create a learning curve to help us determine if we are overfitting or
    underfitting. This is a wrapper over sklearn's excellent learning_curve
    function that adds useful capabilities like multiple estimators and
    automatically creates a plot, hopefully reducing boilerplate code. Returns
    a matplotlib.pyplot object. Don't forget to call plt.show() if needed.
    
    Parameters
    ----------
    estimator: sklearn estimator object type that implements "fit" and "predict"
               as expected from sklearn, or array-like of estimator objects
               Use an array-like if you want to pass multiple estimators to the
               same plot. If passing multiple estimators, scoring must be
               specified.
    
    X: array-like, shape (n_samples, n_features)
       Training vector, where n_samples is the number of samples and
       n_features is the number of features.
    
    y: array-like, shape (n_samples) or (n_samples, n_features), optional
       Target relative to X for classification or regression;
       None for unsupervised learning.
       
    ylim: tuple, shape (ymin, ymax), optional
          Defines min and max y-values plotted

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` used. In all
        other cases, :class:`KFold` is used.
    
    scoring: string, indicating sklearn scoring nickname from sklearn.metrics
        This will be the name of the y-axis on the returned plot if specified
    
    train_sizes: array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    
    train_axis: string, either 'n_examples' or 'per_examples'
        Indicates what should be used on the x-axis of the returned plot.
    
    estimator_titles: array_like shape (n_estimators)
        Indicates the title for each estimator to be used in plotting and added
        to the legend. This is useful to distinguish between multiple models on
        the same learning curve. This should always be specified when using 
        multiple estimators on the same curve; otherwise, the plot will be hard
        to read. (default: None)

    n_jobs: integer, optional
            Number of jobs to run in parallel. (default: 1)

    Returns
    -------
    plt: matplotlib.pyplot object for further editing by user. Don't forget to
         use plt.show() or matplotlib inline if necessary.
    """
    # TODO: test cases / error checking
    plt.figure()

    if ylim is not None:
        plt.ylim(*ylim)
        
    # copy sizes in case we need them later
    per_examples = np.copy(train_sizes)

    if train_axis == 'n_examples':
        plt.xlabel('Number of training examples used')
    elif train_axis == 'per_examples':     
        plt.xlabel('Percent of training examples used')
    if scoring is not None:
        plt.ylabel(scoring.capitalize())
    # we don't want users unintentionally using default scoring
    else:
        raise TypeError('scoring argument must be specified with a string\
        indicating the name of the scoring function, such as accuracy')
        

    # if multiple estimators passed
    if isinstance(estimator, (collections.Sequence, np.ndarray)):
        if not isinstance(estimator_titles, (collections.Sequence, np.ndarray)):
            raise TypeError('When giving an array of estimators,you must\
            specify names for each of the estimators with estimator_titles')
        # get number estimators for color setting
        N = len(estimator)
        color = plt.cm.rainbow(np.linspace(0,1,N))
        for ind, est in enumerate(estimator):
            train_sizes, train_scores, test_scores = learning_curve(
                est, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring,
                train_sizes=train_sizes)
            _plot_lc(train_sizes, train_scores, test_scores, train_axis,
                     scoring, per_examples, color=color[ind], 
                     est_title=estimator_titles[ind])


    # if only 1 estimator
    else:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring,
            train_sizes=train_sizes)
        _plot_lc(train_sizes, train_scores, test_scores, train_axis, scoring,
                 per_examples)

    # add grid and legend
    plt.grid()
    plt.legend(loc="best")
    return plt

def _plot_lc(train_sizes, train_scores, test_scores, train_axis, scoring,
             per_examples, color=None, est_title=None, std_bands=True):
    """
    Plot the learning curve. Helper function for draw_learning_curve below.
    This function adds the actual score curves to the plt object.

    Parameters
    ----------
    train_sizes: array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    
    train_scores: array-like, indicating the scores on the training set.
        This is passed from the sklearn function learning_curve.

    test_scores: array-like, indicating the scores on the testing set.
        This is passed from the sklearn function learning_curve.

    train_axis: string, either 'n_examples' or 'per_examples'
        Indicates what should be used on the x-axis of the returned plot.
    
    scoring: string, indicating sklearn scoring nickname from sklearn.metrics
        This will be the name of the y-axis on the returned plot if specified

    per_examples: array-like, indicating the percentages to be used on x-axis.
        This is sourced from train_sizes but in percentage form rather than
        counts.

    color: cm.rainbow object, providing color choices for multiple curves.
        This is only used when multiple estimators are passed.

    est_title: string, indicating the name of a particular estimator to be 
        used on the legend. This is only used when multiple estimators are
        passed.

    std_bands: boolean, indicating whether standard deviation bands should
        be drawn on the plot. Defaults to True.

    Returns
    -------
    plt object with curves added
    """
    # account for percentage train size
    # this is dirty but works
    if train_axis == 'per_examples':
        train_sizes = per_examples
    # convert regression scoring to positive to make easier to present
    if scoring in ['mean_absolute_error', 'mean_squared_error',
                   'median_absolute_error']:
        train_scores, test_scores = train_scores * -1.0, test_scores * -1.0
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if est_title is None:
        plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'x-', color="g",
                 label="Cross-validation score")
        if std_bands:   
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color='r')
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color='g')
    else:
        plt.plot(train_sizes, train_scores_mean, 'o--',
                 label=est_title + " Training score", color=color)
        plt.plot(train_sizes, test_scores_mean, 'x-',
                 label=est_title + " Cross-validation score", color=color*0.99)
        if std_bands:   
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color=color)
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color=color*0.99)
    return plt
