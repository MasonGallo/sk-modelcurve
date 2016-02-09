sk-modelcurves
==============

A Python wrapper built for software engineers and researchers to facilitate
easy creation of learning and validation curve plots from scikit-learn. 

The module is meant to complement your workflow in scikit-learn and ease the
process of evaluating your models. 

The module includes many quality of life features that should save you precious
time whenever you want to plot a learning curve to check for bias/variance or 
plot a validation curve to see the effect of tuning a hyperparameter.


Background
==========

For those not familiar with learning curves, check out Andrew Ng's excellent 
discussion of their use at http://cs229.stanford.edu/materials/ML-advice.pdf

Over the process of writing many research papers and building many models, I
found myself using boilerplate code that I would copy paste for almost every
project whenever I wanted to plot a learning curve or validation curve to
evaluate models.

Hopefully, this module will save you a few minutes each time you need to plot
a learning or validation curve so you can focus on other things.


Example Usage
=============

Generate a learning curve using accuracy as a metric and 5-fold cross validation.

Assumes a sklearn estimator called knn, training data matrix called X and
training labels called y.

   $ from sk_modelcurves.learning_curve import draw_learning_curve
   $ draw_learning_curve(knn, X, y, scoring='accuracy', cv=5)
   $ plt.show()
   
Generate multiple learning curves for several estimators using F1 score as a 
metric, 5-fold cross validation, and names for each of the estimators.

Assumes 3 sklearn estimators called knn2, knn20, knn40, training data matrix
called X and training labels called y.

   $ from sk_modelcurves.learning_curve import draw_learning_curve
   $ draw_learning_curve([knn2, knn20, knn40], X, y, scoring='f1', cv=5,
     estimator_titles=['2 Neighbors', '20 Neighbors', '40 Neighbors'])
   $ plt.show()

Many other options are available. Check out the source code docstrings or the
upcoming documentation.


Important Links
===============

- Official source code repo: https://github.com/MasonGallo/sk-modelcurve
- HTML documentation: coming soon!
- Issue tracker: https://github.com/MasonGallo/sk-modelcurve/issues


Dependencies
============

sk-modelcurves is tested to work for Python 2.6 and Python 2.7. Python 3.3+ has
not been tested and is assumed to not work until tested.

The required dependencies include scikit-learn (of course!), numpy >= 1.6.1,
and matplotlib >= 1.1.1.

To run tests, you will need nose >= 1.1.2.


Contributing
============

Anyone is welcome!

If you find a bug or would like to discuss a potential feature, please file an
issue first.


Testing
=======

After installation, you can launch the test suite from outside the source 
directory (you will need to have the ``nose`` package installed)::

   $ nosetests -v sk_modelcurves