import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from simulate import bayes_boundary
from estimator import estimate_profit
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV

import seaborn as sns


color_markers_0 = 'blue'
color_markers_1 = 'red'
alpha_markers = 0.7
bayes_cmap = 'jet'
color_levels = 'black'

def make_meshgrid(x, y, z=None, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1

    if z is None:
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    else:
        z_min, z_max = z.min() - 1, z.max() + 1
        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h),
                                 np.arange(z_min, z_max, h))
        return xx, yy, zz


def plot_two_variables(df, x_name = 'col_0', y_name= 'col_1', target='y', fig=None, ax=None, plot_observations=True,
                       plot_bayes_colourmap=False, plot_separation_boundary=False, threshold=0.5,
                       **bayes_kwargs):
    """
    Plots the dependency of the two variables, with different colors based on the category they belong to.
    If plot_bayes_colourmap is True, it will plot the bayesian boundary that was used to generate the targets, provided
    that the boundary function is passed in the **bayes_kwargs.
    Note: this works only if the data is simulated!
    Args:
        df: pandas Dataframe containing the data
        x_name: (string) name of the column to be shown on the x-axis
        y_name: (string) name of the column to be shown on the y-axis
        target: (string or pandas Series) if the target variable is a column of the dataframe df, specify the name
            of the column (default = 'y'). Otherwise pass a pd.Series containing the classes.
        ax: matplotlib figure object
        ax: matplotlib axes object
        plot_observations: (boolean) flag to plot the observations
        plot_bayes_colourmap: (boolean) flag to plot the bayes probability colormap
        plot_separation_boundary: (boolean) flag to plot the class separation line
        threshold: (float) probability threshold to define the separation between the classes
        **bayes_kwargs: kwargs to be passed to the simulate.bayes_boundary function. Most common
            boundary_func (func): Python function defining the boundary
            max_accuracy (float): percentage of points that are not randomly swapped between class 0 and 1. If
            1., then no swap is happening, if 0, all the classes will be swapped
           **kwargs: other kwargs are the inputs to the boundary function, which are in most of the cases coordinates
    Returns:
        ax: matplotlib axes object
    """
    if type(target) == str:
        df_0 = df[df[target] == 0]
        df_1 = df[df[target] == 1]
    elif type(target) == pd.core.series.Series:
        df_0 = df[target == 0]
        df_1 = df[target == 1]
    else:
        raise AttributeError('target is expected to be of type string or pd.Series')

    x, y = df[x_name].values, df[y_name].values

    x_0, y_0 = df_0[x_name].values, df_0[y_name].values
    x_1, y_1 = df_1[x_name].values, df_1[y_name].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    if plot_observations:
        ax.scatter(x_0, y_0, color=color_markers_0, alpha=alpha_markers, marker='x')
        ax.scatter(x_1, y_1, color=color_markers_1, alpha=alpha_markers, marker='+')

    if plot_bayes_colourmap or plot_separation_boundary:
        xx, yy = make_meshgrid(x, y)
        probs, targets, coordinates = bayes_boundary(x=xx,
                                                    y=yy,
                                                    threshold=threshold, **bayes_kwargs)
    if plot_bayes_colourmap:
        pcm = ax.contourf(xx, yy, probs, cmap=bayes_cmap, alpha=0.3)
        if fig is not None: fig.colorbar(pcm, ax=ax, extend='both')
    if plot_separation_boundary:
        pcm2 = ax.contour(xx, yy, probs, linewidths=1.5, colors=color_levels,
                         levels=np.array([threshold - 0.0001, threshold + 0.0001]), alpha=1)


    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    return ax


def plot_clf_decision_boundary(df, x_name = 'col_0', y_name= 'col_1', target='y', ax=None, fig=None, clf=None,
                               plot_observations=True, plot_classifier_boundary=False,
                               plot_classifier_decisions=False, plot_classifier_prob_map = False,
                               threshold=0.5):
    """
    Plots the dependency of the two variables, with different colors based on the category they belong to.
    If plot_classifier_boundary is True, it will plot the decision boundary found by the classifier
    Note: plot_classifier_prob_map=True makes sense only if your classifier is trained on the two dimensional df,
    i.e. df[[x_name,y_name]].
    Args:
        df: pandas DataFrame containing the data
        x_name: (string) name of the column to be shown on the x-axis
        y_name: (string) name of the column to be shown on the y-axis
        target: (string or pandas Series) if the target variable is a column of the dataframe df, specify the name
            of the column (default = 'y'). Otherwise pass a pd.Series containing the classes.
        ax: matplotlib axes object None
        fig: matplotlib figure object, default None. Pass it to draw the cmap of the decision boundary
        clf: (object) trained machine learning classifier
        plot_observations: (boolean) flag to plot the observations
        plot_classifier_boundary: (boolean) flag to plot the classifier boundary
        plot_classifier_decisions: (boolean) flag to plot the different classifier decision zones
        plot_classifier_prob_map: (boolean) if True, it will score the df[[x_name,y_name]] and will use the predicted
            probabilities to create a map of probabilities
        threshold: (float) threshold of the decision boundary to be plotted, default = 0.5. Needs to be between 0 and 1


    Returns:
        ax: matplotlib axes object
    """
    if type(target) == str:
        df_0 = df[df[target] == 0]
        df_1 = df[df[target] == 1]
    elif type(target) == pd.core.series.Series:
        df_0 = df[target == 0]
        df_1 = df[target == 1]
    else:
        raise AttributeError('target is expected to be of type string or pd.Series')

    if plot_classifier_decisions and plot_classifier_prob_map:
        raise ValueError('plot_classifier_decisions and plot_classifier_prob_map cannot be true at the same time')

    x, y = df[x_name].values, df[y_name].values

    x_0, y_0 = df_0[x_name].values, df_0[y_name].values
    x_1, y_1 = df_1[x_name].values, df_1[y_name].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))

    if plot_observations:
        ax.scatter(x_0, y_0, s=20, color=color_markers_0, alpha=alpha_markers, marker='x')
        ax.scatter(x_1, y_1, color=color_markers_1, alpha=alpha_markers, marker='+')

    if plot_classifier_boundary or plot_classifier_prob_map or plot_classifier_prob_map:
        xx, yy = make_meshgrid(x, y, h=0.002)

        if plot_classifier_decisions:
            pcm = plot_contours(ax=ax, clf=clf, xx=xx, yy=yy, cmap=bayes_cmap, alpha=0.2)

        if plot_classifier_prob_map:
            preds_probs = clf.predict_proba(pd.DataFrame({x_name: xx.ravel(), y_name: yy.ravel()}))[:, 1]
            probs = preds_probs.reshape(xx.shape)

            pcm = ax.contourf(xx, yy, probs, cmap=bayes_cmap, alpha=0.3)
            if fig is not None: fig.colorbar(pcm, ax=ax, extend='both')

        if plot_classifier_boundary:
            pcm2 = plot_contours(ax=ax, clf=clf, xx=xx, yy=yy, use_contourf=False,
                            colors=color_levels, alpha=1,
                            linewidths=2,
							levels=np.array([threshold-0.01, threshold+0.01]))
   #     fig.colorbar(pcm, ax=ax, extend='both')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    return ax


def plot_contours(ax, clf, xx, yy,use_contourf=True, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if use_contourf:
        out = ax.contourf(xx, yy, Z, **params)
    else:
        out = ax.contour(xx, yy, Z, **params)
    return out


def plot_roc_curve(clf, X, y, ax=None,
                   title='Receiver operating characteristic', label='ROC curve ', color='darkorange',**kwargs):
    """
    Plot the roc curve
    Args:
        clf: pre-trained classifier
        X: (array or pd.Dataframe) the data to score
        y: (array or pd.Dataframe or pd.Series) the real targets
        ax: matplotlib axes object
        title: (string) Title of the grapsh
        label: (string) text to label the graph
        color: (string) specify the color of the curve

    Returns:
        ax: matplotlib axes object
    """
    predicted_proba = clf.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, predicted_proba)
    fpr, tpr, thr = roc_curve(y, predicted_proba)
    lw = 2

    label = label + str(' (area = %0.4f)' % roc_auc)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(fpr, tpr, color=color,
            lw=lw, label=label,**kwargs)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="best")
    return ax

def plot_precision_recall_curve(clf, X, y, ax=None,
                   title='Precision recall curve', label='Precision recall ', color='darkorange',**kwargs):
    """
    Plot the precision recall curve
    Args:
        clf: pre-trained classifier
        X: (array or pd.Dataframe) the data to score
        y: (array or pd.Dataframe or pd.Series) the real targets
        ax: matplotlib axes object
        title: (string) Title of the grapsh
        label: (string) text to label the graph
        color: (string) specify the color of the curve
        **kwargs: other kwargs for the plt.plot()

    Returns:
        ax: matplotlib axes object
    """
    predicted_proba = clf.predict_proba(X)[:, 1]
    precision, recall, thr = precision_recall_curve(y, predicted_proba)
    lw = 2

    #label = label + str(' (area = %0.4f)' % roc_auc)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(recall, precision, color=color,
            lw=lw, label=label,**kwargs)
    #ax.plot([1, 0], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def _plot_profit_func(profit, threshold, ax=None, x_lims=None, y_lims=None, **kwargs):
    """
    Plots profit (y-axis) versus thresholds =x-axis)

    Args:
        profit: calculated profit (y-axis)
        threshold: probability thresholds (x-axis)
        ax: matplotlib axes object
        x_lims: plotting limit for the x-axis
        y_lims: plotting limit for the y-axis
        **kwargs: kwargs for the ax.plot function

    Returns:

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))

    ax.plot(threshold, profit, **kwargs)

    ax.plot([0, 1], [0, 0], color='darkgreen', lw=2, linestyle='--')


    if x_lims == None:
        x_lims = (min(threshold), max(threshold))
    if y_lims == None:
        y_lims = min(profit), 1.20 * max(profit)

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_xlabel('Model probability threshold', size='x-large' )
    ax.set_ylabel('Profit (euro)', size='x-large')
    ax.legend(loc="best")


    return ax

def plot_classifier_profit(clf, X, y, ax=None, cost_of_bads=1, profit_of_goods=1,
                           n_points=100, x_lims=None, y_lims=None, **kwargs):
    """
    Plots the profit as function of the thresholds
    Args:
        clf: (object) traines classifier
        X: (np.array or pd.DataFrame) data set to predict
        y: (np.array or pd.Series) targets related to the dataset X
        ax: matplotlib axes object
        cost_of_bads: (float) false negative cost, cost of predicting a good outcome for a bad customer
        profit_of_goods: (float) true negative gain, gain of predicting a good outcome for a good customer
        n_points:  n_points: (int) number of points to estimate the thresholds
        x_lims: plotting limit for the x-axis
        y_lims: plotting limit for the y-axis
        **kwargs: kwargs for the ax.plot()

    Returns:

    """
    if type(y) == pd.core.series.Series:
        y = y.values

    profit, _, _, thresholds = estimate_profit(clf=clf, X=X, y=y,
                                               false_negative_cost=cost_of_bads, true_negative_gain=profit_of_goods)
    _plot_profit_func(profit, thresholds, ax=ax, x_lims=x_lims, y_lims=y_lims, **kwargs)

def plot_strategy_curve(clf, X, y, ax=None, n_points=100, **kwargs):
    """
    Plots the strategy curve. Acceptance rate vs the Bad Rate for a classifier clf
    Args:
        clf: (object) traines classifier
        X: (np.array or pd.DataFrame) data set to predict
        y: (np.array or pd.Series) targets related to the dataset X
        ax: matplotlib axes object
        n_points: number of points to estimate
        **kwargs: ax.plot related **kwargs arguments

    Returns:
        ax: matplotlib axes object
    """
    # by setting the costs to 1, it estimate profit returns the accepted goods and accepted bads, necessary
    # for the strategy curve
    _, n_acc_goods, n_acc_bads, _ = estimate_profit(clf, X, y,
                                                    false_negative_cost=1, true_negative_gain=1,
                                                    n_points=n_points)
    n_applications = y.shape[0]

    acc_rate = (n_acc_goods + n_acc_bads) / n_applications
    bad_rate = (n_acc_bads) /n_applications

    # Strategy curve
    ax.plot(acc_rate, bad_rate, **kwargs)
    ax.set_title('Strategy curve', size='xx-large')
    ax.set_xlabel('Acceptance Rate', size='x-large')
    ax.set_ylabel('Bad Rate', size='x-large')

    ax.legend(loc="best")
    return ax

def plot_dataframe_correlations(df,ax=None):
    """
    Plots the correlations of the dataframe in a heatmap matric
    Args:
        df: (pd.DataFrame) the data
        ax: matplotlib axes object
    """

    #calculate the correlation matrix
    corr = df.corr(method="spearman")

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    if ax==None:
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(15, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    correlation_heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                                      square=True, #xticklabels=2, yticklabels=2,
                                      linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


def plot_confusion_matrix(clf, X, y, classes=None, normalize=False, threshold=None,
                          title='Confusion matrix', ax=None, cmap=plt.cm.Blues):
    """
    Plot the confusion matrix for the classifier on the data X with targets y
    Args:
        clf: (object) traines classifier
        X: (np.array or pd.DataFrame) data set to predict
        y: (np.array or pd.Series) targets related to the dataset X
        classes: iterable that contains the labels of the classes. If none, it will be inferred from the y variable
        normalize: (boolean) flag for normalization of the confusion matrix
        threshold: (float), value between 0 and 1, the probability threshold for the classifier decision
        title: (string) title of the graph
        ax: matplotlib axis object
        cmap: maplotlib colormap object

    Returns:

    """

    if threshold == None:
        y_pred = clf.predict(X)
    else:
        y_pred = np.array(
            list(map(lambda x: 1 if x else 0,
                    clf.predict_proba(X)[:,1]>threshold)
                )
        )
        title = title + ' with threshold:{}'.format(threshold)

    if type(y) is pd.core.series.Series:
        y = y.values
    cm =confusion_matrix(y, y_pred)

    if classes==None:
        classes= np.unique(y)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", size='x-large',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',size='x-large')
    plt.xlabel('Predicted label',size='x-large')


def plot_score_vs_hyperparameter(clf, X, y, ax=None, scoring_metric='roc_auc', param_grid=None, cv=5):
    """
    Plots the dependence of the score as function of the hyperparameter
    Args:
        clf:
        X:
        y:
        ax:
        scoring_metric:
        param_grid:
        cv:

    Returns:

    """
    ### make sure that only one parameter is passed
    if len(param_grid) != 1:
        raise ValueError('Please pass only one parameter at the time')

    ## generate the grid search
    gs = GridSearchCV(clf, param_grid=param_grid, scoring=scoring_metric, cv=cv,return_train_score=True)
    gs.fit(X, y)
    results = gs.cv_results_

    # extract the parameter name
    param_name = [x for x in param_grid.keys()][0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 13))

    title = "Score vs " + param_name
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Score")

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results[('param_%s' % param_name)].data, dtype=float)

    # keep all the values to define the y_mimits
    y_values = []

    for sample, style, color in (('train', '--', 'green'), ('test', '-', 'darkorange')):
        sample_score_mean = results['mean_%s_score' % (sample)]
        y_values.append(sample_score_mean)
        sample_score_std = results['std_%s_score' % (sample)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scoring_metric, sample))

        best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
        best_score = results['mean_test_score'][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.4f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    ax.legend(loc="best")

    # define the y_limits of the figure

    max_y = max(y_values[0].max(), y_values[1].max())
    min_y = min(y_values[0].min(), y_values[1].min())

    ax.set_ylim(0.9 * min_y, 1.1 * max_y)
    return ax