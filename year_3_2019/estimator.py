import numpy as np

def compute_gain_loss(y_true, y_pred, fnc, tng):
    """
    Computes the gain and loss of the portfolio.
    If fnc (false negative cost) is 1 and tng (true negative gain) is 1, it will return the number of accepted
    goods (gain) and bads (loss).

    Args:
        y_true: (np.array) array containing the real observations
        y_pred: (np.array) array containing the predicted observations
        fnc: (float) false negative cost
        tng: (float) true negative cost. The gain of predicting well the negatives

    Returns:
        gain, loss: (tuple), contains the total gain and loss of the portfolio
    """
    ## choose only those that matter- predicted as non defaults
    outc = y_true[y_pred == 0]
    # outcomes contains the status of the real portfolio
    bads = outc.sum()
    goods = outc.shape[0] - outc.sum()

    loss = bads * fnc
    gain = goods * tng

    return gain, loss


def estimate_profit(clf, X, y, false_negative_cost=1, true_negative_gain=1, n_points=100):
    """
    Estimates the total profit of the portfolio with a given classifier predicting the outcomes.
    Returns `n_points` estimates of profits at different predicted probabilities
    Args:
        clf: (object) trained classifier
        X: (np.array or pd.DataFrame) the dataset of features
        y: (np.array or pd.DataFrame or pd.Series) the targets
        false_negative_cost: (float) false negative cost, cost of predicting a good outcome for a bad customer
        true_negative_gain: (float) true negative gain, gain of predicting a good outcome for a good customer
        n_points: (int) number of points to estimate the thresholds

    Returns:
        profit, gain, loss, thresholds: (tuple of np.arrays)
    """
    # compute probability for each observation
    probs = clf.predict_proba(X)[:, 1]

    # calculate predcitions for all different tresholds
    thresholds = np.linspace(0, 1, n_points)

    # generator of all predictions. array or list with outcomes of different thresholds
    all_predictions = (np.array(list(map(lambda x: 1 if x > thresh else 0, probs)))
                       for thresh in thresholds)

    gain_loss = np.array([compute_gain_loss(y_true=y, y_pred=predictions,
                                            fnc=false_negative_cost, tng=true_negative_gain)
                          for predictions in all_predictions])
    gain = gain_loss[:, 0]
    loss = gain_loss[:, 1]
    profit = gain - loss

    return profit, gain, loss, thresholds

#
#from sklearn.calibration import calibration_curve
# plt.figure(figsize=(10, 10))
# ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
# ax2 = plt.subplot2grid((3, 1), (2, 0))

# ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
# clf = rf
# clf.fit(X_train, y_train)
# if hasattr(clf, "predict_proba"):
#     prob_pos = clf.predict_proba(X_test)[:, 1]
# else:  # use decision function
#     prob_pos = clf.decision_function(X_test)
#     prob_pos = \
#         (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
# fraction_of_positives, mean_predicted_value = \
#     calibration_curve(y_test, prob_pos, n_bins=10)

# ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
#          label="%s" % ('RF', ))

# ax2.hist(prob_pos, range=(0, 1), bins=10, label='RF',
#          histtype="step", lw=2)

# ax1.set_ylabel("Fraction of positives")
# ax1.set_ylim([-0.05, 1.05])
# ax1.legend(loc="lower right")
# ax1.set_title('Calibration plots  (reliability curve)')

# ax2.set_xlabel("Mean predicted value")
# ax2.set_ylabel("Count")
# ax2.legend(loc="upper center", ncol=2)