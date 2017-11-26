def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
"""Compute confusion matrix to evaluate the accuracy of a classification
    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.
    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
    Read more in the :ref:`User Guide <confusion_matrix>`.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    C : array, shape = [n_classes, n_classes]
        Confusion matrix
    References
    ----------
    .. [1] `Wikipedia entry for the Confusion matrix
           <https://en.wikipedia.org/wiki/Confusion_matrix>`_
    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    In the binary case, we can extract true positives, etc as follows:
    >>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    >>> (tn, fp, fn, tp)
    (0, 2, 1, 1)
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
if y_type not in ("binary", "multiclass"):
raise ValueError("%s is not supported" % y_type)
if labels is None:
        labels = unique_labels(y_true, y_pred)
else:
        labels = np.asarray(labels)
if np.all([l not in y_true for l in labels]):
raise ValueError("At least one label specified must be in y_true")
if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
else:
        sample_weight = np.asarray(sample_weight)
    check_consistent_length(sample_weight, y_true, y_pred)
    n_labels = labels.size
    label_to_ind = dict((y, x) for x, y in enumerate(labels))
# convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
# intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
# also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]
# Choose the accumulator dtype to always have high precision
if sample_weight.dtype.kind in {'i', 'u', 'b'}:
        dtype = np.int64
else:
        dtype = np.float64
CM = coo_matrix((sample_weight, (y_true, y_pred)),
shape=(n_labels, n_labels), dtype=dtype,
                    ).toarray()
return CM
