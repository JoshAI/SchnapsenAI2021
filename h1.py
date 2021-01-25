def optimizing(theta, data = data, target = target):
    """ Simple two-dimensional function containing several local maxima.
    From: The Merits of a Parallel Genetic Algorithm in Solving Hard 
    Optimization Problems, A. J. Knoek van Soest and L. J. R. Richard 
    Casius, J. Biomech. Eng. 125, 141 (2003)
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1
       * - Type
         - maximization
       * - Range
         - :math:`x_i \in [-100, 100]`
       * - Global optima
         - :math:`\mathbf{x} = (8.6998, 6.7665)`, :math:`f(\mathbf{x}) = 2`\n
       * - Function
         - :math:`f(\mathbf{x}) = \\frac{\sin(x_1 - \\frac{x_2}{8})^2 + \
            \\sin(x_2 + \\frac{x_1}{8})^2}{\\sqrt{(x_1 - 8.6998)^2 + \
            (x_2 - 6.7665)^2} + 1}`
    .. plot:: code/benchmarks/h1.py
       :width: 67 %
    """
    # num = (sin(individual[0] - individual[1] / 8))**2 + (sin(individual[1] + individual[0] / 8))**2
    # denum = ((individual[0] - 8.6998)**2 + (individual[1] - 6.7665)**2)**0.5 + 1
    learner = MLPClassifier(hidden_layer_sizes=theta[0], learning_rate_init=theta[1], alpha=theta[2], verbose=True, early_stopping=True, n_iter_no_change=6)

    model = learner.fit(data, target)
    return model.best_loss_ # return the best error/residual (best error is cloes to zero) The error of a model is the difference between your predicted outcome and the real observed outcome and therefore 0 is desired

    # individual as a factor containing x, y = to tetha which 