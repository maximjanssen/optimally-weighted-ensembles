'''
Use different solvers to calculate optimally weighted convex combinations of machine learning systems
Currently included:
    - CVXPY (using convex_combinations.py)
    - any solver from qpsolvers (using matrix calculations performed in this module)
'''

# For calculations:
import numpy as np

# For getWeights_cvxpy:
from convex_combinations import lossFunction_ols
import cvxpy as cp  # used for quadratic & linear programming solving

# For getWeights_solveqp
import qpsolvers

# For reporting:
from sklearn.metrics import mean_squared_error

# For conversion to csc_matrix:
from scipy import sparse

import time


def getPrediction(model, X, required_shape):
    '''
    Quick function to get the prediction of a model and check if it conforms
    to a required shape. If it doesn't, the prediction is reshaped to conform
    input variables:
        model - fitted model get a prediction from
        X - attributes to be used to get a prediction
        required shape - object with the shape a prediction needs to be in
    output:
        A prediction with the requested shape
    '''
    prediction = model.predict(X)
    if prediction.shape != required_shape:
        prediction = np.reshape(prediction, required_shape)
    return prediction


def removeLowWeigths(weights: np.ndarray, cut_off: float) -> np.ndarray:
    '''
    Removes all weights lower than cut_off
    input variables:
        weights - numpy array of weights, with np.sum == 1. If weights is a list, function converts it to an numpy array
        cut_off - numeric value, any weights lower than cut_off get set to 0
    output:
        numpy array of weights with np.sum == 1. All values are >= cut_off
    '''

    # Set all weights below cut_off to 0
    try:
        remove_weights = weights < cut_off
    except TypeError:
        # weights is probably a list, convert to numpy.array
        weights = np.array(weights)
        remove_weights = weights < cut_off

    weights[remove_weights] = 0

    # Correct the rest of the weigths so the sum of weights is still 1
    total_sum = np.sum(weights)

    if total_sum == 0:
        raise ValueError(
            f'cut_off ({cut_off}) is higher than highest weight')

    weights = weights / total_sum

    return weights


def getWeights_cvxpy(models: list, features: list, target: list, lambda_reg: float = 0, cut_off: float = 0) -> list[float]:
    '''
    Calculates the optimal combination of weights to be assigned to a list of models
    uses cvxpy for quadratic/linear programming solving
    input variables:
        models      - list of models fitted to a training set
        features    - list of features used to be able to predict target
        target      - list of output values to be predicted using features
        lambda_reg  - regularization parameter >= 0 to decrease the weight given to underperforming models
                    the higher lambda_reg, the lower the weight assigned to models with a high
                    individual error
    output:
        list of optimal weights to be assigned to predictions of models

    corresponds to (1) on page 4 of paper

    This function is mainly for comparing the performance of other getWeights programs
    '''

    variables = list()  # list of variables to be used for solving
    constr = list()     # list of contraints to be used for solving
    weights = list()  # list of weights to be assigned to model predictions

    if lambda_reg < 0:
        raise ValueError(
            f'lambda reg has to be >= 0, current value is {lambda_reg}')

    if len(models) == 0:
        raise ValueError('list of models is empty')

    if len(features) != len(target):
        raise ValueError('target and features lists should be the same length')

    if cut_off < 0:
        raise ValueError(f'cutoff has to be >=0, current value is {cut_off}')

    if cut_off >= 1:
        raise ValueError(f'cut_off has to be < 1, current value is {cut_off}')

    # contruction of list of weights using cp.Variables
    for i in range(len(models)):
        variable = cp.Variable()
        variables += [variable]
        constr += [variable >= 0]

    # cost function for solving
    cost = lossFunction_ols(
        models, variables, features, target, lambda_reg)
    objective = cp.Minimize(cost)   # objective for solving
    constr += [np.sum(variables) == 1]  # sum of weights has to be 1

    prob = cp.Problem(objective, constr)  # defining optimization problem

    # solver was chosen at random.
    opt_val = prob.solve(solver='OSQP')
    # TODO: experiment with different solvers
    # TODO: let solver be a variable that user can enter

    for var in variables:
        weights += [var.value]

    if cut_off > 0:
        weights = removeLowWeigths(np.array(weights), cut_off)

    return weights


def getMatrices(models: list, X: list, target: list, lambda_reg: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Uses the predictions by models to return the Q and c matrices that are used for the equation of quadratic programming
    input variables:
        models - list of models used to make predictions
        X      - list of X values used to predict target
        target - list of targets to be predicted
    output:
        two variables representing the Q and c matrices
    '''
    predictions = list()
    for features in X:
        model_predictions = list()
        for name, model in models:
            prediction = getPrediction(model, [features], (1,))
            model_predictions.append(prediction[0])
        predictions += [model_predictions]

    Q = np.matmul(np.transpose(predictions), predictions)
    c = np.matmul(np.transpose(predictions), -target)
    # Negative target gives comparable performance as CVXPY on full algorithm from paper (getWeights_cvxpy)
    # Should be -2 * target, but this gives a worse performance
    # TODO: find out why

    if lambda_reg == 0:  # prevent unnecessary calculations
        return Q, c

    errors = list()
    for name, model in models:
        prediction = model.predict(X)
        errors += [mean_squared_error(target, prediction)]

    errors = lambda_reg * np.array(errors)
    c = np.add(c, errors)

    return Q, c


def getSolvers() -> list:
    '''
    returns a list of all available solvers that can be used by qpsolvers
    '''
    return(qpsolvers.available_solvers)


def getWeights_solveqp(models: list, features: list, target: list, solver: str = 'osqp', lambda_reg: int = 0, cut_off: float = 0) -> dict:
    '''
    Uses the qpsolvers package to get the optimal combination of weights
    input variables:
        models      - list of models used for the ensemble
        features    - list of features used to validate/get weights. these variables should
                        not have been used in fitting the models
        target      - list of values that are predicted using features
        solver      - the solver to be used by the solvegp package, defaults to quadprog
        lambda_reg  - regularization parameter >= 0 to decrease the weight given to underperforming models
                    the higher lambda_reg, the lower the weight assigned to models with a high
                    individual error
    output:
        list of weights to create a weighted average
    '''

    if lambda_reg < 0:
        raise ValueError(
            f'lambda reg has to be >= 0, current value is {lambda_reg}')

    if len(models) == 0:
        raise ValueError('list of models is empty')

    if len(features) != len(target):
        raise ValueError('target and features lists should be the same length')

    if cut_off < 0:
        raise ValueError(f'cutoff has to be >=0, current value is {cut_off}')

    if cut_off >= 1:
        raise ValueError(f'cut_off has to be < 1, current value is {cut_off}')

    noOfModels = len(models)
    P, q = getMatrices(models, features, target, lambda_reg)

    P = sparse.csr_matrix(P)

    A = np.ones(noOfModels)
    A = sparse.csr_matrix(A)
    b = [1]

    lower_bound = np.zeros(noOfModels)
    upper_bound = np.ones(noOfModels)

    weights = qpsolvers.solve_qp(P=P, q=q, A=A, b=b, lb=lower_bound,
                                 ub=upper_bound, solver=solver)

    weights = removeLowWeigths(np.array(weights), cut_off)

    weights_dict = {}

    for (name, model), weight in zip(models, weights):
        weights_dict[name] = weight

    return weights_dict


def getWeightedPrediction(predictions: dict, weights: dict, prediction_size: int) -> list[float]:
    '''
    returns a weighted prediction
    input variables:
        predictions - dict of predictions made by different models with model names as keys
        weights     - dict of weights to be assigned with model names as keys
    output:
        a list of weighted predictions
    '''
    weighted_prediction = [0] * prediction_size
    for model_name in weights.keys():
        weighted_prediction += predictions[model_name] * weights[model_name]

    return weighted_prediction
