import numpy as np
import pandas as pd

def logit(x):
    """
    Calculates the logistic function
    Args:
    	x:

    Returns:

    """
    return 1. / (1 + np.exp(-x))


def create_cont_variables(N_variables=3, size_population=10):
    '''create continuous variables'''
    dictio = {"cont_var_{}".format(i): np.random.rand(size_population) for i in range(N_variables)}
    return dictio


def create_exponential_variables(N_variables=3, size_population=10):
    '''create exponential variables'''
    dictio = {"exp_var_{}".format(i): np.random.exponential(size=size_population) for i in range(N_variables)}
    return dictio


def create_normal_variables(N_variables=3, size_population=10):
    '''create norma; variables'''
    dictio = {"norm_var_{}".format(i): np.random.normal(size=size_population) for i in range(N_variables)}
    return dictio


def convert_dictionaries_to_df(*dictionaries):
    '''
    convert dictionaries to
    '''
    out_dict = dict()
    for dictio in dictionaries:
        out_dict.update(dictio)
    out = pd.DataFrame(out_dict)
    out.columns = ["col_{}".format(i) for i in range(out.shape[1])]
    return out


def simulate_data(population_size, N_cont_variables=0, N_exponential_variables=0, N_normal_variables=0):
    '''
    General function to simulate all the data
    '''
    data1 = create_cont_variables(N_cont_variables, population_size)
    data2 = create_exponential_variables(N_exponential_variables, population_size)
    data3 = create_normal_variables(N_normal_variables, population_size)
    data = convert_dictionaries_to_df(data1, data2, data3)
    return data


def assign_class(probability, threshold=0.5, max_accuracy=1.):
    '''
    Assign a class to the observation based on the probability.
    Set the maximum accuracy to generate some swaps of targets. If 1, no swaps
    '''
    if probability < threshold:
        out = 0
    else:
        out = 1

    # change class given maximum accuracy
    if np.random.rand() > max_accuracy:
        if out == 0:
            out = 1
        else:
            out = 0
    return out


def _get_kwargs_for_function(func, **kwargs):
    """
    Extract the boundary functions
    """
    func_kwargs = {k: v for k, v in kwargs.items() if k in func.__code__.co_varnames}
    return func_kwargs


def bayes_boundary(boundary_func, print_median=False, **kwargs):
    """
    Generate the bayesian boundary.
    The boundary function in a user defined function to separate two classes, and it can depend on n-variables.
    Given the inputs, the function returns the generated probability of the datapoints to be in category 1,
    (for category 0 is obviously 1-prob), and the assigned targets.


    Args:
        boundary_func (func): Python function defining the boundary
        print_median (boolean):, set it to true to print the median of the boundary function calculated on the
            input data. This can be useful to balance the categories
        **threshold (float): threshold to define to which class does the observation belong
        **max_accuracy (float): percentage of points that are not randomly swapped between class 0 and 1. If
            1., then no swap is happening, if 0, all the classes will be swapped
        **kwargs: other kwargs are the inputs to the boundary function, which are in most of the cases coordinates


    Returns:
        tuple:
            probs: probability of the datapoints
            targets: binary category to which the observation belogs
            coordinates: dictionary containing the input coordinates to use in the bayes boundary computation
    """

    # Extract kwargs for the boundary function
    coordinates = _get_kwargs_for_function(boundary_func, **kwargs)
    rel = boundary_func(**coordinates)

    # Extract kwargs for the assign class function
    assign_class_args = _get_kwargs_for_function(assign_class, **kwargs)

    # Printing the median -
    if print_median:
        try:
            print(rel.median())
        except AttributeError:
            print(np.median(rel))

    if type(rel) == pd.core.series.Series:
        probs = rel.apply(lambda var: logit(var))
        targets = probs.apply(lambda var: assign_class(var, **assign_class_args))
    else:
        probs = logit(rel)
        vect_ = np.vectorize(assign_class)
        targets = vect_(probs, **assign_class_args)

    return probs, targets, coordinates


def boundary_1(x,y):
    return 10*np.sin(x) + 3*np.exp(x*y) - (x - y)/(x + y) + x*y - 10

def boundary_2(x,y,z):
    return x*y+z*y - z + x

def circle(x,y):
    return 5*(np.power((x-0.5),2) + np.power((y-0.5),2))-0.2

def parabola(x,y):
    return y + 5*np.power(x,8) - 0.5

