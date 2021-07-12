                                    #################################################
                                    ### We build our own utilize based on GPflow. ###
                                    #################################################

##### we import from gpflow and tensorflow #####
import tensorflow as tf
from gpflow import default_float
from gpflow.utilities import to_default_float
from collections.abc import Iterable
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def ndiag_mc_updated(funcs, S: int, Fmu, Fvar, logspace: bool = False, epsilon=None, **Ys):
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Monte Carlo samples. The Gaussians must be independent.

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise
    :param S: number of Monte Carlo sampling points
    :param Fmu: array/tensor
    :param Fvar: array/tensor
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param **Ys: arrays/tensors; deterministic arguments to be passed by name

    Fmu, Fvar, Ys should all have same shape, with overall size `N`
    :return: shape is the same as that of the first Fmu
    """

    ######################################################################################################################
    ### This funciton is similar to the funciton in GPflow. where I change the  N, D = Fmu.shape[0], Fvar.shape[1] to ####
    ###             N, D = tf.shape(Fmu)[0], tf.shape(Fvar)[1] compile to the tensorflow in static graph.             ####
    ######################################################################################################################

    N, D = tf.shape(Fmu)[0], tf.shape(Fvar)[1]

    if epsilon is None:
        epsilon = tf.random.normal((S, N, D), dtype=default_float())

    mc_x = Fmu[None, :, :] + tf.sqrt(Fvar[None, :, :]) * epsilon
    mc_Xr = tf.reshape(mc_x, (S * N, D))

    for name, Y in Ys.items():
        D_out = Y.shape[1]
        # we can't rely on broadcasting and need tiling
        mc_Yr = tf.tile(Y[None, ...], [S, 1, 1])  # [S, N, D]_out
        Ys[name] = tf.reshape(mc_Yr, (S * N, D_out))  # S * [N, _]out

    def eval_func(func):
        feval = func(mc_Xr, **Ys)
        feval = tf.reshape(feval, (S, N, -1))
        if logspace:
            log_S = tf.math.log(to_default_float(S))
            return tf.reduce_logsumexp(feval, axis=0) - log_S  # [N, D]
        else:
            return tf.reduce_mean(feval, axis=0)

    if isinstance(funcs, Iterable):
        return [eval_func(f) for f in funcs]
    else:
        return eval_func(funcs)


def run_adam(model, iterations,data,N,minibatch_size, Moreoutput = False):
    """
    Utility function running the Adam optimizer
    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    if Moreoutput == True:
        train_iter = iter(data.batch(minibatch_size))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(N)
        train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(0.01)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 100 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            print(elbo)
    return logf


def Combining_X_with_Index(X,Num_output):
    """
    We combine the X with index
    Retrun a list that include all the data with index
    """
    ### combine all x and Y with the index
    X_combine_data = []
    for i in range(Num_output):
        ## Combine the index for X and Y separately
        X_combine = np.hstack((X[i],i * np.ones_like(X[i][:, 0][:, None])))
        X_combine_data.append(X_combine)
    # X_data = np.vstack(X_combine_data)
    return X_combine_data


def Transform_data_for_training(X,Y,Num_class,Num_output):
    """
    To do set up different classes. It is use for training dataset for all output together
    Here we set up a function for training
    """
    ### We set up all the class
    Class_cumsum = np.cumsum(Num_class)
    ### We change corresponding labels
    index_all = [Y[0]]
    for i in range(Num_output - 1):
        index_all.append(Y[i + 1] + Class_cumsum[i])

    ### combine all x and Y with the index
    ### Transfer them into the format for optimzating
    Data_opt = []
    for i in range(Num_output):
        ## Combine the index for X and Y separately
        X_combine = np.hstack((X[i], index_all[i]))
        Y_combine = np.hstack((Y[i], i * np.ones_like(Y[i])))

        ## combine two dataset together
        Dataset = (X_combine, Y_combine)

        ## Changing the data in tensorflow format
        Dataset_tf = tf.data.Dataset.from_tensor_slices(Dataset).repeat().shuffle(Y_combine.shape[0])

        ## add the data into a list
        Data_opt.append(Dataset_tf)

    ## Combining all the dataset together
    Data_toget = tf.data.Dataset.zip(tuple(Data_opt))

    return Data_toget


def Performance_measure(mu, Yest):
    """
    We calculate each performance measure by using predictive values and existed values
    """
    ### predictive mean
    mu_y = mu
    ### existed values
    Ytest = Yest

    ### Prediciton error
    Test_error = np.mean(mu_y == Ytest.squeeze())
    ### Precision
    P_w = precision_score(Ytest, mu_y[:, None], average='weighted')
    ### Recall
    R_w = recall_score(Ytest, mu_y[:, None], average='weighted')
    ### F1
    F1_w = f1_score(Ytest, mu_y[:, None], average='weighted')

    return Test_error, P_w, R_w, F1_w


def Performance_list():
    """
    This function helps up to build a performance list for each performance metrics
    Building this function is only for cleaning the code.
    """
    ### Time training and elbo_minibatch
    time_training = []
    ### Prediction error
    Prediction_error = []
    ### Precision
    Precision_weighted = []
    ### Recall
    Recall_weighted = []
    ### F1
    F1_weighted = []
    return time_training, Prediction_error, Precision_weighted, Recall_weighted, F1_weighted