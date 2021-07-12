                                    ########################################################
                                    ### We build our own conditionals based on GPflow. #####
                                    ########################################################

##### we import from gpflow and tensorflow #####
import tensorflow as tf
### import from our building model
from MOGP_convolutionalKernel import covariances
from MOGP_convolutionalKernel.kernels import lmc_kernel
from MOGP_convolutionalKernel.kernels import Convolutional_SE
### import from gpflow
from gpflow.kernels import Kernel
from gpflow.config import default_float, default_jitter
from gpflow.conditionals.dispatch import conditional
from gpflow.inducing_variables import InducingVariables, InducingPatches, SharedIndependentInducingVariables, SeparateIndependentInducingVariables
from gpflow.conditionals.util import base_conditional, \
    expand_independent_outputs, rollaxis_left


###################################################################
############### Multiple Output Gaussian Processes ################
###################################################################

@conditional.register(object, SharedIndependentInducingVariables, lmc_kernel, object)
@conditional.register(object, SeparateIndependentInducingVariables, lmc_kernel,
                      object)
def lmc_conditional_mogp(Xnew: tf.Tensor,
                    inducing_variable: InducingVariables,
                    kernel: Kernel,
                    f: tf.Tensor,
                    *,
                    full_cov=False,
                    full_output_cov=False,
                    q_sqrt=None,
                    white=False):
    """Multi-output GP with independent GP priors.
    The covariance matrices used to calculate the conditional have the
    following shape:
    - Kuu: [L, M, M]
    - Kuf: [L, M, N]
    - Kff: [N] or [N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput
      framework.
    - See above for the parameters and the return value.
    """
    # Following are: [L, M, M]  -  [L, M, N]  -  [L, N](x N)
    Kmms = covariances.Kuu(inducing_variable, kernel,
                           jitter=default_jitter())  # [L, M, M]
    Kmns = covariances.Kuf(inducing_variable, kernel, Xnew)  # [L, M, N]
    Knn = kernel(Xnew, full_cov=full_cov)  # [N, N] or [N]
    fs = tf.transpose(f)[:, :, None]  # [L, M, 1]
    # [L, 1, M, M]  or  [L, M, 1]
    q_sqrts = tf.transpose(q_sqrt)[:, :, None] if q_sqrt.shape.ndims == 2 \
        else q_sqrt[:, None, :, :]

    def single_gp_conditional(t):
        Kmm, Kmn, f, q_sqrt = t
        return base_conditional(
            Kmn,
            Kmm,
            tf.zeros_like(Knn, dtype=default_float()),  # dummy Knn  # noqa
            f,
            full_cov=full_cov,
            q_sqrt=q_sqrt,
            white=white,
        )

    rmu, rvar = tf.map_fn(
        single_gp_conditional, (Kmms, Kmns, fs, q_sqrts), (default_float(), default_float())
    )  # [L, N, 1], [L, 1, N, N] or [L, N, 1]

    fmu = rollaxis_left(rmu[..., 0], 1)  # [N, L]
    fmu = tf.reduce_sum(fmu, axis=-1, keepdims=True)  # [N, 1]

    if full_cov:
        fvar = rvar[:, 0]  # [L, N, N]
        fvar = tf.reduce_sum(fvar, axis=0)[0]  # [N, N]
        fvar = Knn + fvar
    else:
        fvar = rollaxis_left(rvar[..., 0], 1)  # [N, L]
        fvar = tf.reduce_sum(fvar, axis=-1, keepdims=True)  # [N, 1]
        fvar = Knn[:, None] + fvar

    return fmu, expand_independent_outputs(fvar, full_cov, full_output_cov)



#################################################################
############### Single Output Gaussian Processes ################
#################################################################

@conditional.register(object, InducingVariables, lmc_kernel, object)
@conditional.register(object, InducingPatches, Convolutional_SE, object)
def lmc_conditional_singleogp(
        Xnew: tf.Tensor,
        inducing_variable: InducingVariables,
        kernel: Kernel,
        f: tf.Tensor,
        *,
        full_cov=False,
        full_output_cov=False,
        q_sqrt=None,
        white=False,
):
    """
    Single-output GP conditional.
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: [N, N]
    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` (below) for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.
    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, R]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
           NOTE: as we are using a single-output kernel with repetitions
                 these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, R]
        - variance: [N, R], [R, N, N], [N, R, R] or [N, R, N, R]
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    Kmm = covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())  # [M, M]
    Kmn = covariances.Kuf(inducing_variable, kernel, Xnew)  # [M, N]
    Knn = kernel(Xnew, full_cov=full_cov)
    fmean, fvar = base_conditional(
        Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )  # [N, R],  [R, N, N] or [N, R]
    return fmean, expand_independent_outputs(fvar, full_cov, full_output_cov)
