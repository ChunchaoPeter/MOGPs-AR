                                    ###################################################
                                    ### We build our convariances based on GPflow.#####
                                    ###################################################

##### we import from gpflow and tensorflow #####
import tensorflow as tf
from gpflow.base import TensorLike
from gpflow.kernels import Kernel, Convolutional
from MOGP_convolutionalKernel.kernels import lmc_kernel, Convolutional_SE
from gpflow.inducing_variables import (
    InducingPoints,
    InducingPatches,
    SharedIndependentInducingVariables,
    SeparateIndependentInducingVariables,
)
from gpflow.covariances.dispatch import Kuu, Kuf
from gpflow.config import default_float

                            ###########################################################################
                            ################## Building new Kuu and Kuf ###############################
                            ###########################################################################

####################################################################################
###########  Building Single-Output Gaussian Processes Kuu  ########################
####################################################################################

##################
### lmc_kernel ###
##################
# This is ususally for single output Gaussian processes. However, we are interested in the multiple-output Gaussian processes
# so that we use the LMC kernel. Thus, the Kuu_lmc_test is calculated for the kuu when we use the InducingPoints and lmc_kenel.
@Kuu.register(InducingPoints, lmc_kernel)
def Kuu_lmc_test(inducing_variable: InducingPoints, kernel: lmc_kernel, *, jitter=0.0):
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz

################################
### Convolutional_SE kernel ####
################################
# The Kuu_conv_SEpatch is calculated for the kuu when we use the InducingPatches and Convolution_SE.
@Kuu.register(InducingPatches, Convolutional_SE)
def Kuu_conv_SEpatch(feat, kern, jitter=0.0):
    return kern.K_SE(feat.Z) + jitter * tf.eye(feat.num_inducing, dtype=default_float())



####################################################################################
###########  Building Single-Output Gaussian Processes Kuf  ########################
####################################################################################

##################
### lmc_kernel ###
##################
# This is ususally for single output Gaussian processes. However, we are interested in the multiple-output Gaussian processes
# so that we use the LMC kernel. Thus, the Kuf_kernel_test is calculated for the kuf when we use the InducingPoints and lmc_kenel.
@Kuf.register(InducingPoints, lmc_kernel, TensorLike)
def Kuf_kernel_test(inducing_variable: InducingPoints, kernel: lmc_kernel, Xnew):
    return kernel(inducing_variable.Z, Xnew)

################################
### Convolutional_SE kernel ####
################################
# The Kuf_conv_SEpatch is calculated for the kuu when we use the InducingPatches and Convolution_SE.
@Kuf.register(InducingPatches, Convolutional_SE, object)
def Kuf_conv_SEpatch(feat, kern, Xnew):
    Xp = kern.get_patches(Xnew)  # N x num_patches x patch_len
    bigKzx = kern.K_SE(feat.Z, Xp)  # [M, N, P] -- thanks to broadcasting of kernels
    Kzx = tf.reduce_sum(bigKzx * kern.weights if hasattr(kern, 'weights') else bigKzx, [2])
    return Kzx / kern.num_patches


######################################################################################
###########  Building Multiple-Output Gaussian Processes Kuu  ########################
######################################################################################

######################
### lmc kernel  ######
######################
# In lmc kernel, we have more than one kernel, is called base kernel.
# If all the base kernels use the same inducing variable, the we use Kuu_lmc_shardiv to calculate the Kuu
# We calcualte the Kuu when we use SharedIndependentInducingVariables and lmc_kernel.
@Kuu.register(SharedIndependentInducingVariables, lmc_kernel)
def Kuu_lmc_sharediv(inducing_variable: SharedIndependentInducingVariables,
                     kernel: lmc_kernel, *, jitter=0.0):
    Kmm = tf.stack([Kuu(inducing_variable.inducing_variable_shared, k)
                    for k in kernel.kernels], axis=0)  # [L, M, M]
    # jittermat = tf.eye(len(inducing_variable),
    #                    dtype=Kmm.dtype)[None, :, :] * jitter
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat

# In lmc kernel, we have more than one kernel, is called base kernel.
# If all the base kernels use the different inducing variable, the we use Kuu_lmc_shardiv to calculate the Kuu.
# We calcualte the Kuu when we use SeparateIndependentInducingVariables and lmc_kernel.
@Kuu.register(SeparateIndependentInducingVariables, lmc_kernel)
def Kuu_lmc_separateiv(inducing_variable: SeparateIndependentInducingVariables,
                       kernel: lmc_kernel, *, jitter=0.0):
    Kmm = tf.stack([Kuu(f, k) for f, k in
                    zip(inducing_variable.inducing_variable_list,
                        kernel.kernels)], axis=0)  # [L, M, M]
    jittermat = tf.eye(inducing_variable.num_inducing,
                       dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


######################################################################################
###########  Building Multiple-Output Gaussian Processes Kuf  ########################
######################################################################################

######################
### lmc kernel  ######
######################
# In lmc kernel, we have more than one kernel, is called base kernel.
# If all the base kernels use the same inducing variable, the we use Kuf_lmc_shardiv to calculate the Kuf
# We calcualte the Kuf when we use SharedIndependentInducingVariables and lmc_kernel.
@Kuf.register(SharedIndependentInducingVariables, lmc_kernel, TensorLike)
def Kuf_lmc_sharediv(inducing_variable: SharedIndependentInducingVariables,
                     kernel: lmc_kernel, Xnew: tf.Tensor):
    Kufs = [Kuf(inducing_variable.inducing_variable_shared, k, Xnew[:, :-1])
            for k in kernel.kernels]
    Ws = [tf.reduce_sum(coreg.W, -1) for coreg in kernel.coregs]
    ind = tf.cast(Xnew[:, -1], tf.int32)
    return tf.stack([tf.gather(W, ind) * Kuf for W, Kuf in zip(Ws, Kufs)])


# In lmc kernel, we have more than one kernel, is called base kernel.
# If all the base kernels use the different inducing variable, the we use Kuf_lmc_shardiv to calculate the Kuf
# We calcualte the Kuf when we use SeparateIndependentInducingVariables and lmc_kernel.
@Kuf.register(SeparateIndependentInducingVariables, lmc_kernel, TensorLike)
def Kuf_lmc_separateiv(inducing_variable: SharedIndependentInducingVariables,
                       kernel: lmc_kernel, Xnew: tf.Tensor):
    Kufs = [Kuf(f, k, Xnew[:, :-1]) for f, k in
            zip(inducing_variable.inducing_variable_list, kernel.kernels)]
    Ws = [tf.reduce_sum(coreg.W, -1) for coreg in kernel.coregs]
    ind = tf.cast(Xnew[:, -1], tf.int32)
    return tf.stack([tf.gather(W, ind) * Kuf for W, Kuf in zip(Ws, Kufs)])


###########################################################################
################## This is only for testing ###############################
###########################################################################
### Convolutional kernel
@Kuu.register(InducingPatches, Convolutional)
def Kuu_conv_patch_test(feat, kern, jitter=0.0):
    return kern.base_kernel.K(feat.Z) + jitter * tf.eye(len(feat), dtype=default_float())
####################################
###########  Kuf  ##################
####################################
### General Kernel
@Kuf.register(InducingPoints, Kernel, TensorLike)
def Kuf_kernel_test(inducing_variable: InducingPoints, kernel: Kernel, Xnew):
    return kernel(inducing_variable.Z, Xnew)
### Convolutional kernel
@Kuf.register(InducingPatches, Convolutional, object)
def Kuf_conv_patch_test(feat, kern, Xnew):
    Xp = kern.get_patches(Xnew)  # [N, num_patches, patch_len]
    bigKzx = kern.base_kernel.K(feat.Z, Xp)  # [M, N, P] -- thanks to broadcasting of kernels
    Kzx = tf.reduce_sum(bigKzx * kern.weights if hasattr(kern, "weights") else bigKzx, [2])
    return Kzx / kern.num_patches


