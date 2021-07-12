                                    #################################################
                                    ### We build our own kernel based on GPflow.#####
                                    #################################################

##### we import from gpflow and tensorflow #####
import tensorflow as tf
import numpy as np
from gpflow.kernels import SquaredExponential, Combination, Coregion
from gpflow.config import default_float
from gpflow.base import Parameter
from gpflow.utilities import to_default_float
                                                #################################
                                                #### Building new kernel ########
                                                #################################

######################################
########## lmc kernel   ##############
######################################
class lmc_kernel(Combination):
    '''
    LMC kernel. See the Kernels for Vector-Valued Functions: a Review by Mauricio etc.
    '''
    def __init__(self, output_dim, kernels, ranks=None, name=None):
        """
        A Kernel for Linear Model of Coregionalization
        """
        self.output_dim = output_dim
        if ranks is None:
            ranks = np.ones_like(kernels)

        ks = []
        self.coregs = []
        for k, r in zip(kernels, ranks):
            coreg = Coregion(output_dim, r, active_dims=slice(-1, None))
            # coreg.kappa = default_jitter() * tf.constant(np.zeros(output_dim))  # noqa
            coreg.kappa = tf.constant(np.zeros(output_dim))
            coreg.W.assign(np.random.rand(output_dim, r))
            self.coregs.append(coreg)
            ks.append(k)

        Combination.__init__(self, ks, name)

    def Kgg(self, X, X2=None, full_cov=True): # [L, N, N2]
        if full_cov:
            if X2 is None:
                return tf.stack([coreg(X) * k(X[:, :-1]) for coreg, k in
                                 zip(self.coregs, self.kernels)], axis=0)
            return tf.stack([coreg(X, X2) * k(X[:, :-1], X2[:, :-1])
                             for coreg, k in zip(self.coregs, self.kernels)],
                            axis=0)
        return tf.stack([coreg(X, full_cov=False) * k(X[:, :-1], full_cov=False)
                         for coreg, k in zip(self.coregs, self.kernels)],
                        axis=0)

    def K(self, X, X2=None):  # [N, N2]
        return tf.reduce_sum(self.Kgg(X, X2), axis=0)

    def K_diag(self, X):  # [N]
        return tf.reduce_sum(self.Kgg(X, full_cov=False), axis=0)



######################################
####### convolutional kernel #########
######################################
class Convolutional_SE(SquaredExponential):
    '''
    This is a convolution kernel based on SE or RBF kernel based. See Convolutional Gaussian Processes by Mark
    '''
    def __init__(self, image_shape, patch_shape, weights=None, colour_channels=1, **kwargs):
        super().__init__(**kwargs)
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.colour_channels = colour_channels
        self.weights = Parameter(np.ones(self.num_patches, dtype=default_float()) if weights is None
                                 else weights)

    def get_patches(self, X):
        """
        Extracts patches from the images X. Patches are extracted separately for each of the colour channels.
        :param X: (N x input_dim)
        :return: Patches (N, num_patches, patch_shape)
        """
        # Roll the colour channel to the front, so it appears to
        # `tf.extract_image_patches()` as separate images. Then extract patches
        # and reshape to have the first axis the same as the number of images.
        # The separate patches will then be in the second axis.
        num_data = tf.shape(X)[0]
        castX = tf.transpose(tf.reshape(X, [num_data, -1, self.colour_channels]), [0, 2, 1])
        patches = tf.image.extract_patches(
            tf.reshape(castX, [-1, self.image_shape[0], self.image_shape[1], 1], name="rX"),
            [1, self.patch_shape[0], self.patch_shape[1], 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            "VALID",
        )
        shp = tf.shape(patches)  # img x out_rows x out_cols
        reshaped_patches = tf.reshape(
            patches, [num_data, self.colour_channels * shp[1] * shp[2], shp[3]]
        )
        return to_default_float(reshaped_patches)

    def K_SE(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r2 = self.scaled_squared_euclid_dist(X, X2)
        return self.K_r2(r2)

    def K(self, X, X2=None):
        Xp = self.get_patches(X)  # [N, P, patch_len] N is the number of image, P is the number of pathes in each image, patch_le is the length of the each patch, it is a vector.
        Xp2 = Xp if X2 is None else self.get_patches(X2)

        bigK = self.K_SE(Xp, Xp2)  # [N, num_patches, N, num_patches]

        W2 = self.weights[:, None] * self.weights[None, :]  # [P, P]
        W2bigK = bigK * W2[None, :, None, :]
        return tf.reduce_sum(W2bigK, [1, 3]) / self.num_patches ** 2.0
        ### The dimensions to reduce [1, 3]

    def K_diag(self, X):
        Xp = self.get_patches(X)  # N x num_patches x patch_dim
        W2 = self.weights[:, None] * self.weights[None, :]  # [P, P]
        bigK = self.K_SE(Xp)  # [N, P, P]
        return tf.reduce_sum(bigK * W2[None, :, :], [1, 2]) / self.num_patches ** 2.0

    @property
    def patch_len(self):
        return np.prod(self.patch_size)

    @property
    def num_patches(self):
        return (
            (self.image_shape[0] - self.patch_shape[0] + 1)
            * (self.image_shape[1] - self.patch_shape[1] + 1)
            * self.colour_channels
        )
