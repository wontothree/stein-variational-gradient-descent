import numpy as np
from scipy.spatial.distance import pdist, squareform

class SVGD():
    def __init__(self):
        pass

    def svgd(self, theta, h = -1):
        """
        Computes the Stein Variational Gradient Descent (SVGD) kernel and its gradient.

        This function calculates the pairwise squared distances between samples in 
        the input array `theta`, applies the RBF kernel, and computes the gradient of 
        the kernel with respect to the samples.

        Parameters:
        ----------
        theta : ndarray
            An array of shape (n_samples, n_dimensions) containing the sample points 
            for which the kernel and its gradient are to be computed.

        h : float, optional
            Bandwidth parameter for the RBF kernel. If h < 0, the bandwidth is 
            computed using the median trick. The default is -1.

        Returns:
        -------
        Kxy : ndarray
            The kernel matrix of shape (n_samples, n_samples) representing the pairwise 
            RBF kernel values between the samples.

        dxkxy : ndarray
            The gradient of the kernel with respect to the samples, of shape 
            (n_samples, n_dimensions). This represents how the kernel values change 
            with respect to the input samples.
        """

        # squared euclidean distance
        pairwise_distance = squareform(pdist(theta))**2

        # if h < 0, using median trick
        if h < 0:
            h = np.median(pairwise_distance)  
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        # RBF kernel (Radial Basis Function kernel)
        rbf_kernel = np.exp(-pairwise_distance / h**2 / 2)
        
        # gradient of RBF kernel
        gradient_rbf_kernel = -np.matmul(rbf_kernel, theta)
        kernel_sums = np.sum(rbf_kernel, axis=1)
        for i in range(theta.shape[1]):
            gradient_rbf_kernel[:, i] = gradient_rbf_kernel[:, i] + np.multiply(theta[:, i], kernel_sums)
        gradient_rbf_kernel = gradient_rbf_kernel / (h**2)

        # gradient theta
        gradient_theta = (np.matmul(rbf_kernel, lnprob(theta)) + gradient_rbf_kernel) # /

        