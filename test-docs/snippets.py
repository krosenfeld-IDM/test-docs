'''
Examples from:
https://onefishy.github.io/Rwanda-Data-Science/

'''
import numpy as np

__all__ = ['get_posterior_samples', 'generate_data']

def get_posterior_samples(prior_var, noise_var, x_matrix, y_matrix, x_test_matrix, samples=100):
    prior_variance = np.diag(prior_var * np.ones(x_matrix.shape[1]))
    prior_precision = np.linalg.inv(prior_variance)

    joint_precision = prior_precision + x_matrix.T.dot(x_matrix) / noise_var
    joint_variance = np.linalg.inv(joint_precision)
    joint_mean = joint_variance.dot(x_matrix.T.dot(y_matrix)) / noise_var

    #sampling 100 points from the posterior
    posterior_samples = np.random.multivariate_normal(joint_mean.flatten(), joint_variance, size=samples)

    #take posterior predictive samples
    posterior_predictions = np.dot(posterior_samples, x_test_matrix.T) 
    posterior_predictive_samples = posterior_predictions + np.random.normal(0, noise_var**0.5, size=posterior_predictions.shape)

    return posterior_predictions, posterior_predictive_samples

def generate_data(number_of_points=10, noise_variance=0.5):
    #training x
    x = np.hstack((np.linspace(-10, -5, number_of_points), np.linspace(5, 10, number_of_points)))
    #function relating x and y
    f = lambda x:  0.01 * x**3
    #y is equal to f(x) plus gaussian noise
    y = f(x) + np.random.normal(0, noise_variance**0.5, 2 * N)
    x_test = np.linspace(-11, 11, 100)
    return x, y, x_test
