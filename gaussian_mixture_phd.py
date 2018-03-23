import numpy as np
from scipy.stats import chi2
from utils import Model, Measurement, Truth

def kalman_predict_multiple(model, m, P):
    plength = m.shape[1]
    m_predict = np.zeros(m.shape)
    P_predict = np.zeros(P.shape)

    for i in xrange(plength):
        pass

def kalman_predict_single(F, Q, m, P):
    pass
    #m_predict = np.

class GaussianMixturePHD:
    def __init__(self, model, meas):
        # Output variables
        self.X = [np.array([])] * meas.K
        self.N = np.zeros(meas.K)
        self.L = [np.array([])] * meas.K
        # Filter parameters
        self.L_max = 100 # Limit on number of Gaussians
        self.elim_threshold = 1e-5 # pruning threshold
        self.merge_threshold = 4 # merging threshold
        self.P_G = 0.999 # gate size in percentage
        self.gamma = chi2.ppf(self.P_G, model.z_dim) # inv chi^2 dn gamma value
        self.gate_flag = True # gating on or off True/False
        self.run_flag = 'disp' # 'disp' or 'silence' for on the fly output
        self.pruning = 'default' # 'default' or 'dpp'
        self.initialized = False

    def initialize(self):
        self.w_update = np.array([ [ np.finfo(np.float64).eps ] ])
        self.m_update = np.array([ [0.1], [0.0], [0.1], [0.0] ])
        self.P_update = np.diag([1, 1, 1, 1])**2
        self.L_update = 1
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def predict(self, observations):
        if self.is_initialized():
            pass

if __name__ == '__main__':
    model = Model()
    truth = Truth(model)
    meas = Measurement(model, truth)

    filter = GaussianMixturePHD(model, meas)

    for observations in meas.Z:
        if not(filter.is_initialized()):
            filter.initialize()
        else:
            filter.predict(observations)