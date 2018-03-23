import numpy as np

class Model:
    def __init__(self):
        # Basic parameters
        self.x_dim = 4
        self.z_dim = 2
        # Dynamical model parameters
        self.T = 1 # Sampling period
        self.A0 = np.array([ [1.0, self.T], [0.0, 1.0]]) # Transition matrix
        self.F = np.concatenate( ( np.concatenate((self.A0, np.zeros((2,2))), axis = 1), np.concatenate((np.zeros((2,2)), self.A0), axis = 1) ) )
        self.B0 = np.array([[ float(self.T**2)/2.0, self.T ]])
        self.B = np.concatenate( (np.concatenate((self.B0, np.zeros((1,2))), axis = 1), np.concatenate((np.zeros((1,2)), self.B0), axis = 1))).T
        self.sigma_V = 5.0
        self.Q = self.sigma_V**2 * np.matmul(self.B, self.B.T) # Process noise covariance
        # Survival/death parameters
        self.P_S = 0.99
        self.Q_S = 1.0 - self.P_S
        # Birth parameters (Poisson birth model, multiple Gaussian components)
        self.L_birth = 4 # Number of Gaussian birth terms
        self.w_birth = np.zeros((4,1)) # Weights of Gaussian birth terms (per scan) [sum gives average rate of target birth per scan]
        self.m_birth = np.zeros((self.x_dim, self.L_birth)) # Means of Gaussian birth terms
        self.B_birth = np.zeros((self.x_dim, self.x_dim, self.L_birth)) # Std of Gaussian birth terms
        self.P_birth = np.zeros((self.x_dim, self.x_dim, self.L_birth)) # Cov of Gaussian birth terms
        # Birth term 1
        self.w_birth[0] = 3.0/100
        self.m_birth[:,0] = np.array([0, 0, 0, 0])
        self.B_birth[:,:,0] = np.diag((10.0, 10.0, 10.0, 10.0))
        self.P_birth[:,:,0] = np.matmul(self.B_birth[:,:,0], self.B_birth[:,:,0].T)
        # Birth term 2
        self.w_birth[1] = 3.0/100
        self.m_birth[:,1] = np.array([400, 0, -600, 0])
        self.B_birth[:,:,1] = np.diag((10.0, 10.0, 10.0, 10.0))
        self.P_birth[:,:,1] = np.matmul(self.B_birth[:,:,1], self.B_birth[:,:,1].T)
        # Birth term 3
        self.w_birth[2] = 3.0/100
        self.m_birth[:,2] = np.array([-800, 0, -200, 0])
        self.B_birth[:,:,2] = np.diag((10.0, 10.0, 10.0, 10.0))
        self.P_birth[:,:,2] = np.matmul(self.B_birth[:,:,2], self.B_birth[:,:,2].T)
        # Birth term 4
        self.w_birth[3] = 3.0/100
        self.m_birth[:,3] = np.array([-200, 0, 800, 0])
        self.B_birth[:,:,3] = np.diag((10.0, 10.0, 10.0, 10.0))
        self.P_birth[:,:,3] = np.matmul(self.B_birth[:,:,3], self.B_birth[:,:,3].T)
        # Observation model parameters (noise x/y only)
        self.H = np.array([ [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0] ]) # Observation matrix
        self.D = np.diag((10.0, 10.0))
        self.R = np.matmul(self.D, self.D.T) # Observation noise covariance
        # Detection parameters
        self.P_D = 0.98 # Probability of detection in measurements
        self.Q_D = 1.0 - self.P_D # Probability of missed detection in measurements
        # Clutter parameters
        self.lambda_c = 60.0 # Poisson average rate of uniform clutter (per scan)
        self.range_c = np.array([ [-1000.0, 1000.0], [-1000.0, 1000.0] ]) # Uniform clutter region
        self.pdf_c = 1.0/np.prod( self.range_c[:,0] - self.range_c[:,1] )

class Truth:
    def __init__(self, model):
        # Variables
        self.K = 100 # Length of data/number of scans
        self.X = [np.array([])] * self.K # Ground truth for states of targets
        self.N = np.zeros(self.K, dtype = int) # Ground truth for number of targets
        self.L = [[]] * self.K # Ground truth for labels of targets (k,i)
        self.track_list = [np.array([])] * self.K # Absolute index target identities (plotting)
        self.total_tracks = 0.0 # Total number of appearing tracks
        # Target initial states and birth/death times
        nbirths = 12
        xstart = np.array([ [0.0, 0.0, 0.0, -10.0],
                            [400.0, -10.0, -600.0, 5.0],
                            [-800.0, 20.0, -200.0, -5.0],
                            [400.0, -7.0, -600.0, -4.0],
                            [400.0, -2.5, -600.0, 10.0],
                            [0.0, 7.5, 0.0, -5.0],
                            [-800.0, 12.0, -200.0, 7.0],
                            [-200.0, 15.0, 800.0, -10.0],
                            [-800.0, 3.0, -200.0, 15.0],
                            [-200.0, -3.0, 800.0, -15.0],
                            [0.0, -20.0, 0.0, -15.0],
                            [-200.0, 15.0, 800.0, -5.0] ])
        tbirth = np.array([0, 0, 0, 20, 20, 20, 40, 40, 60, 60, 80, 80])
        tdeath = np.array([70, self.K + 1, 70, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1])
        # Generate the tracks
        for target_num in xrange(nbirths):
            target_state = xstart[target_num,:]
            for k in xrange(tbirth[target_num], min(tdeath[target_num],self.K)):
                target_state = gen_newstate_fn(model, target_state, 'noiseless')
                #print target_state
                if not(self.X[k].size):
                    self.X[k].shape = (0, model.x_dim)
                self.X[k] = np.append(self.X[k], target_state, axis = 0)
                target_state = target_state[0,:]
                self.track_list[k] = np.append(self.track_list[k], [target_num])
                self.N[k]+=1
        self.total_tracks = nbirths

class Measurement:
    def __init__(self, model, truth):
        # Variables
        self.K = truth.K
        self.Z = [np.array([])] * self.K
        # Generate measurements
        for k in xrange(self.K):
            if truth.N[k] > 0:
                (idx, _) = np.where( np.random.rand(truth.N[k], 1) <= model.P_D ) # detected target indices
                if not(self.Z[k].size):
                    self.Z[k].shape = (0, model.z_dim)
                self.Z[k] = gen_observation_fn(model, truth.X[k][idx,:], 'noise') # single target observations if detected


def gen_newstate_fn(model, Xd, V):
    if V == 'noise':
        V = np.matmul(model.sigma_V * model.B, np.random.rand(model.B.shape[1], 1))
    elif V == 'noiseless':
        V = np.zeros((model.B.shape[0], 1))
    if not(Xd.size): #if isempty
        X = np.array([[]])
    else:
        X = np.matmul(model.F, Xd.T) + V.T
    return X

def gen_observation_fn(model, X, W):
    if W == 'noise':
        W = np.matmul(model.D, np.random.rand(model.D.shape[1], X.shape[0]))
    elif W == 'noiseless':
        W = np.zeros((model.D.shape[0]), 1)
    if not(X.size):
        Z = np.array([[]])
    else:
        Z = np.matmul(model.H, X.T) + W
    return Z.T

if __name__== '__main__':
    model = Model()
    truth = Truth(model)
    meas = Measurement(model, truth)