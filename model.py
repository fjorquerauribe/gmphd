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

        print self.B_birth.shape
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

model = Model()

print model.A0
print model.H
print model.D
print model.B.shape