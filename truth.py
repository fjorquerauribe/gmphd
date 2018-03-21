import numpy as np

def gen_newstate_fn(model, Xd, V):
    if V == "noise":
        V = model.sigma_V * model.B * np.random.rand(model.B.shape[1], Xd.shape[1])
    elif V == "noiseless":
        V = np.zeros(model.B.shape[0], Xd.shape[1])

    if Xd.size:
        X = []
    else:
        X = model.F * Xd + V
    
    return X
    

class Truth:
    def __init__(self, model):
        # Variables
        self.K = 100 # Length of data/number of scans
        self.X = [[]] * self.K # Ground truth for states of targets
        self.N = np.zeros(self.K) # Ground truth for number of targets
        self.L = [[]] * self.K # Ground truth for labels of targets (k,i)
        self.track_list = [0] * self.K # Absolute index target identities (plotting)
        self.total_tracks = 0.0 # Total number of appearing tracks

        # Target initial states and birth/death times
        nbirths = 12
        
        xstart = np.array([ [0.0, 400.0, -800.0, 400.0, 400.0, 0.0, -800.0, -200.0, -800.0, -200.0, 0.0, -200.0],
                            [0.0, -10.0, 20.0, -7.0, -2.5, 7.5, 12.0, 15.0, 3.0, -3.0, -20.0, 15.0],
                            [0.0, -600.0, -200.0, -600.0, -600.0, 0.0, -200.0, 800.0, -200.0, 800.0, 0.0, 800.0],
                            [-10.0, 5.0, -5.0, -4.0, 10.0, -5.0, 7.0, -10.0, 15.0, -15.0, -15.0, -5.0] ])
        tbirth = np.array([1, 1 , 1, 20, 20, 20, 40, 40, 60, 60, 80, 80])
        tdeath = np.array([70, self.K + 1, 70, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1, self.K + 1])
        
        # Generate the tracks
        for target_num in xrange(nbirths):
            target_state = xstart[:,target_num]
            print tdeath[target_num]
            for k in xrange(tbirth[target_num], min(tdeath[target_num],self.K)):
                target_state = gen_newstate_fn(model, target_state, 'noiseless')
                #target_state =

truth = Truth(None)