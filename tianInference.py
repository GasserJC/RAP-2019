# this program is currently only going to be psuedocode and ideas
# the end goal is initiating a simulated inference using tian's code.

#Large issue one, tians code is only set up to train, not test. --> create a new file
#Large issue two, incompatability, it is possible my code/resources will not work on older tensorflow  --> use tensorflow2/tf2 federated.

#current approach, use weights = layer.get_weights() to pull weights from a single layer. We will start with a non-dynamic,
#single chain model to reduce initial challenges.

#going to use a new class, a child of Server

from .fedavg import Server

class InferenceServer(Server):
    def __init__(self, params, learner, dataset): #constructor
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate']) # learning rate
        # Calls the base class to send the arguments needed for its constructor
        super(InferenceServer, self).__init__(params, learner, dataset)
        
    def test(self):
        self.model.test(self.eval_data)
         
        


    
