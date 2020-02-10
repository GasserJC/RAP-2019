import numpy as np
import tensorflow as tf
from tqdm import tqdm

from flearn.models.client import Client #client is the class for an indvidual node
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad

#clients are the nodes
#BaseFederated is a federated system of nodes

class BaseFedarated(object): #class for each a federated system
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);  

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed) #using client class objects
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    # Closes the object / closes the federated system
    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        # If groups is empty, then it does something with nulling for every underscore in users
        if len(groups) == 0:
            groups = [None for _ in users]
        # The data in users and groups is being indexed together with zip()
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        # Defines a group of arrays for keeping track of samples, losses, etc..
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:  # this loop tests all of the clients and returns their correct, loss, and # of samples 
            # ct = total correct, cl = loss, ns = number of samples
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct*1.0)  # Possibly appending the total correct from the previous line?
            num_samples.append(ns)  # Appending the number of samples into its array
            losses.append(cl*1.0)  # Appending losses to the array

        # Adds every id and group of the clients into an array
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses

    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size  # Processes the model length by getting size of latest model
        global_grads = np.zeros(model_len)  # Global gradient is a numpy array filled with as many zeroes as model_len

        intermediate_grads = []
        samples = []

        # The client's model is setting its parameters to match the latest_model
        self.client_model.set_params(self.latest_model)
        # For every client, get the number of samples and the client gradients, add them to appropriate numpy array
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples) #weighted client grad, mutliplies by number of samples.
            intermediate_grads.append(client_grads)

        # Multiplys the global grad by one and divides by the sum of samples converted into an array
        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)  # Stores the global gradient afterward

        return intermediate_grads

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        # Sets the parameters of the client to match the latest_model
        self.client_model.set_params(self.latest_model)
        # For every client, have them test and record the total correct/number of samples into their arrays
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        # Records the ids and groups of the clients
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        # Finds the smallest number between number of clients and the length of clients
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        # Something to do with random choice between the range of the length of clients and num_clients
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        # The result of the previous line is returned, along with self.clients as an array
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln
