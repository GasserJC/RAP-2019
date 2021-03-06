import numpy as np             #mathematics library
from tqdm import trange, tqdm  #progress bar
import tensorflow as tf        #machine learning library

from .fedbase import BaseFedarated  #importing parent class from FedBase.py
from flearn.utils.tf_utils import process_grad  #there is no process_grad in that file.


class Server(BaseFedarated): #creating server class, child class of BaseFederated
    def __init__(self, params, learner, dataset): #constructor
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate']) # learning rate
        # Calls the base class to send the arguments needed for its constructor
        super(Server, self).__init__(params, learner, dataset)

    def train(self): #this function initiates the training of the clients/nodes
                     #self refers to the Server/Entire federated system
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round)) #pritns # of participating nodes
        
        for i in range(self.num_rounds): #num_rounds is a member of BaseFederated and passed down to server
            # test model
            if i % self.eval_every == 0: # << in client class
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()  # Have the clients train on the model and record errors

                # The test accuracy, training accuracy, and loss are all outputted
                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # Re-seeds the generator
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)

            csolns = []  # buffer for receiving client solutions

            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

                # ------------------START OF CODE CHANGE WE WILL APPLY A TOP-K HERE--------------------#

                # this reformats the csolns matrix into a one dimensional array
                csolns.reshape(,)  # have to dynamically reshape due to matrix size                
                # start of the manipulation code
                weight = csolns.copy()

                # top k amount, it is technically a decimal, not a percent
                top_k_percent = .1  # ten percent

                # creates the weight array in descending order
                topValue = weight.copy()
                topValue.sort()
                topValue.reverse()

                # creates two arrays for manipulation of weight array
                evalWeight = weight.copy()
                residWeight = weight.copy()

                # findes the top-k value
                indexVal = int((len(topValue) - 1) * top_k_percent)
                topK = topValue[indexVal]

                # makes all values below topK zero
                for w in range(0, len(evalWeight)):
                    if evalWeight[w] <= topK:
                        evalWeight[w] = 0

                # keeps an array of only residules
                for w in range(0, len(residWeight)):
                    if residWeight[w] > topK:
                        residWeight[w] = 0

                # makes csolns evaluation
                csolns = evalWeight.copy()
                # end of topK code

                # change csolns back into a multidimensional array
                csolns.reshape(,)  # dynamically reshape back to csolns orginal matrix size.

                # ------------END OF TOP K ADDITION---------------------------

            # update models
            self.latest_model = self.aggregate(csolns)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
