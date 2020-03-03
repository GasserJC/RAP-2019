# this program is currently only going to be psuedocode and ideas
# the end goal is initiating a simulated inference using tian's code.


#current approach, use weights = layer.get_weights() to pull weights from a single layer. We will start with a non-dynamic,
#single chain model to reduce initial challenges.

#implement get weights during testing: (diffuculty) 
#1st, have a dynamic model (easy)
#2nd, have the test function return the weights (very hard)
#3rd, transmit to the server class the weights (easy)
#4th, train the residule weights on the new model (medium)
#5th, return the final answer to the client, ( easy, we will start with classifaction, therefore we recieve an array resulting from softMax. )
#6th, client returns the final answer given the result from server (easy)
         
        
#1) the code already is set up for dynamic model selection, therefore feeding the client a new model is easy.
#2) Two methods: having a model that output layer == to the final layers weights (very-hard) or using a getWeights function that
#.. can retrieve the weights at a layer and dynamically break the testing (prefered, (very-hard)).
#3) The rest should be simple simulation.


#Dynamic model selection means choosing when to break during testing, this would most easily be done by updating the model every x amount
#seconds/tests/units. We would implement a set_models(model) function.

#After dynamic breaking can be achieved we need to find away to get weights from the last layer. After recieving the weights we can 
#easily transmit
    
