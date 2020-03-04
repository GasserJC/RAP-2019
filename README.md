# RAP-2019


Inference on Tians Code Progress:

Was able to simulate inference on the guess Name code. 
Results / learned:
accuracy as a whole will decrease, as the method I used was implementing two images of one whole model. Image one being the client and the first half of the model, and the server model which was the last half. While easy to implement this code has obvious negatives - firstly the models are static and cannot change due to communicational strain, secondly, we have to train a half model, never the full model all at once, this decreases final accuracy. This was a usefull expriment as it shows a way to easily access the weights in the program. We should not move forward with the partial image approach as it has lower dynamic capability and accuracy. I am still looking to see a way to dynamically 'break' the model and return the last/current weights and transmit them at a dynamic point in another model. This task has high accuracy and dynamic capability, however lacks in ease of implementation and availible resources for help.
