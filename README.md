# Timeseries-data-prediction-using-Transformers

This project of mine was predicting traffic data via transformers.

Usually we use LSTM, GRU model for this task, and I thought of trying transformer model.
I have implemented a complete transformer architechture with the help of builtin tools.

Since LSTM is heavy, transformers trained quicker. Transformer has its own weakness though. Since the way its model is designed,  if we have 250 samples, we will have to predict 250 samples (more like “input dimension” and “output dimension” has to be the same).
Multi-horizon predictions can still be done as long as we have the same dimension of input and target. In transformer,  there are few features received from the previous values and the prediction probably converged to an average value of the sequential data, resulting in smaller loss.  
So transformer might not be able to predict sequences that are way chaotic. I implemented LSTM as well and despite its higher loss, I think it's still a better model :)

As for the data sets, I used my TAs Research data sets so I will not be uploading them. 
Traffic data was generated via some Weibull distribution, so everything was already integers which allowed me to skip "preparing" the data. 

Data info : N x 10000 ( N= # data samples, 10000= # sequences)






