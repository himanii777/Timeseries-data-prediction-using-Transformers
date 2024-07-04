# Timeseries-data-prediction-using-Transformers

Usually we use LSTM, GRU model for this task, I thought of using transformer model.
I have implemented a complete transformer architechture with the help of builtin pytorch tools

Since LSTM is heavy, transformers train quicker. Transformer has its own weakness though. Since the way its model is designed,  if we have 250 samples, we will have to predict 250 samples (more like “input dimension” and “output dimension” has to be the same).
Multi horizon predictions can still be done as long as we have same dimension of input and target. In transformer,  there are only few features received from the previous values and the prediction probably converged to average value of the sequential data, resulting in lower loss.  
So transformer might not be able to predict sequences that are way chaotic. I implemented LSTM as well and despite its higher loss, I think its still a good model :)




