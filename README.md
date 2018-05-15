# Graduation Project
## Example Model
```python
batch_size = 1
timesteps = 303
data_dim = 6

model = Sequential()
model.add(LSTM(128, return_sequences=True, batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(64))
model.add(Dense(32))
model.add(Dense(4, activation='softmax'))
```
### Summary of the model and total number of parameters:
```python
model.summary()
```
```python
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_7 (LSTM)                (1, 303, 128)             69120     
_________________________________________________________________
lstm_8 (LSTM)                (1, 64)                   49408     
_________________________________________________________________
dense_7 (Dense)              (1, 32)                   2080      
_________________________________________________________________
dense_8 (Dense)              (1, 4)                    132       
=================================================================
Total params: 120,740
Trainable params: 120,740
Non-trainable params: 0
"""
```
## Running Time(seconds) Comparison with Different Configurations
|      Batch/Device/Run. Time      | LSTM  | CuDNNLSTM |
| :------------------------------: | :---: | :-------: |
| Batch Size:**1** Device: **CPU** |  724  |     -     |
| Batch Size:**8** Device: **CPU** |**116**|     -     |
| Batch Size:**1** Device: **GPU** | 3936  |    251    |
| Batch Size:**8** Device: **GPU** |  520  |   **36**  |
