# Copyright (C) Laboratory for Translation Medicine - All Rights Reserved
"""
This script is used to predict upcoming displacements using 
LSTMs. Before running it, ensure that the 
target displacements obtained through image registration 
have already been computed. The script utilizes these 
precomputed values to generate predictions for future displacements.
"""
import PredictionUtils
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, Input, TimeDistributed
import csv
# import time

# Preparing
directoryMeasurements = r'..\ProcessingResults'
prediction_horizon = 3 # Number of measurements to predict
window_size = 7 # Number of measurements for prediction
training_size = 60 # Number (60) of measurements for initial model training
nbr_file_process = 4 # We evaluate only on the first 4 datasets
nb_file_processed = 0
# all_times = []

# split a multivariate sequence into samples
def split_sequence(sequences, window_size, prediction_horizon):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + window_size
        out_end_ix = end_ix + prediction_horizon
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix], sequences[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_model(num_layers = 5, hidden_features = 15):#dropout = 0
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))
    model.add(LSTM(hidden_features, activation='relu'))
    model.add(RepeatVector(prediction_horizon))
    for _ in range(num_layers):
        model.add(LSTM(hidden_features, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to process each file
def process_file(params):
    directory, filename, training_size, window_size, prediction_horizon = params
    global all_times
        
    # Create models
    modelX, modelY, modelZ = create_model(), create_model(), create_model()

    df_full = pd.read_csv(os.path.join(directory, filename)) #reading the file
    df_full.drop(['Image', 'Registration Status'], axis=1, inplace=True)
    df = df_full.iloc[::2]  # Taking only 1 measurement out of 2 for X and Y
    
    # Place each column separately in an array
    dfX = pd.DataFrame(df['X'], columns=['X'])
    dfY = pd.DataFrame(df['Y'], columns=['Y'])
    dfZ = pd.DataFrame(df_full['Z'], columns=['Z'])
    
    temp, scalerY = PredictionUtils.augmentAndScale(dfY[dfY.index<training_size])
    dfY = np.concatenate((temp, scalerY.transform(dfY[dfY.index>=training_size].values), scalerY.transform(dfY[dfY.index>=(len(dfY)-prediction_horizon)].values))).reshape(-1, 1)
    j = (len(temp) - prediction_horizon - window_size) 
    
    temp, scalerX = PredictionUtils.augmentAndScale(dfX[dfX.index<training_size])
    dfX = np.concatenate((temp, scalerX.transform(dfX[dfX.index>=training_size].values), scalerX.transform(dfX[dfX.index>=(len(dfX)-prediction_horizon)].values))).reshape(-1, 1)
    
    temp, scalerZ = PredictionUtils.augmentAndScale(dfZ[dfZ.index<training_size])
    jZ = (len(temp) - prediction_horizon - window_size) 
    dfZ = np.concatenate((temp, scalerZ.transform(dfZ[dfZ.index>=training_size].values), scalerZ.transform(dfZ[dfZ.index>=(len(dfZ)-prediction_horizon)].values))).reshape(-1, 1)
    del temp

    # Split into samples into data instances X and labels y
    XdfX, ydfX = split_sequence(dfX, window_size, prediction_horizon)
    XdfX = XdfX.reshape((XdfX.shape[0], XdfX.shape[1], 1))
    XdfY, ydfY = split_sequence(dfY, window_size, prediction_horizon)
    XdfY = XdfY.reshape((XdfY.shape[0], XdfY.shape[1], 1))
    XdfZ, ydfZ = split_sequence(dfZ, window_size, prediction_horizon)
    XdfZ = XdfZ.reshape((XdfZ.shape[0], XdfZ.shape[1], 1))

    # start_time = time.time()
    modelX.fit(XdfX[:j + 2], ydfX[:j + 2], epochs=600, verbose=0, batch_size=2, shuffle=False)#because coronal acquired second
    # end_time = time.time()
    # print('time:', (end_time - start_time))
    # start_time = time.time()
    modelY.fit(XdfY[:j + 1], ydfY[:j + 1], epochs=600, verbose=0, batch_size=2, shuffle=False)#because sagittal acquired first
    # end_time = time.time()
    # print('time:', (end_time - start_time))
    # start_time = time.time()
    modelZ.fit(XdfZ[:jZ + 1], ydfZ[:jZ + 1], epochs=600, verbose=0, batch_size=2, shuffle=False)
    # end_time = time.time()
    # print('time:', (end_time - start_time))
    
    jp = j + prediction_horizon
    jpZ = jZ + prediction_horizon
    j += 2
    jZ += 2
    all_preds = []
    
    # Initial prediction of 61st image
    predX = modelX.predict(XdfX[jp+1].reshape(1, window_size, 1),verbose = 0)
    predY = modelY.predict(XdfY[jp].reshape(1, window_size, 1),verbose = 0)
    predZ = modelZ.predict(XdfZ[jpZ].reshape(1, window_size, 1),verbose = 0)
    predX = scalerX.inverse_transform(predX.reshape(-1, 1)) 
    predY = scalerY.inverse_transform(predY.reshape(-1, 1)) 
    predZ = scalerZ.inverse_transform(predZ.reshape(-1, 1)) 
    
    jp += 1
    jpZ += 1
    # 61st image (sagittal) just acquired. Retrain with 61st image (Y and Z just measured) and 62nd image (X) ahead of time but won't be used until the right time
    modelX.fit(XdfX[j:j+1], ydfX[j:j+1], epochs=1, verbose=0, batch_size=1, shuffle=False) #check if i train with 62, if so ok
    modelY.fit(XdfY[j-1:j], ydfY[j-1:j], epochs=1, verbose=0, batch_size=1, shuffle=False)
    modelZ.fit(XdfZ[jZ-1:jZ], ydfZ[jZ-1:jZ], epochs=1, verbose=0, batch_size=1, shuffle=False)   

    all_preds.append([predX[0][0], predY[0][0], predZ[0][0],
                      predX[1][0], predY[1][0], predZ[1][0],  
                      predX[2][0], predY[2][0], predZ[2][0]])
    
    while jpZ < len(XdfZ)-1:
        # Y was just measured - predict 63rd image - Y displacement
        # start_time = time.time()
        predY = modelY.predict(XdfY[jp].reshape(1, window_size, 1),verbose = 0)
        predY = scalerY.inverse_transform(predY.reshape(-1, 1)) #predY.reshape(-1, 1)#
        j += 1
        jp += 1
        # 63rd image (sagittal) just acquired. Retrain with 63rd image
        modelY.fit(XdfY[j-1:j], ydfY[j-1:j], epochs=1, verbose=0, batch_size=1, shuffle=False)
        # end_time = time.time()
        # all_times.append(end_time - start_time)
        
        # Z was just measured - predict 62nd image - Z displacement
        # start_time = time.time()
        predZ = modelZ.predict(XdfZ[jpZ].reshape(1, window_size, 1),verbose = 0)
        predZ = scalerZ.inverse_transform(predZ.reshape(-1, 1)) #predZ.reshape(-1, 1)#
        jZ += 1
        jpZ += 1 # 62nd image just acquired. Retrain with 62nd image
        modelZ.fit(XdfZ[jZ-1:jZ], ydfZ[jZ-1:jZ], epochs=1, verbose=0, batch_size=1, shuffle=False)
        # end_time = time.time()
        # all_times.append(end_time - start_time)
        
        all_preds.append([predX[0][0], predY[0][0], predZ[0][0],
                          predX[1][0], predY[1][0], predZ[1][0],  
                          predX[2][0], predY[2][0], predZ[2][0]])
        
        if jpZ > len(XdfZ)-2:
            break
        
        # X was just measured - predict 64th image
        # start_time = time.time()
        predX = modelX.predict(XdfX[jp].reshape(1, window_size, 1),verbose = 0)
        predX = scalerX.inverse_transform(predX.reshape(-1, 1)) #predX.reshape(-1, 1)#
        # 64th image just acquired. Retrain with 64th image
        modelX.fit(XdfX[j:j+1], ydfX[j:j+1], epochs=1, verbose=0, batch_size=1, shuffle=False)
        # end_time = time.time()
        # all_times.append(end_time - start_time)
        
        # Z was just measured - predict 63rd image - Z displacement
        # start_time = time.time()
        predZ = modelZ.predict(XdfZ[jpZ].reshape(1, window_size, 1),verbose = 0)
        predZ = scalerZ.inverse_transform(predZ.reshape(-1, 1)) #predZ.reshape(-1, 1)#
        jZ += 1
        jpZ += 1  # 63rd image just acquired. Retrain with 63rd image
        modelZ.fit(XdfZ[jZ-1:jZ], ydfZ[jZ-1:jZ], epochs=1, verbose=0, batch_size=1, shuffle=False)
        # end_time = time.time()
        # all_times.append(end_time - start_time)

        all_preds.append([predX[0][0], predY[0][0], predZ[0][0],
                          predX[1][0], predY[1][0], predZ[1][0],  
                          predX[2][0], predY[2][0], predZ[2][0]])
    
    filepath_pred_data = os.path.join(directory, 'LSTM_Prediction_' + filename.split('_')[1] + '.csv')
    print('saving in', filepath_pred_data)
    # Writing data to CSV file
    with open(filepath_pred_data, 'w', newline='') as csvPreds:
        csvPreds_writer = csv.writer(csvPreds)
        csvPreds_writer.writerow(['X', 'Y', 'Z', 'X+2', 'Y+2', 'Z+2', 'X+3', 'Y+3', 'Z+3'])
        csvPreds_writer.writerows(all_preds)
        # Adding an extra line for same length as other predictive methods
        csvPreds_writer.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0])
           
    # return all_times
            
# Main calling point
# Loop through each file in the directory
tasks = []
for filename in os.listdir(directoryMeasurements):
    if not os.path.isfile(os.path.join(directoryMeasurements, filename)):
        continue # If this is not a file, continue
    if nbr_file_process > nb_file_processed and '_Measurement' in filename:  # taking only measurement files
        print(filename)
        nb_file_processed += 1  
        tasks.append((directoryMeasurements, filename, training_size, window_size, prediction_horizon))

# Process each task
for task in tasks:
    all_times = process_file(task)

# Optionally print average time taken for predictions
# print(np.average(all_times))
            