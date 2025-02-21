# Copyright (C) Laboratory for Translation Medicine - All Rights Reserved
"""
This script is used to predict upcoming displacements using 
LightGBM. Before running it, ensure that the 
target displacements obtained through image registration 
have already been computed. The script utilizes these 
precomputed values to generate predictions for future displacements.
"""
import PredictionUtils
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sktime.forecasting.compose import make_reduction
import csv
# import time

# Preparing
directoryMeasurements = r'..\ProcessingResults'
prediction_horizon = 3 # Number of measurements to predict
window_size = 14 # Number of measurements for prediction
training_size = 60 # Number (60) of measurements for initial model training
nbr_file_process = 4 # We evaluate only on the first 4 datasets
nb_file_processed = 0
# all_times = []
params = {
    "max_depth": 10, # If larger prediction time increases
    "num_leaves": 15,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "min_data_in_leaf": 9,
    'verbosity': -1  
}

# Initialize LGBM models
regressorX = lgb.LGBMRegressor(**params)
regressorY = lgb.LGBMRegressor(**params)
regressorZ = lgb.LGBMRegressor(**params)

# Creating forecasters with recursive strategy
forecasterX = make_reduction(regressorX, window_length=window_size, strategy="recursive")
forecasterY = make_reduction(regressorY, window_length=window_size, strategy="recursive")
forecasterZ = make_reduction(regressorZ, window_length=window_size, strategy="recursive")
fh = np.arange(prediction_horizon) + 1

# Loop through each file in the directory
for filename in os.listdir(directoryMeasurements):
    if not os.path.isfile(os.path.join(directoryMeasurements, filename)):
        continue # If this is not a file, continue
    if nbr_file_process > nb_file_processed and '_Measurement' in filename:  # taking only measurement files
        print(filename)
        
        # Reading the data file and extracting the measurements
        df_full = pd.read_csv(os.path.join(directoryMeasurements, filename))  # Read the file
        df_full.drop(['Image', 'Registration Status'],axis=1, inplace=True)
        df = df_full.iloc[::2]  # Take every second row (for X and Y)

        dfX = pd.DataFrame(df['X'], columns=['X'])
        dfY = pd.DataFrame(df['Y'], columns=['Y'])
        dfZ = pd.DataFrame(df_full['Z'], columns=['Z'])

        # Augmenting the training set
        trainX = PredictionUtils.augment(dfX[dfX.index<training_size])
        testX = dfX[dfX.index >= training_size].values.reshape(-1, 1)
        trainY = PredictionUtils.augment(dfY[dfY.index<training_size])
        testY = dfY[dfY.index >= training_size].values.reshape(-1, 1)
        trainZ = PredictionUtils.augment(dfZ[dfZ.index<training_size])
        # Averaging available superior/inferior measurements for stability
        trainZ = (trainZ[:-1]+trainZ[1:])/2
        testZ = dfZ[dfZ.index >= training_size - 1].values.reshape(-1, 1)
        testZ = (testZ[:-1] + testZ[1:]) / 2

        forecasterX.fit(trainX)
        predX = forecasterX.predict(fh=fh)
        forecasterY.fit(trainY)
        predY = forecasterY.predict(fh=fh)
        forecasterZ.fit(trainZ)
        predZ = forecasterZ.predict(fh=fh)
        all_preds = []
        all_preds.append([predX[0][0], predY[0][0], predZ[0][0],
                          predX[1][0], predY[1][0], predZ[1][0],  ## Hard-coded for now. To be improved. 
                          predX[2][0], predY[2][0], predZ[2][0]]) ## We are reporting the 3 forecasted measurements
            
        # orientation image is now 'coronal'
        while len(testZ) != 0: # Retraining when new measurement is available 
            
            # Y was just measured
            # start_time = time.time()
            trainY = np.append(trainY, testY[0])
            testY = np.delete(testY,0)
            forecasterY.fit(trainY)
            predY = forecasterY.predict(fh=fh)
            # end_time = time.time()
            # all_times.append(end_time - start_time)
            
            # start_time = time.time()
            trainZ = np.append(trainZ, testZ[0])
            testZ = np.delete(testZ,0)
            forecasterZ.fit(trainZ)
            predZ = forecasterZ.predict(fh=fh)
            # end_time = time.time()
            # all_times.append(end_time - start_time)
            
            all_preds.append([predX[0][0], predY[0][0], predZ[0][0],
                              predX[1][0], predY[1][0], predZ[1][0],  ## Hard-coded for now. To be improved. 
                              predX[2][0], predY[2][0], predZ[2][0]]) ## We are reporting the 3 forecasted measurements
            
            # Check we are not done with the data
            if len(testZ)==0:
                break
            
            # X was just measured
            # start_time = time.time()
            trainX = np.append(trainX, testX[0])
            testX = np.delete(testX,0)
            forecasterX.fit(trainX)
            predX = forecasterX.predict(fh=fh)
            # end_time = time.time()
            # all_times.append(end_time - start_time)
            
            # start_time = time.time()
            trainZ = np.append(trainZ, testZ[0])
            testZ = np.delete(testZ,0)
            forecasterZ.fit(trainZ)
            predZ = forecasterZ.predict(fh=fh)    
            # end_time = time.time()
            # all_times.append(end_time - start_time)
            
            all_preds.append([predX[0][0], predY[0][0], predZ[0][0],
                              predX[1][0], predY[1][0], predZ[1][0],  ## Hard-coded for now. To be improved. 
                              predX[2][0], predY[2][0], predZ[2][0]]) ## We are reporting the 3 forecasted measurements
        
        # Save predictions to a CSV file
        filepath_pred_data = os.path.join(
            directoryMeasurements, 'LGBM_Prediction_' + filename.split('_')[1] + '.csv')
        with open(filepath_pred_data, 'w', newline='') as csvPreds:
            csvPreds_writer = csv.writer(csvPreds)
            csvPreds_writer.writerow(['X', 'Y', 'Z', 'X+2', 'Y+2', 'Z+2', 'X+3', 'Y+3', 'Z+3']) # Write headers ## Hard-coded for now. To be improved. 
            csvPreds_writer.writerows(all_preds) 
        
        nb_file_processed += 1  
            
# print(np.average(all_times))
            