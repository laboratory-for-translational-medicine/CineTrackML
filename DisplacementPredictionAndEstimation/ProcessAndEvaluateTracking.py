# Copyright (C) Laboratory for Translation Medicine - All Rights Reserved
"""
This is the final script in the workflow.
It processes the prediction data and estimates the target's
3D displacement for each image, from image 60.
Additionally, a comprehensive analysis about the computed 
target displacements using all methods is performed.
"""
import os
import math
import pandas as pd
import numpy as np
import csv

strategy = 'LR' # Value van only be LR or LSTM or LGBM
orientationFirstImage = 'sagittal'
directory = r'..\ProcessingResults'
directoryGT = r'..\GroundTruth'
outFilename = strategy + "_FinalResults.csv"
trainingSize = 60
nbr_file_process = 4

def displacementsFromMeasurementsAndPredictions200Ahead(predictions, measurements_full, orientationFirstImage='sagittal', trainingSize = 60):
    displacements = []
    for i in range(len(predictions)):
        if orientationFirstImage == 'sagittal': # X is first missing
            displacements.append([(predictions['X'][i] + measurements_full['X'][trainingSize+i])/2,#X was measured just before 
                              measurements_full['Y'][trainingSize+i], 
                              measurements_full['Z'][trainingSize+i]])   
            orientationFirstImage = 'coronal'
            
        else: # Y is missing
            displacements.append([measurements_full['X'][trainingSize+i], 
                              (predictions['Y'][i]+ measurements_full['Y'][trainingSize+i])/2,
                              measurements_full['Z'][trainingSize+i]])   
            orientationFirstImage = 'sagittal'
    return displacements

def displacementsFromPredictions200Ahead(predictions, measurements_full, orientationFirstImage='sagittal', trainingSize = 60):
    displacements = []
    for i in range(len(predictions)):
        if orientationFirstImage == 'sagittal': # X is first missing
            displacements.append([(predictions['X'][i] + measurements_full['X'][trainingSize+i])/2,# X was measured 200 ms ago
                              predictions['Y'][i], # Y was measured 400 ms ago
                              (predictions['Z'][i]+predictions['Z+2'][i])/2])   # Z was measured 200 ms ago but due to filtering, its imut was 300 ms ago
            orientationFirstImage = 'coronal'

        else: # Y is missing
            displacements.append([predictions['X'][i],#X was measured 400 ms ago
                              (predictions['Y'][i]+ measurements_full['Y'][trainingSize+i])/2, # Y was measured 200 ms ago
                              (predictions['Z'][i]+predictions['Z+2'][i])/2])   # Z was measured 200 ms ago but due to filtering, its imut was 300 ms ago
            orientationFirstImage = 'sagittal'
    return displacements

def displacementsFromPredictions400Ahead(predictions, orientationFirstImage='sagittal'):
    displacements = []
    for i in range(1,len(predictions)):
        if orientationFirstImage == 'sagittal': # X is first missing
            displacements.append([(predictions['X+2'][i-1] + predictions['X'][i-1])/2,# X was measured 200 ms ago
                              predictions['Y'][i-1], # Y was measured 400 ms ago
                              (predictions['Z+3'][i-1]+predictions['Z+2'][i-1])/2])   # Z was measured 200 ms ago but due to filtering, its imut was 300 ms ago
            orientationFirstImage = 'coronal'
            
        else: # Y is missing
            displacements.append([predictions['X'][i-1],# X was measured 400 ms ago
                              (predictions['Y+2'][i-1] + predictions['Y'][i-1])/2,# Y was measured 200 ms ago
                              (predictions['Z+3'][i-1]+predictions['Z+2'][i-1])/2])   # Z was measured 200 ms ago but due to filtering, its imut was 300 ms ago
            orientationFirstImage = 'sagittal'
    return displacements

def displacementsFromPredictions600Ahead(predictions, orientationFirstImage='sagittal'):
    displacements = []
    for i in range(2,len(predictions)):
        if orientationFirstImage == 'sagittal': # X is first missing
            displacements.append([(predictions['X+3'][i-2] + predictions['X+2'][i-2])/2,#X was measured just before 
                              predictions['Y+2'][i-2], 
                              predictions['Z+3'][i-2]])   
            orientationFirstImage = 'coronal'
            
        else: # Y is missing
            displacements.append([predictions['X+2'][i-2],#X was measured just before 
                              (predictions['Y+3'][i-2] + predictions['Y+2'][i-2])/2,
                              predictions['Z+3'][i-2]])   
            orientationFirstImage = 'sagittal'
    return displacements

def distance(coord1,coord2):
    if len(coord1) != len(coord2):
        raise Exception("The DataFrames don't have the same length!")
    distances = []
    for i in range(len(coord1)):
        distances.append(math.sqrt((coord1['X'][i] - coord2['X'][i])**2 + (coord1['Y'][i] - coord2['Y'][i])**2 + (coord1['Z'][i] - coord2['Z'][i])**2))
    return distances

if __name__ == "__main__":
    
    # Write the results in CSV file
    with open(os.path.join(directory, outFilename), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', "p000_gt", 'meas_gt', 
                         "p200_gt", 'meas_gt_200ahead', 
                         'p400_gt_400ahead', 'meas_gt_400ahead', 
                         'p600_gt_600ahead','meas_gt_600ahead',
                         'dist_peak_peak','dist_gt_gt_200'])

        all_avg = []
        for file in os.listdir(directoryGT):
            if 'GroundTruth' not in file or not os.path.isfile(os.path.join(directoryGT, file)):
                continue
            else:
                gt_file = os.path.join(directoryGT, file)
                measurement_file = ''
                pred_files = ''
            if nbr_file_process == 0:
                break
            for filename in os.listdir(directory):
                if 'Measurement' in filename and file.split('_')[1].split('.')[0] in filename:
                    measurement_file = os.path.join(directory, filename)
                elif 'Prediction_' in filename and file.split('_')[1] in filename and strategy in filename:
                    pred_files = os.path.join(directory, filename)
            
            if len(gt_file) != 0 and len(pred_files) != 0: # found the 3 corresponding files
                nbr_file_process -= 1
                
                meas_full = pd.read_csv(os.path.join(directory, measurement_file)) #reading the measurements
                gt_full = pd.read_csv(os.path.join(directory, gt_file)) #reading the ground truth
                pred_full = pd.read_csv(os.path.join(directory, pred_files)) #reading the predictions
                pred_full = pred_full.iloc[:-1] # Last predicted displacements: neither the measurements nor the ground truth will be available.
                                       
                # Compute the displacements using only the predictions
                p000 = displacementsFromMeasurementsAndPredictions200Ahead(pred_full, meas_full)  # first image is sagittal at +200
                p000 = pd.DataFrame(p000, columns=['X', 'Y', 'Z'])
                # Compute the displacements using only the predictions
                p200 = displacementsFromPredictions200Ahead(pred_full, meas_full)  # first image is sagittal at +200
                p200 = pd.DataFrame(p200, columns=['X', 'Y', 'Z'])
                # Compute the displacements 400 sec ahead of time 
                p400 = displacementsFromPredictions400Ahead(pred_full, orientationFirstImage='coronal') # first image is coronal at +400
                p400 = pd.DataFrame(p400, columns=['X', 'Y', 'Z'])
                # Compute the displacements 600 sec ahead of time 
                p600 = displacementsFromPredictions600Ahead(pred_full) # first image is sagittal at +600
                p600 = pd.DataFrame(p600, columns=['X', 'Y', 'Z'])
                                    
                # Compute the errors (Euclidean distances between the 3D coordinates)
                #200-> GT+0
                dist_p000_gt = distance(p000, gt_full.iloc[trainingSize:].reset_index())
                avg_dist_p000_gt = np.mean(dist_p000_gt)
                dist_p200_gt = distance(p200, gt_full.iloc[trainingSize:].reset_index())
                avg_dist_p200_gt = np.mean(dist_p200_gt)
                dist_meas_gt = distance(meas_full.iloc[trainingSize:].reset_index(), gt_full[trainingSize:].reset_index())
                avg_dist_meas_gt = np.mean(dist_meas_gt)
                
                # 400-> GT+1, 600->GT+2
                dist_meas_gt_200ahead = distance(meas_full.iloc[trainingSize-1:-1].reset_index(), gt_full[trainingSize:].reset_index())
                avg_dist_meas_gt_200ahead = np.mean(dist_meas_gt_200ahead)

                dist_p400_gt = distance(p400, gt_full.iloc[trainingSize+1:].reset_index())
                avg_dist_p400_gt_400ahead = np.mean(dist_p400_gt)
                dist_meas_gt_400ahead = distance(meas_full.iloc[trainingSize-1:-2].reset_index(), gt_full[trainingSize+1:].reset_index())
                avg_dist_meas_gt_400ahead = np.mean(dist_meas_gt_400ahead)
    
                dist_p600_gt = distance(p600, gt_full.iloc[trainingSize+2:].reset_index())
                avg_dist_p600_gt_600ahead = np.mean(dist_p600_gt)
                dist_meas_gt_600ahead = distance(meas_full.iloc[trainingSize-1:-3].reset_index(), gt_full[trainingSize+2:].reset_index())
                avg_dist_meas_gt_600ahead = np.mean(dist_meas_gt_600ahead)                        

                # Computing average step size using GT
                dist_gt_gt_200ahead = distance(gt_full.iloc[trainingSize-1:-1].reset_index(), gt_full[trainingSize:].reset_index())
                avg_dist_gt_gt_200ahead = np.mean(dist_gt_gt_200ahead)
                
                dist_p200_meas = distance(p200, meas_full.iloc[trainingSize:].reset_index())
                avg_dist_p200_meas = np.mean(dist_p200_meas)
                dist_meas200_meas = distance(meas_full.iloc[trainingSize-1:-1].reset_index(), meas_full.iloc[trainingSize:].reset_index())
                avg_dist_meas200_meas = np.mean(dist_meas200_meas)

                # Computing distance peak-to-peak
                # Because displacements could be negative, make sure all of them are linearly shifted to positive values                                    
                gt_0 = pd.DataFrame({'X': gt_full['X']-min(gt_full['X'])+5,
                                     'Y': gt_full['Y']-min(gt_full['Y'])+5,
                                     'Z': gt_full['Z']-min(gt_full['Z'])+5})
                # Compute distance 
                disp_gt_peaks =  distance(gt_0,gt_0*0)
                # Compute the peak-to-peak difference between these magnitudes
                dist_peak_peak = max(disp_gt_peaks)-min(disp_gt_peaks)
                
                writer.writerow([
                    file.split('_')[1], 
                    round(avg_dist_p000_gt,3), round(avg_dist_meas_gt,3),
                    round(avg_dist_p200_gt,3), round(avg_dist_meas_gt_200ahead,3),
                    round(avg_dist_p400_gt_400ahead,3), round(avg_dist_meas_gt_400ahead,3),
                    round(avg_dist_p600_gt_600ahead,3), round(avg_dist_meas_gt_600ahead,3),
                    round(dist_peak_peak,3), round(avg_dist_gt_gt_200ahead,3)])
                
                all_avg.append([
                    round(avg_dist_p000_gt,3), round(avg_dist_meas_gt,3),
                    round(avg_dist_p200_gt,3), round(avg_dist_meas_gt_200ahead,3),
                    round(avg_dist_p400_gt_400ahead,3), round(avg_dist_meas_gt_400ahead,3),
                    round(avg_dist_p600_gt_600ahead,3), round(avg_dist_meas_gt_600ahead,3),
                    round(dist_peak_peak,3), round(avg_dist_gt_gt_200ahead,3)])
                
        writer.writerow(['averages']+list(np.mean(np.array(all_avg),axis=0)))
        
        writer.writerow(['filename', "p000_gt", 'meas_gt', 
                         "p200_gt", 'meas_gt_200ahead', 
                         'p400_gt_400ahead', 'meas_gt_400ahead', 
                         'p600_gt_600ahead','meas_gt_600ahead',
                         'dist_peak_peak','dist_gt_gt_200'])
