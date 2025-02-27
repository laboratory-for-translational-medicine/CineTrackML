# Copyright (C) Laboratory for Translation Medicine - All Rights Reserved

import SimpleITK as sitk
import numpy as np
import csv
import os
import Utils

def meanInterpolationZCorronal(z_intra_shift_cor: np.array, zFirstSag: float,  zLastSag: float, first: str) -> np.array:
    array = np.zeros(len(z_intra_shift_cor)*2)  
    index_cor_array = 1
    if(first == "Sagittal"):
        array[0] = zFirstSag 
        array[1] = z_intra_shift_cor[0]
        for i in range(3, len(z_intra_shift_cor)*2, 2):
            array[i-1] = (z_intra_shift_cor[index_cor_array] + z_intra_shift_cor[index_cor_array-1])/2
            array[i] = z_intra_shift_cor[index_cor_array]
            index_cor_array = index_cor_array + 1
        return array
    else:#not tested
        for i in range(0, len(z_intra_shift_cor)*2-2, 2):
            array[i] = z_intra_shift_cor[index_cor_array-1]
            array[i+1] = (z_intra_shift_cor[index_cor_array] + z_intra_shift_cor[index_cor_array-1])/2
            index_cor_array = index_cor_array + 1
        array[-1] = zLastSag
        return array
        
def meanInterpolationZSagittal(z_intra_shift_sag: np.array, zFirstCor: float, zLastCor: float, first: str) -> np.array:
    array = np.zeros(len(z_intra_shift_sag)*2)  
    index_sag_array = 1
    if(first == "Coronal"):
        array[0] = zFirstCor
        array[1] = z_intra_shift_sag[0]
        for i in range(3, len(z_intra_shift_sag)*2, 2):
            array[i-1] = (z_intra_shift_sag[index_sag_array] + z_intra_shift_sag[index_sag_array-1])/2
            array[i] = z_intra_shift_sag[index_sag_array]
            index_sag_array = index_sag_array + 1
        return array
    else:
        for i in range(0, len(z_intra_shift_sag)*2-2, 2):
            array[i] = z_intra_shift_sag[index_sag_array-1]
            array[i+1] = (z_intra_shift_sag[index_sag_array] + z_intra_shift_sag[index_sag_array-1])/2
            index_sag_array = index_sag_array + 1
        array[-2] = z_intra_shift_sag[-1]
        array[-1] = zLastCor
        return array

def combineSagCor(sag_array: np.array, cor_array: np.array, sag_split: float) -> np.array:
    
    # Gets the length of the arrays
    length = len(sag_array)
    
    # Create an empty array
    array = np.zeros(length)
    
    for i in range(0, length):
        array[i] = sag_array[i]*sag_split + cor_array[i]*(1-sag_split)
    
    return array

def SSECalculation(inter: np.array, intra: np.array) -> float:
    
    # Calculates the length of the array
    length = len(inter)

    # Initialize the SSE
    SSE = 0.0

    # Loops through all the values and calculates the distance between the actual value and the predicted value
    for i in range(0, (length-1)):
        SSE = SSE + (intra[i] - inter[i])**2
    
    return SSE
    
def Registration2DRefTo3DRef(ref3DPathName, maskPathName, training2DPathName, logPath):
    # Load 3D volume and mask
    fixed3D = sitk.ReadImage(ref3DPathName, sitk.sitkFloat32)
    mask3D = sitk.ReadImage(maskPathName, sitk.sitkUInt32)
    # Find all the moving images in the 'cine2DPathName' folder
    trainingFiles = Utils.ListImages(training2DPathName)
    
    # Initialize a few items
    prevInterOffset = np.array([[0.0], [0.0], [0.0]])
    prevIntraOffset = np.array([[0.0], [0.0], [0.0]])
    ref3DTref2DSag = []
    zRef3DTref2DSag = []
    ref3DTref2DCor = []
    zRef3DTref2DCor = []
    zRef3DTCineCor = []
    zRef3DTCineSag = []
    
    if os.path.exists(logPath):
        os.remove(logPath)
    csvFile = open(logPath, 'a+', newline='')
    
    try:
        write_outfile = csv.writer(csvFile)
        write_outfile.writerow(['Image', 'Registration Status', 
                                'X Inter', 'Y Inter', 'Z Inter',
                                'X Intra', 'Y Intra', 'Z Intra']) 
        
        # Load the images one by one    
        for imgIterator in range(len(trainingFiles)):
            # Read image and compute its orientation
            moving = sitk.ReadImage(trainingFiles[imgIterator], sitk.sitkFloat32)
            # Reorient image if needed
            moving, orientation = Utils.GetOrientation(moving)
            moving2D = Utils.CopyCollapse(moving)
            directionSlice = moving.GetDirection()
            if orientation == "coronal":
                # If this is the first time this orientation is met:
                if ("coronalInterReg" not in locals()):
                    coronalInterReg = Utils.Registration(orientation, directionSlice, mask3D, moving, fixed3D, "Inter")
                    global coronalIntraReg 
                    coronalIntraReg = Utils.Registration(orientation, directionSlice, mask3D, moving, trainingFiles, "Intra")               
                prevInterOffset = coronalInterReg.RegisterAdaptive(moving2D, prevInterOffset)
                prevIntraOffset = coronalIntraReg.RegisterStatic(moving2D, prevIntraOffset)
                ref3DTref2DCor.append(prevInterOffset-prevIntraOffset)
                zRef3DTref2DCor.append(prevIntraOffset[2][0])
                zRef3DTCineCor.append(prevInterOffset[2][0])
                
                write_outfile.writerow([imgIterator, 'success', 
                                        prevInterOffset[0][0], prevInterOffset[1][0], prevInterOffset[2][0],
                                        prevIntraOffset[0][0], prevIntraOffset[1][0], prevIntraOffset[2][0]])            
            elif orientation == "sagittal":
                # If this is the first time this orientation is met:
                if ("sagittalInterReg" not in locals()):
                    sagittalInterReg = Utils.Registration(orientation, directionSlice, mask3D, moving, fixed3D, "Inter")
                    global sagittalIntraReg 
                    sagittalIntraReg = Utils.Registration(orientation, directionSlice, mask3D, moving, trainingFiles, "Intra")              
                prevInterOffset = sagittalInterReg.RegisterAdaptive(moving2D, prevInterOffset)
                prevIntraOffset = sagittalIntraReg.RegisterStatic(moving2D, prevIntraOffset)
                ref3DTref2DSag.append(prevInterOffset-prevIntraOffset)
                zRef3DTref2DSag.append(prevIntraOffset[2][0])
                zRef3DTCineSag.append(prevInterOffset[2][0])
                
                write_outfile.writerow([imgIterator, 'success', 
                                        prevInterOffset[0][0], prevInterOffset[1][0], prevInterOffset[2][0],
                                        prevIntraOffset[0][0], prevIntraOffset[1][0], prevIntraOffset[2][0]]) 
            else: #oblique and transverse images are ignored
                write_outfile.writerow([imgIterator, 'ignored', 0, 0, 0, 0, 0, 0])
         
        if "sagittalInterReg" in locals():
            ref3DTref2DSag = np.concatenate(ref3DTref2DSag, axis =1) 
            ref3DTref2DSag = np.mean(ref3DTref2DSag[:,1:], 1)
            ref3DTref2DSag = np.array([[0], [ref3DTref2DSag[1]], [ref3DTref2DSag[2]]])
        else:
            ref3DTref2DSag = None
        if "coronalInterReg" in locals():
            ref3DTref2DCor = np.concatenate(ref3DTref2DCor, axis =1) 
            ref3DTref2DCor = np.mean(ref3DTref2DCor[:,1:], 1)
            ref3DTref2DCor = np.array([[ref3DTref2DCor[0]], [0], [ref3DTref2DCor[2]]])
        else:
            ref3DTref2DCor = None
            
        if ("sagittalInterReg" in locals()) and ("coronalInterReg" in locals()):
            z_intra_shift_sag = np.array(zRef3DTref2DSag) + ref3DTref2DSag[2][0] # Shift the intra curve by the shift amount
            z_intra_shift_cor = np.array(zRef3DTref2DCor) + ref3DTref2DCor[2][0] # Shift the intra curve by the shift amount
           
            SSE_Sagital = SSECalculation(zRef3DTCineSag, z_intra_shift_sag)
            SSE_Coronal = SSECalculation(zRef3DTCineCor, z_intra_shift_cor)
            total_SSE = SSE_Sagital + SSE_Coronal
            perc_sag = SSE_Sagital/total_SSE
            perc_cor = SSE_Coronal/total_SSE
            
            z_mean_interpolation_cor = meanInterpolationZCorronal(z_intra_shift_cor, z_intra_shift_sag[0], z_intra_shift_sag[-1], first = 'Sagittal')
            z_mean_interpolation_sag = meanInterpolationZSagittal(z_intra_shift_sag, z_intra_shift_cor[0], z_intra_shift_cor[-1], first = 'Sagittal')
            # Combination of the Sagittal and Coronal using a % split
            z_combined_mean = combineSagCor(z_mean_interpolation_sag, z_mean_interpolation_cor, (1-perc_sag))
            distanceSag = np.mean(z_combined_mean - z_mean_interpolation_sag)
            distanceCor = np.mean(z_combined_mean - z_mean_interpolation_cor)
            ref3DTref2DSag[2][0] += distanceSag
            ref3DTref2DCor[2][0] += distanceCor
            
    # Close files created
    finally:
        csvFile.close()
    
    return [ref3DTref2DSag, ref3DTref2DCor]