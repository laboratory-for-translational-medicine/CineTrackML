# Copyright (C) Laboratory for Translation Medicine - All Rights Reserved

import SimpleITK as sitk
import numpy as np
import csv
import os
import Utils
import TrainingSetTo3DRef

def RegistrationCineTo3DRef(cine2DPathName, ref3DTref2DSag, ref3DTref2DCor, logPath = 'None'):
    trainingFiles = Utils.ListImages(cine2DPathName)
    # Initialize a few items
    prevOffset = np.array([[0.0], [0.0], [0.0]])
    cineTref3D = []
    
    if logPath != 'None':
        if os.path.exists(logPath):
            os.remove(logPath)
        csvFile = open(logPath, 'a+', newline='')
        write_outfile = csv.writer(csvFile)
        write_outfile.writerow(['Image', 'Registration Status', 'X', 'Y', 'Z']) 
    try:
        regIter = 0
        # Load the images one by one    
        for imgIterator in range(len(trainingFiles)):
            # Read image and compute its orientation
            moving = sitk.ReadImage(trainingFiles[imgIterator], sitk.sitkFloat32)
            moving, orientation = Utils.GetOrientation(moving)
            directionSlice = moving.GetDirection()
            moving2D = Utils.CopyCollapse(moving)
            if (orientation == "coronal") or (orientation == "sagittal"):
                if orientation == "coronal":
                    # If this is not the first time this orientation is met:
                    if ("coronalR" not in locals()):
                        coronalR = TrainingSetTo3DRef.coronalIntraReg
                    prevOffset = coronalR.RegisterStatic(moving2D, prevOffset) + ref3DTref2DCor
                    cineTref3D.append(prevOffset)
                elif orientation == "sagittal":
                    # If this is not the first time this orientation is met:
                    if ("sagittalR" not in locals()):
                        sagittalR = TrainingSetTo3DRef.sagittalIntraReg
                    prevOffset = sagittalR.RegisterStatic(moving2D, prevOffset) + ref3DTref2DSag
                    cineTref3D.append(prevOffset)                
                
                if logPath != 'None':
                    write_outfile.writerow([imgIterator, 'success', cineTref3D[regIter][0][0], cineTref3D[regIter][1][0], cineTref3D[regIter][2][0]])  
                regIter = regIter + 1
            elif logPath != 'None': #oblique and transverse images are ignored
                if regIter>0:
                    write_outfile.writerow([imgIterator, 'ignored', cineTref3D[regIter-1][0][0], cineTref3D[regIter-1][1][0], cineTref3D[regIter-1][2][0]])
                else:
                    write_outfile.writerow([imgIterator, 'ignored', 0, 0, 0])

    finally:
        if logPath != 'None':
            # Close files created
            csvFile.close()
    return cineTref3D