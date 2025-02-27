# Copyright (C) Laboratory for Translation Medicine - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import SimpleITK as sitk
import numpy as np
from numpy.linalg import inv
import csv
import glob
import os
import sys

def get_anatomical_orientation(image):
    """
    Determines the anatomical orientation of an image based on its direction cosines.
    """
    direction = image.GetDirection()
    anatomical_labels = ['R', 'A', 'I', 'L', 'P', 'S']
    orientation = []

    # Determine the dominant direction for each axis
    for axis in range(3):
        # Extract the direction vector for the current axis
        vector = direction[axis::3]
        # Determine the index of the dominant direction
        max_index = max(range(3), key=lambda i: abs(vector[i]))
        # Append the corresponding anatomical label
        orientation.append(anatomical_labels[max_index + (3 if vector[max_index] < 0 else 0)])

    return ''.join(orientation)

def RetrieveOrientation(direct):
    """
    Finds the orientation of each image
    """
    orientation = "oblique"
    if (round(abs(direct[2])) == 1):
        orientation = "sagittal"
    elif (round(abs(direct[5])) == 1):
        orientation = "coronal"
    elif (round(abs(direct[8])) == 1):
        orientation = "transverse"
    return orientation

def GetOrientation(image):
    """
    Returns the orientation of each image and reorients it if needed.
    """
    if image.GetSize()[0] == 1:
        orientation = get_anatomical_orientation(image)
        # Single-slice in x-direction, check for sagittal, coronal, or axial
        if orientation[0] in ('L', 'R'):
            return sitk.DICOMOrient(image, 'PIR'), "sagittal"
        elif orientation[0] in ('A', 'P'):
            return sitk.DICOMOrient(image, 'LIA'), "coronal"
        else:
            return sitk.DICOMOrient(image, 'LPS'), "transverse"
    elif image.GetSize()[1] == 1:
        orientation = get_anatomical_orientation(image)
        # Single-slice in y-direction, check for sagittal, coronal, or axial
        if orientation[1] in ('L', 'R'):
            return sitk.DICOMOrient(image, 'PIR'), "sagittal"
        elif orientation[1] in ('A', 'P'):
            return sitk.DICOMOrient(image, 'LIA'), "coronal"
        else:
            return sitk.DICOMOrient(image, 'LPS'), "transverse"
    return image, RetrieveOrientation(image.GetDirection())

def CopyCollapse(sliceIn3D):
    size = list(sliceIn3D.GetSize())
    size[2] = 0
    index = [0, 0, 0]
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size)
    extractor.SetIndex(index)
    extractor.SetDirectionCollapseToStrategy(extractor.DIRECTIONCOLLAPSETOIDENTITY)
    image2D = extractor.Execute(sliceIn3D)
    return image2D

def Extract2DFrom3D(image3D, target2D):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target2D) 
    fixed2D = CopyCollapse(resampler.Execute(image3D))
    return fixed2D

def Dilate2DMask(mask2D):
    statistics_filter = sitk.StatisticsImageFilter()
    statistics_filter.Execute(mask2D)
    max_intensity = statistics_filter.GetMaximum()
    filter = sitk.BinaryDilateImageFilter()
    filter.SetKernelType(sitk.sitkBall)
    filter.SetKernelRadius(15)
    filter.SetForegroundValue(max_intensity)
    filter.SetBackgroundValue(0)
    return filter.Execute(mask2D)
                
def ListImages(imagePathName):
    """
    Finds all the moving images in the 'movingFolder' folder.
    Assumes images have extension .dcm or .mha. If a different extension is used, then add the extension in list 'types'.
    """
    types = ["*.dcm", "*.mha"]
    imageFiles = []
    for type in types:
        this_type_files = glob.glob(imagePathName + '\\' + type, recursive = True)
        imageFiles += this_type_files
    if len(imageFiles) == 0:
        print(imagePathName + " is empty")
        sys.exit(1)
    return imageFiles

class Registration:
    def __init__(self, orientation, directionSlice, mask3D, moving3D, fixedFiles, keyword): 
        self.moving3D = moving3D
        self.fixedFiles = fixedFiles
        self.mask3D = mask3D
        self.mask2DNoDilation =   Extract2DFrom3D(mask3D, moving3D)
        self.mask2D     =   Dilate2DMask(self.mask2DNoDilation)   
        self.directionSlice = directionSlice
        self.R = sitk.ImageRegistrationMethod()
        self.R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 100, 0.9)
        self.keyword = keyword
        if keyword == "Intra":
            self.fixed  =   self.Get2DFrom2DRef(directionSlice, fixedFiles)
            self.R.SetMetricAsMeanSquares()
        elif keyword == "Inter":
            self.fixed  =    Extract2DFrom3D(fixedFiles, moving3D)
            self.R.SetMetricAsMattesMutualInformation(32)
        self.R.SetOptimizerScalesFromPhysicalShift() 
        self.outTxInSliceCoord = sitk.TranslationTransform(self.mask2D.GetDimension())
        self.R.SetInitialTransform(self.outTxInSliceCoord, inPlace=True)
        self.R.SetInterpolator(sitk.sitkLinear)
        self.R.SetMetricFixedMask(self.mask2D)

    def Get2DFrom2DRef(self, directionSlice, fixedFiles):
        refImgIterator = 0#len(fixedFiles)
        found = False
        while (refImgIterator != len(fixedFiles) and found == False):
            
            ref2D2D = sitk.ReadImage(fixedFiles[refImgIterator], sitk.sitkFloat32)
            ref2D2D, orientation = GetOrientation(ref2D2D)
            refImgIterator = refImgIterator+1
            direction2D2D = ref2D2D.GetDirection()
            if direction2D2D == directionSlice:
                found = True
                return CopyCollapse(ref2D2D)

    def GuessBestInitialT(self, moving, outTxInSliceCoord):
        reg = self.R
        reg.SetInitialTransform(outTxInSliceCoord, inPlace=True) 
        reg.SetOptimizerAsRegularStepGradientDescent(0.001, .001, 1, 0.9)
        initialGuessMetric = reg.MetricEvaluate(self.fixed, moving)
        reg.SetInitialTransform(sitk.TranslationTransform(moving.GetDimension()), inPlace=True) 
        if reg.MetricEvaluate(self.fixed, moving) < initialGuessMetric:
            return sitk.TranslationTransform(moving.GetDimension()) 
        else:
            return outTxInSliceCoord
        
    def RegisterStatic(self, moving, prevOffset):
        prevOffsetInSliceCoord = inv(np.reshape(self.directionSlice, (3,3))) @ prevOffset
        self.outTxInSliceCoord.SetParameters((prevOffsetInSliceCoord[0,0], prevOffsetInSliceCoord[1,0]))        
        bestInitialGuess = self.GuessBestInitialT(moving, self.outTxInSliceCoord)
        self.R.SetInitialTransform(bestInitialGuess, inPlace=True)
        self.R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 100, 0.9) #Shallow copy in GuessBestInitialT
        self.outTxInSliceCoord = self.R.Execute(self.fixed, moving)
        
        # Convert to reference system
        inPlaneOffsetInSliceCoord = np.array([[self.outTxInSliceCoord.GetParameters()[0]], 
                                     [self.outTxInSliceCoord.GetParameters()[1]],
                                     [prevOffsetInSliceCoord[2,0]]])
        
        prevOffset = np.reshape(self.directionSlice, (3,3)) @ inPlaneOffsetInSliceCoord
        return prevOffset

    def RegisterAdaptive(self, moving, prevOffset):
        prevOffsetInSliceCoord = inv(np.reshape(self.directionSlice, (3,3))) @ prevOffset
        self.outTxInSliceCoord.SetParameters((prevOffsetInSliceCoord[0,0], prevOffsetInSliceCoord[1,0]))
        
        # Calculate previous thru-plane offsets for adaptive slicing of mask and 3D volume
        prevThruPlaneOffsetInSliceCoord = prevOffsetInSliceCoord
        prevThruPlaneOffsetInSliceCoord[0] = prevThruPlaneOffsetInSliceCoord[1] = 0
        prevThruPlaneOffset = np.reshape(self.directionSlice, (3,3)) @ prevThruPlaneOffsetInSliceCoord
        prevThruPlaneTransform = sitk.TranslationTransform(3)
        prevThruPlaneTransform.SetParameters((prevThruPlaneOffset[0,0], prevThruPlaneOffset[1,0], prevThruPlaneOffset[2,0]))
        # Extract adaptive slice of mask and 3D volume
        self.mask2DNoDilation    =    sitk.Resample(self.mask3D, self.moving3D, prevThruPlaneTransform.GetInverse())
        self.mask2DNoDilation    =    CopyCollapse(self.mask2DNoDilation)
        self.mask2D              =    Dilate2DMask(self.mask2DNoDilation)   
        self.fixed               =    sitk.Resample(self.fixedFiles, self.moving3D, prevThruPlaneTransform.GetInverse())
        self.fixed               =    CopyCollapse(self.fixed)   
        
        bestInitialGuess = self.GuessBestInitialT(moving, self.outTxInSliceCoord)
        self.R.SetInitialTransform(bestInitialGuess, inPlace=True)
        self.R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 100, 0.9) #Shallow copy in GuessBestInitialT
        self.outTxInSliceCoord = self.R.Execute(self.fixed, moving)
        
        # Convert to reference system
        inPlaneOffsetInSliceCoord = np.array([[self.outTxInSliceCoord.GetParameters()[0]], 
                                     [self.outTxInSliceCoord.GetParameters()[1]],
                                     [prevOffsetInSliceCoord[2,0]]])
        
        prevOffset = np.reshape(self.directionSlice, (3,3)) @ inPlaneOffsetInSliceCoord
        return prevOffset
    
def Registration2DRefTo3DRef(ref3DPathName, maskPathName, training2DPathName, logPath):
    # Load 3D volume and mask
    fixed3D = sitk.ReadImage(ref3DPathName, sitk.sitkFloat32)
    mask3D = sitk.ReadImage(maskPathName, sitk.sitkUInt32)
    # Find all the moving images in the 'cine2DPathName' folder
    trainingFiles = ListImages(training2DPathName)
    
    # Initialize a few items
    prevInterOffset = np.array([[0.0], [0.0], [0.0]])
    prevIntraOffset = np.array([[0.0], [0.0], [0.0]])
    ref3DTref2D = []
    if os.path.exists(logPath):
        os.remove(logPath)
    csvFile = open(logPath, 'a+', newline='')
    write_outfile = csv.writer(csvFile)
    write_outfile.writerow(['Image', 'Registration Status', 
                            'X Inter', 'Y Inter', 'Z Inter',
                            'X Intra', 'Y Intra', 'Z Intra']) 
    
    # Load the images one by one    
    for imgIterator in range(len(trainingFiles)):
        print("image:", imgIterator)
        # Read image and compute its orientation
        moving = sitk.ReadImage(trainingFiles[imgIterator], sitk.sitkFloat32)
        directionSlice = moving.GetDirection()
        orientation = RetrieveOrientation(directionSlice)
        moving2D = CopyCollapse(moving)
        if orientation == "coronal":
            # If this is not the first time this orientation is met:
            if ("coronalInterReg" not in locals()):
                coronalInterReg = Registration(orientation, directionSlice, mask3D, moving, fixed3D, "Inter")
                coronalIntraReg = Registration(orientation, directionSlice, mask3D, moving, trainingFiles, "Intra")               
            prevInterOffset = coronalInterReg.RegisterAdaptive(moving2D, prevInterOffset)
            prevIntraOffset = coronalIntraReg.RegisterStatic(moving2D, prevIntraOffset)
            ref3DTref2D.append(prevInterOffset-prevIntraOffset)
            write_outfile.writerow([imgIterator, 'success', 
                                    prevInterOffset[0][0], prevInterOffset[1][0], prevInterOffset[2][0],
                                    prevIntraOffset[0][0], prevIntraOffset[1][0], prevIntraOffset[2][0]])            
        elif orientation == "sagittal":
            # If this is not the first time this orientation is met:
            if ("sagittalInterReg" not in locals()):
                sagittalInterReg = Registration(orientation, directionSlice, mask3D, moving, fixed3D, "Inter")
                sagittalIntraReg = Registration(orientation, directionSlice, mask3D, moving, trainingFiles, "Intra")              
            prevInterOffset = sagittalInterReg.RegisterAdaptive(moving2D, prevInterOffset)
            prevIntraOffset = sagittalIntraReg.RegisterStatic(moving2D, prevIntraOffset)
            ref3DTref2D.append(prevInterOffset-prevIntraOffset)
            write_outfile.writerow([imgIterator, 'success', 
                                    prevInterOffset[0][0], prevInterOffset[1][0], prevInterOffset[2][0],
                                    prevIntraOffset[0][0], prevIntraOffset[1][0], prevIntraOffset[2][0]]) 
        else: #oblique and transverse images are ignored
            write_outfile.writerow([imgIterator, 'ignored', 0, 0, 0, 0, 0, 0])
            
    ref3DTref2D = np.concatenate(ref3DTref2D, axis =1) 
    ref3DTref2D = np.mean(ref3DTref2D, 1)
    
    # Close files created
    csvFile.close()
    return ref3DTref2D