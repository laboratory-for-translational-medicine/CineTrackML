# Copyright (C) Laboratory for Translation Medicine - All Rights Reserved

import TrainingSetTo3DRef
import CineTo3DRef
import os

argumentList = [   
    [[r'..\LiverImages\Liver1\Volume3D.mha'], 
      [r'..\LiverImages\Liver1\StructureLiver.mha'], 
      [r"..\LiverImages\Liver1\Liver1Training"], 
      [r"..\LiverImages\Liver1\Liver1Cine"]],
    
    [[r'..\LiverImages\Liver2\Volume3D.mha'], 
      [r'..\LiverImages\Liver2\StructureLiver.mha'], 
      [r"..\LiverImages\Liver2\Liver2Training"], 
      [r"..\LiverImages\Liver2\Liver2Cine"]],
    
    [[r'..\LiverImages\Liver3\Volume3D.mha'], 
      [r'..\LiverImages\Liver3\StructureLiver.mha'], 
      [r"..\LiverImages\Liver3\Liver3Training"], 
      [r"..\LiverImages\Liver3\Liver3Cine"]],
    
    [[r'..\LiverImages\Liver4\Volume3D.mha'], 
      [r'..\LiverImages\Liver4\StructureLiver.mha'], 
      [r"..\LiverImages\Liver4\Liver4Training"], 
      [r"..\LiverImages\Liver4\Liver4Cine"]],    
    
    [[r'..\LiverImages\Liver5\Volume3D.mha'], 
      [r'..\LiverImages\Liver5\StructureLiver.mha'], 
      [r'..\LiverImages\Liver5\Liver5Training'], 
      [r'..\LiverImages\Liver5\Liver5Cine']],    
    
    [[r'..\LiverImages\Liver6\Volume3D.mha'], 
      [r'..\LiverImages\Liver6\StructureLiver.mha'], 
      [r"..\LiverImages\Liver6\Liver6Training"], 
      [r"..\LiverImages\Liver6\Liver6Cine"]],    
    
    [[r'..\LiverImages\Liver7\Volume3D.mha'], 
      [r'..\LiverImages\Liver7\StructureLiver.mha'], 
      [r"..\LiverImages\Liver7\Liver7Training"], 
      [r"..\LiverImages\Liver7\Liver7Cine"]],    
    
    [[r'..\LiverImages\Liver8\Volume3D.mha'], 
      [r'..\LiverImages\Liver8\StructureLiver.mha'], 
      [r"..\LiverImages\Liver8\Liver8Training"], 
      [r"..\LiverImages\Liver8\Liver8Cine"]], 
    
    [[r'..\LiverImages\Liver9\Volume3D.mha'], 
      [r'..\LiverImages\Liver9\StructureLiver.mha'], 
      [r"..\LiverImages\Liver9\Liver9Training"], 
      [r"..\LiverImages\Liver9\Liver9Cine"]], 
    ]

for i in range(len(argumentList)):
    ref3DPathName = argumentList[i][0][0] 
    maskPathName = argumentList[i][1][0] 
    training2DPathName = argumentList[i][2][0] 
    cine2DPathName = argumentList[i][3][0] 

    print('Processing: ', os.path.basename(os.path.dirname(ref3DPathName)).split('-')[0] )
    ### Log results ###
    log2DRefTo3DrefPath = r"..\ProcessingResults\2DRefTo3Dref_" + argumentList[i][0][0].split('\\')[2] + '.csv'
    logCineTo3DrefPath =  r'..\ProcessingResults\CineTo3Dref_' + argumentList[i][0][0].split('\\')[2] + '_Measurement.csv'

    ### Create 2D reference images from training set and register to 3D reference images ###
    [ref3DTref2DSag, ref3DTref2DCor] = TrainingSetTo3DRef.Registration2DRefTo3DRef(ref3DPathName, maskPathName, training2DPathName, log2DRefTo3DrefPath)

    ### Compute displacement using incoming cine images from 3D reference position ###
    CineTo3DRef.RegistrationCineTo3DRef(cine2DPathName, ref3DTref2DSag, ref3DTref2DCor, logCineTo3DrefPath)