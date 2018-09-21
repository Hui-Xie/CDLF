//
// Created by Hui Xie on 9/21/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "DataManager.h"
#include "FileTools.h"


DataManager::DataManager(const string dataSetDir) {
    m_dataSetDir = dataSetDir;
    m_trainImagesDir = m_dataSetDir +"/trainImages";
    m_trainLabelsDir = m_dataSetDir +"/trainLabels";
    m_testImagesDir = m_dataSetDir +"/testImages";
    m_testLabelsDir = m_dataSetDir +"/testLabels";

    getFileVector(m_trainImagesDir, m_trainImagesVector);
    m_NTrainFile = m_trainImagesDir.size();
    cout<<"Info: totally read in "<<m_NTrainFile << "train images file. "<<endl;

    getFileVector(m_testImagesDir, m_testImagesVector);
    m_NTestFile = m_testImagesDir.size();
    cout<<"Info: totally read in "<<m_NTestFile << "test images file. "<<endl;

}

DataManager::~DataManager(){

}