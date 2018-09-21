//
// Created by Hui Xie on 9/20/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <iostream>
#include "FileTools.h"


void printUsage(char* argv0){
    cout<<"============= Split Train Files into Tain and Test Set ==========="<<endl;
    cout<<"As our Test files in hands has no label, we need split some train files with labels into test files"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<trainImagesDir> <trainLabelsDir> <outputDir> "<<endl;
    cout<<endl;
}



int main(int argc, char *argv[]) {

    printUsage(argv[0]);
    if (argc != 4) {
        cout << "Error: the number of parameters is incorrect." << endl;
        return -1;
    }

    const string trainImagesDir = argv[1];
    const string trainLabelsDir = argv[2];
    const string outputDir      = argv[3];

    vector<string> fileVector;
    getFileVector(trainImagesDir, fileVector);
    int N = fileVector.size();
    cout<<"Info: total "<<N<<"  images files in "<< trainImagesDir<<endl;
    int testN = int(N*0.2);  // 20% files as test data set
    int step = N/testN;


    const string outputTrainImagesDir = outputDir +"/trainImages";
    const string outputTrainLabelsDir = outputDir +"/trainLabels";
    const string outputTestImagesDir = outputDir +"/testImages";
    const string outputTestLabelsDir = outputDir +"/testLabels";
    if (!dirExist(outputTrainImagesDir)){
        mkdir(outputTrainImagesDir.c_str(),S_IRWXU |S_IRWXG | S_IROTH |S_IXOTH);
    }
    if (!dirExist(outputTrainLabelsDir)){
        mkdir(outputTrainLabelsDir.c_str(),S_IRWXU |S_IRWXG | S_IROTH |S_IXOTH);
    }
    if (!dirExist(outputTestImagesDir)){
        mkdir(outputTestImagesDir.c_str(),S_IRWXU |S_IRWXG | S_IROTH |S_IXOTH);
    }
    if (!dirExist(outputTestLabelsDir)){
        mkdir(outputTestLabelsDir.c_str(),S_IRWXU |S_IRWXG | S_IROTH |S_IXOTH);
    }

    int countTrain = 0;
    int countTest  = 0;
    for (int i=0; i< N ; ++i){
        if (0 !=i && 0 == i%step){ //copy to test dir


            ++countTest;
        }
        else{  //copy to train dir


            ++countTrain;
        }

    }
    cout<<"Infor: output "<<countTrain << " files to train directory. "<<endl;
    cout<<"Infor: output "<<countTest <<  " files to test directory. "<<endl;
    
    return 0;
}

