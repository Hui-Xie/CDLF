//
// Created by Hui Xie on 9/20/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <iostream>
#include "FileTools.h"


void printUsage(char* argv0){
    cout<<"============= Split Train Files into Tain and Test Set ==========="<<endl;
    cout<<"As our Test files in hands has no label, we need split some train files with labels into test set."<<endl;
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
        const string srcImage = fileVector[i];
        const string filename = getFileName(srcImage);
        const string srcLabel = trainLabelsDir +"/"+ filename;
        string dstImage = "";
        string dstLabel = "";
        if (0 !=i && 0 == i%step){ //copy to test dir
            dstImage = outputTestImagesDir +"/" + filename;
            dstLabel = outputTestLabelsDir +"/" + filename;
            ++countTest;
        }
        else{  //copy to train dir
            dstImage = outputTrainImagesDir +"/" + filename;
            dstLabel = outputTrainLabelsDir +"/" + filename;
            ++countTrain;
        }
        copyFile(srcImage, dstImage);
        copyFile(srcLabel, dstLabel);

    }
    cout<<"Infor: output "<<countTrain << " image files to train image directory: "<<outputTrainImagesDir<<endl;
    cout<<"Infor: output "<<countTrain << " label files to train label directory: "<<outputTrainLabelsDir<<endl;
    cout<<"Infor: output "<<countTest  << " image files to test image directory: "<<outputTestImagesDir<<endl;
    cout<<"Infor: output "<<countTest  << " label files to test label directory: "<<outputTestLabelsDir<<endl;

    return 0;
}

