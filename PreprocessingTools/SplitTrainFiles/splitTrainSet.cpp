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




    return 0;
}

