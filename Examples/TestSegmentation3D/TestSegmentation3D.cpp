//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "TestSegmentation3D.h"
#include <iostream>
using namespace std;

void printUsage(char* argv0){
    cout<<"Test 3D Medical Image Segmentation: "<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" fullPathOfImagesDir"<<endl;
}

int main(int argc, char *argv[]) {

    if (2 != argc) {
        cout << "Error: input parameter error." << endl;
        printUsage(argv[0]);
        return -1;
    }


}