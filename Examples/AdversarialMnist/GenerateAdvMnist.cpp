//
// Created by Hui Xie on 11/9/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "MNIST.h"
//#include "MnistConvNet.h"
#include "AdverMnistNet.h"
#include <cstdio>

void printUsage(char* argv0){
    cout<<"Generate Advesarial Samples for MNIST Dataset:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfMnistDataDir>  <fullPathAdvData>"<<endl;
    cout<<"for examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters /home/hxie1/Projects/mnist  /home/hxie1/temp_advData"<<endl;
}


int main(int argc, char *argv[]){
    printCurrentLocalTime();
    if (4 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string netDir = argv[1];
    const string mnistDir = argv[2];
    string advDataDir = argv[3];

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    // Load MNIST Data
    bool onlyTestSet = true;
    MNIST mnist(mnistDir, onlyTestSet);
    mnist.loadData();

    //Load Mnist Net
    AdverMnistNet net("MnistNet", netDir);
    if (!isEmptyDir(net.getDir())) {
        net.load();
    }
    else{
        cout<<"Error: program can not load trained Mnist net."<<endl;
        return -2;
    }
    net.printArchitecture();

    //create Adversarial data directory
    advDataDir +="/"+ getCurTimeStr();
    createDir(advDataDir);

    // choose an initial digit file and make sure it is predicted correctly
    srand (time(NULL));
    bool bPridictCorrect = false;
    Tensor<float> inputTensor;
    Tensor<float> groundTruth;
    int correctLabel;
    while (! bPridictCorrect){
        long index = rand() % 10000;
        mnist.getTestImageAndLabel(index, inputTensor, correctLabel);
        net.m_inputTensor = inputTensor;
        net.constructGroundTruth(correctLabel, groundTruth);
        net.m_groundTruth = groundTruth;
        if (net.predict() == correctLabel){
            bPridictCorrect = true;
        }
    }
    string originFile = to_string(correctLabel)+".txt";
    originFile = advDataDir +"/"+ originFile;
    inputTensor.save(originFile, true);
    printf("Info: output original file: %s\n", originFile.c_str());

    vector<int> targetVec= {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for(vector<int>::iterator it= targetVec.begin(); it != targetVec.end(); ++it){
        if (correctLabel == *it){
            targetVec.erase(it);
            break;
        }
    }

    for(int i=0; i< targetVec.size(); ++i){
        bool bFoundAdvesary = false;
        net.m_inputTensor = inputTensor;
        net.constructGroundTruth(targetVec[i], groundTruth);
        net.m_groundTruth = groundTruth;
        int nIter = 0;
        while(!bFoundAdvesary){
            net.train();
            nIter++;
            if (net.predict() == targetVec[i]){
                bFoundAdvesary = true;
            }
        }
        string advFile = to_string(correctLabel)+"-Ad"+ to_string(targetVec[i])+".txt";
        advFile = advDataDir +"/"+ advFile;
        net.m_inputTensor.save(originFile, true);
        printf("After %d back propagation iterations, we find adversarial file: %s\n", nIter, advFile.c_str());
    }

    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    return 0;
}