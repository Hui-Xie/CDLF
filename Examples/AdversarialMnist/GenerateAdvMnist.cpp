//
// Created by Hui Xie on 11/9/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include "MNIST.h"
#include "AdverMnistNet.h"
#include <cstdio>

void printUsage(char* argv0){
    cout<<"Generate Advesarial Samples for MNIST Dataset:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfMnistDataDir>  <fullPathAdvData>"<<endl;
    cout<<"for examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters/MnistNet /home/hxie1/Projects/mnist  /home/hxie1/temp_advData"<<endl;
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

    bool bGenerateGradientFiles = false;

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
    AdverMnistNet net(netDir);
    AdamOptimizer adamOptimizer(0.001,0.9,0.999);
    net.setOptimizer(&adamOptimizer);

    if (!isEmptyDir(net.getDir())) {
        net.load();
    }
    else{
        cout<<"Error: program can not load trained Mnist net."<<endl;
        return -2;
    }
    net.printArchitecture();
    //net.setLearningRate(10000);
    net.setLambda(0.000001);// lambda* lr ==1 means erase changes differing with origin Tensor.

    //create Adversarial data directory
    //advDataDir +="/"+ getCurTimeStr();  //todo: debug stage, No need it now.
    createDir(advDataDir);

    // choose an initial digit file and make sure it is predicted correctly
    srand (time(NULL));
    bool bPredictCorrect = false;
    Tensor<float> inputTensor;
    Tensor<float> groundTruth;
    int label; // correct label, which is different with target.
    while (!bPredictCorrect){
        int index = rand() % 10000;
        mnist.getTestImageAndLabel(index, inputTensor, label);
        net.constructGroundTruth(label, groundTruth);
        net.m_groundTruth = groundTruth;
        if (net.predict(inputTensor) == label){
            bPredictCorrect = true;
        }
    }
    net.m_originTensor = inputTensor;

    string originFile = to_string(label)+".txt";
    originFile = advDataDir +"/"+ originFile;
    inputTensor.save(originFile, true);
    printf("Info: output original file: %s with label of %d. \n", originFile.c_str(), label);

    vector<int> targetVec= {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for(vector<int>::iterator it= targetVec.begin(); it != targetVec.end(); ++it){
        if (label == *it){
            targetVec.erase(it);
            break;
        }
    }

    for(int i=0; i< targetVec.size(); ++i){
        const int target = targetVec[i];
        bool bFoundAdversary = false;
        net.m_adversaryTensor = net.m_originTensor; // re-assign original input Tensor
        net.constructGroundTruth(target, groundTruth);
        net.m_groundTruth = groundTruth;  //targeted groundtruth, not the correct label.
        int nCount = 0;
        int MaxCount = 1000;
        while(nCount < MaxCount){
            net.train();
            // generate gardient files for specific target
            if (bGenerateGradientFiles){
                string gradientFile = to_string(label)+"-Ad"+ to_string(target)+"-G"+to_string(nCount)+".txt";
                gradientFile = advDataDir +"/" + gradientFile;
                net.saveInputDY(gradientFile);
            }

            ++nCount;
            if (net.predict(net.m_adversaryTensor ) == target){
               bFoundAdversary = true;
               break;
            }
        }

        if (bFoundAdversary){
            string advFile = to_string(label)+"-Ad"+ to_string(target)+".txt";
            advFile = advDataDir +"/"+ advFile;
            net.m_adversaryTensor.save(advFile, true);
            printf("After %d back propagation iterations, an adversarial file output at: %s\n", nCount, advFile.c_str());
        }
        else{
            printf("Infor: program can not find adversary for target %d, basing on original digit: %d\n", target, label);
        }

    }

    cout<< "=========== End of Generating Adversary by a trained network"<<net.getName() <<" ============"<<endl;
    return 0;
}