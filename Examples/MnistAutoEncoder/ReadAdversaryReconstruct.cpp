//
// Created by Hui Xie on 1/3/19.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include "MnistAutoEncoder.h"

void printUsage(char* argv0){
    cout<<"Read Mnist Adversary Sample and generate its reconstruction image:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir>  <adversaryFilesDir>"<<endl;
    cout<<"For example: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters /home/hxie1/temp_advData "<<endl;
}


int main(int argc, char *argv[]){
    printCurrentLocalTime();
    srand (time(NULL));
    if (3 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string netDir = argv[1];
    const string advDir = argv[2];

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    //Load MnistAutoEncoder Net
    MnistAutoEncoder net(netDir, nullptr);
    AdamOptimizer adamOptimizer(0.001,0.9,0.999);
    net.setOptimizer(&adamOptimizer);

    if (!isEmptyDir(net.getDir())) {
        net.load();  //at Jan  3th,2018, the trained G net has an accuracy of 97.43%
    }
    else{
        cout<<"Error: program can not load a trained Mnist net."<<endl;
        return -2;
    }
    net.printArchitecture();
    //net.setLearningRate(0.001);
    net.setUnlearningLayerID(200);  // 18 is the FC2 behind the Softmax of original G net.

    vector<string> advFileVector;
    getFileVector(advDir, advFileVector);
    const int N = advFileVector.size();

    Tensor<float>  inputImage({28,28});
    Tensor<float>  reconstrutImage;
    int  predictLabel;

    for (int i=0; i<N; ++i){
        string filename = advFileVector[i];
        string fullPathStem;
        string suffix;
        parseFullPathname(filename, fullPathStem, suffix);

        inputImage.load(filename, true);
        net.autoEncode(inputImage, predictLabel, reconstrutImage);

        string reconstructFilename = fullPathStem+ "-R"+ to_string(predictLabel) +suffix;
        reconstrutImage.save(reconstructFilename,true);


        //print specific layers for analysis
        const int layerID = 50;
        string layerOutputFilename = fullPathStem+ "-LayerID"+ to_string(layerID) +suffix;
        net.outputLayer(inputImage, layerID, layerOutputFilename);
    }
    cout << "input files and reconstruction files  at " <<advDir<<endl;
    cout<< "=========== finished AutoEncoder's image reconstruction from adversary samples ============"<<endl;
    return 0;
}

