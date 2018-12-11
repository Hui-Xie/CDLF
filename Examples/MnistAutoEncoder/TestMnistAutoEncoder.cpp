//
// Created by Hui Xie on 12/11/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "MnistAutoEncoder.h"

void printUsage(char* argv0){
    cout<<"Test MNIST Dataset AutoEncoder:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfMnistDataDir> <outputImageDir> <nImages>"<<endl;
    cout<<"For examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters /home/hxie1/Projects/mnist /home/hxie1/temp_DecoderOutput 400"<<endl;
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
    const string outputDir = argv[3];
    const int nImages = atoi(argv[4]);

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

    //Load MnistAutoEncoder Net
    MnistAutoEncoder net("MnistAutoEncoder", netDir, &mnist);
    if (!isEmptyDir(net.getDir())) {
        net.load();  //at Dec 11th,2018, the trained G net has an accuracy of 97.11%
    }
    else{
        cout<<"Error: program can not load a trained Mnist net."<<endl;
        return -2;
    }
    net.printArchitecture();
    net.setLearningRate(0.001);
    net.setUnlearningLayerID(200);  // 18 is the FC2 behind the Softmax of original G net.

    Tensor<float>  inputImage;
    Tensor<float>  reconstrutImage;
    int  label;
    int  predictLabel;
    cout << "Hint: output file format: \n Tindex-RrealLabel.txt, Tindex-PpredictLabel.txt, e.g. T32-R4.txt, T32-P7.txt"<<endl;
    for (int i=0; i<nImages; ++i){
        long index = rand() % 10000;
        mnist.getTestImageAndLabel(index, inputImage, label);
        net.autoEncode(inputImage, predictLabel, reconstrutImage);
        string filename = outputDir +"/" + "T" + to_string(i)+ "-R"+ to_string(label) +".txt";
        inputImage.save(filename, true);
        filename = outputDir +"/" + "T" + to_string(i)+ "-P"+ to_string(predictLabel) +".txt";
        reconstrutImage.save(filename,true);
    }
    cout<< "=========== finished AutoEncoder's image reconstruction ============"<<endl;
    return 0;
}