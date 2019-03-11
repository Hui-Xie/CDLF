//
// Created by Hui Xie on 12/11/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "MnistAutoEncoder.h"

void printUsage(char* argv0){
    cout<<"Test MNIST Dataset AutoEncoder:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfMnistDataDir> <outputImageDir> <nImages> <0|1>"<<endl;
    cout<<"For example: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters/MnistAutoEncoder /home/hxie1/Projects/mnist /home/hxie1/temp_DecoderOutput 40  0"<<endl;
    cout<<"<0|1> 0: indicate general reconstruction with correct label prediction; 1: indicate only using incorrectly recognized image for reconstruction"<<endl;
}


int main(int argc, char *argv[]){
    printCurrentLocalTime();
    srand (time(NULL));
    if (6 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string netDir = argv[1];
    const string mnistDir = argv[2];
    const string outputDir = argv[3];
    const int nImages = atoi(argv[4]);
    const bool usingCorrectPrediction = (0 == atoi(argv[5]))? true: false;


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
    MnistAutoEncoder net(netDir, &mnist);
    if (!isEmptyDir(net.getDir())) {
        net.load();  //at Jan  3th,2018, the trained G net has an accuracy of 97.43%
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

    cout << "Hint: output file format: \n Tindex-IrealLabel.txt, Tindex-RpredictLabel.txt, e.g. T32-R4.txt, T32-P7.txt"<<endl;
    for (int i=0; i<nImages; ++i){
        int index = rand() % 10000;
        mnist.getTestImageAndLabel(index, inputImage, label);
        net.autoEncode(inputImage, predictLabel, reconstrutImage);
        bool predictCorrectly = (label == predictLabel)? true: false;
        const string leadingCapital =    predictCorrectly? "T": "F";

        if (!usingCorrectPrediction && predictCorrectly){
            --i;
            continue;
        }

        string filename = outputDir +"/" + leadingCapital + to_string(i)+ "-I"+ to_string(label) +".txt";
        inputImage.save(filename, true);
        filename = outputDir +"/" + leadingCapital + to_string(i)+ "-R"+ to_string(predictLabel) +".txt";
        reconstrutImage.save(filename,true);

    }
    cout << "input files and reconstruction files output at " <<outputDir<<endl;
    cout<< "=========== finished AutoEncoder's image reconstruction ============"<<endl;
    return 0;
}