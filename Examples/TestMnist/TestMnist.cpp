//
// Created by Hui Xie on 8/6/2018.
//

#include "MnistConvNet.h"
#include "MNIST.h"

void printUsage(char* argv0){
    cout<<"Test 10 digits 0-9 Simultaneously in MNIST Dataset:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfMnistDataDir> <learningRate>"<<endl;
    cout<<"for examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters/MnistNet /home/hxie1/Projects/mnist  0.0001"<<endl;
    cout<<argv0<<" /Users/hxie1/temp_netParameters/MnistNet /Users/hxie1/Projects/mnist  0.0001"<<endl;
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
    const float learningRate = stof(argv[3]);


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
    MNIST mnist(mnistDir);
    mnist.loadData();

    cout<<"test the matching between image and label in whole dataset"<<endl;
    srand (time(NULL));
    int index = rand() % 10000;
    mnist.displayImage(mnist.m_pTestImages, index);
    cout<<"Image is "<<(int)(mnist.m_pTestLabels->e(index))<<endl;

    //tailor data and delete wholeDataSet, and keep PartDataSet
    //mnist.tailorData();
    //mnist.deleteWholeDataSet();

    //cout<<"test the matching between image and label in part dataset"<<endl;
    //index = rand() % 10000;
    //mnist.displayImage(mnist.m_pTrainImagesPart, index);
    //cout<<"Image is "<<(int)(mnist.m_pTrainLabelsPart->e(index))<<endl;

    // Construct FeedForwardNet and Train, Test
    MnistConvNet net(netDir, &mnist);
    if (isEmptyDir(net.getDir())) {
        net.build();
        net.initialize();
        net.setLearningRate(0.001);
        net.setLossTolerance(0.02);
        net.setBatchSize(100);
    }
    else{
        net.load();
    }
    net.printArchitecture();
    net.setUnlearningLayerID(2);
    net.setLearningRate(learningRate);

    int epoch= 20000;
    float accuracy = 0;
    for (int i=0; i<epoch; ++i){
        net.train();
        net.save();
        accuracy = net.test();
        cout<<"Epoch_"<<i<<": "<<" accuracy = "<<accuracy<<endl;
    }
    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    return 0;
}