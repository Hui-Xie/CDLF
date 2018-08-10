//
// Created by Hui Xie on 8/6/2018.
//

#include "MnistTools.h"
#include "MNIST.h"

const string mnistDir= "E:\\CProject\\mnist";

int main (int argc, char *argv[])
{
    MNIST mnist(mnistDir);
    mnist.loadData();

    cout<<"test the matching between image and label in whole dataset"<<endl;
    srand (time(NULL));
    long index = rand() % 10000;
    mnist.displayImage(mnist.m_pTestImages, index);
    cout<<"Image is "<<(int)(mnist.m_pTestLabels->e(index))<<endl;

    //tailor data
    mnist.tailorData();

    cout<<"test the matching between image and label in part dataset"<<endl;
    index = rand() % 10000;
    mnist.displayImage(mnist.m_pTrainImagesPart, index);
    cout<<"Image is "<<(int)(mnist.m_pTrainLabelsPart->e(index))<<endl;

    mnist.buildNet();
    mnist.setNetParameters();
    mnist.m_net.printArchitecture();
    long epoch= 10;//1000;
    float accuracy = 0;
    for (long i=0; i<epoch; ++i){
        mnist.trainNet();
        accuracy = mnist.testNet();
        cout<<"Epoch_"<<i<<": "<<" accuracy = "<<accuracy<<endl;
     }
    cout<<"==========End of Mnist Test==========="<<endl;
    return 0;
}