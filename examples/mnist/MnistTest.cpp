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

    //test the matching between image and label, it is good.
    srand (time(NULL));
    long index = rand() % 10000;
    mnist.displayImage(mnist.m_pTestImages, index);
    cout<<"Image is "<<(int)(mnist.m_pTestLabels->e(index))<<endl;

    mnist.buildNet();
    mnist.setNetParameters();
    mnist.m_net.printArchitecture();
    long epoch= 10;//1000;
    for (long i=0; i<epoch; ++i){
        cout<<"%%%%%%%%%%%%%%%%%%   Start Epoch: "<<i<<"   %%%%%%%%%%%%%%%%%%"<<endl;
        mnist.trainNet();
        mnist.testNet();
        cout<<"%%%%%%%%%%%%%%%%%%   End of Epoch: "<<i<<"   %%%%%%%%%%%%%%%%%%"<<endl;
        cout<<endl;
    }
    cout<<"==========End of Mnist Test==========="<<endl;
    return 0;
}