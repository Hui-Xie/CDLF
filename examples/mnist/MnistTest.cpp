//
// Created by Sheen156 on 8/6/2018.
//

#include "MnistTools.h"
#include "MNIST.h"

const string mnistDir= "E:\\CProject\\mnist";

int main (int argc, char *argv[])
{
    MNIST mnist(mnistDir);
    mnist.loadData();

    //test the matching between image and label, it is good.
    long index = 4879;
    mnist.displayImage(mnist.m_pTestImages, index);
    cout<<"Image is "<<(int)(mnist.m_pTestLabels->e(index))<<endl;

    mnist.buildNet();

    mnist.trainNet();

    cout<<"==========End of Mnist Test==========="<<endl;
    return 0;
}