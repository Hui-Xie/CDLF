//
// Created by Sheen156 on 8/6/2018.
//

#include "MnistTools.h"
#include "MNIST.h"

const string MnistDir= "E:\\CProject\\mnist";

int main (int argc, char *argv[])
{
    MNIST mnist(MnistDir);
    mnist.loadData();

    //test the matching between image and label
    long index = 25031;
    mnist.displayImage(mnist.m_pTrainImages, index);
    cout<<"Image is "<<(int)(mnist.m_pTrainLabels->e(index))<<endl;

    cout<<"Info: MNIST Test finished."<<endl;
    return 0;
}