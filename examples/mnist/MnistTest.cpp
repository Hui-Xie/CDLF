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

    cout<<"Info: MNIST Test finished."<<endl;
    return 0;
}