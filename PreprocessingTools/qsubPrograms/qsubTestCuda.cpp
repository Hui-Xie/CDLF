//
// Created by Hui Xie on 9/22/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <iostream>
using namespace std;

string cmdPath = "/Users/hxie1/temp_release/Examples/TestCudaConv/TestConvCuda";
string cmdPara = "";


int main(int argc, char *argv[]) {
    // notes: all command paramaeters have a space at front and at tail
    string jobName = "CudaTest";
    string qsubStrBasic = string(" qsub -b y -cwd ")
                          + " -N " + jobName + " "
                          + " -q COE-GPU"
                          + " -pe smp 10"
                          + " -e ~/temp_qsub/Error_" + jobName + ".txt "
                          + " -o ~/temp_qsub/StdOutput_" + jobName + ".txt ";
    string qsubStrCmd = " " + cmdPath +" "+ cmdPara;

    string qsubStr = qsubStrBasic + " " + qsubStrCmd;
    system(qsubStr.c_str());

    cout << "qsubTest submitted:" << jobName<<endl;
    return 0;

}