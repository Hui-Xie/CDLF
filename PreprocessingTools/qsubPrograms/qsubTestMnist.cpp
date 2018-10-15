//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <iostream>
using namespace std;

string cmdPath = "/Users/hxie1/temp_release/Examples/TestMnist/TestMnist";
string cmdPara = "/Users/hxie1/Projects/mnist 2D";


int main(int argc, char *argv[]) {
    // notes: all command paramaeters have a space at front and at tail
    string jobName = "M2D_GPU";
    string qsubStrBasic = string(" qsub -b y -cwd ")
                          + " -N " + jobName + " "
                          + " -q COE-GPU "
                          + " -pe smp 20 "
                          + " -e ~/temp_qsub/Error_" + jobName + ".txt "
                          + " -o ~/temp_qsub/StdOutput_" + jobName + ".txt ";
    string qsubStrCmd = " " + cmdPath +" "+ cmdPara;

    string qsubStr = qsubStrBasic + " " + qsubStrCmd;
    system(qsubStr.c_str());

    cout << "qsubTestMnist submitted: " << jobName <<endl;
    return 0;

}