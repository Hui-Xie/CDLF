//
// Created by hxie1 on 9/8/18.
//

// qSubProgram submits file for HPC server processing.
// It is better for you to manually modify this program to qsub for HPC computation

#include <cstdlib>
#include "FileTools.h"


const string program = "./UniformImage";
const string inputDir = "/Users/hxie1/msd/Task07_Pancreas/imagesTr";
const string pathSuffix = "_uniform";
const string labelChange = "0";
const string sizeX = "277";
const string sizeY = "277";
const string sizeZ = "120";
const string spacingX = "1";
const string spacingY = "1";
const string spacingZ = "1";

int main(int argc, char *argv[]) {

    vector<string> fileVector;
    getFileVector(inputDir, fileVector);
    const long numFiles = fileVector.size();
    cout<<"Totally read "<<numFiles <<" files in "<<inputDir<<endl;

    for(long i=0; i<numFiles; ++i){
        // notes: all command paramaeters have a space at front and at tail
        string jobName = "X" + to_string(i);
        string qsubStrBasic = string(" qsub -b y -cwd ")
                 + " -N " + jobName +" "
                 + " -q COE,COE-GPU,UI-DEVELOP "
                 + " -pe smp 1 "
                 + " -e ~/temp_qsub/Error_"+jobName+".txt "
                 + " -o ~/temp_qsub/StdOutput_"+jobName+".txt ";
        string qsubStrCmd = " " +program
                          +" " + fileVector[i]
                          +" " + pathSuffix
                          +" " + sizeX
                          +" " + sizeY
                          +" " + sizeZ
                          +" " + spacingX
                          +" " + spacingY
                          +" " + spacingZ;

        string qsubStr = qsubStrBasic +" " + qsubStrCmd;
        system(qsubStr.c_str());
    }
    cout<<"qsub submitted."<<endl;
    return 0;

}