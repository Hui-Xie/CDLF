//
// Created by Sheen156 on 8/6/2018.
//

#include "MnistTools.h"
#include <fstream>

long hexchar4ToLong(char *buff){
    long  temp =0;
    for (int i=0; i<4; ++i){
       temp += ((unsigned char)buff[i])* pow(16,(3-i)*2);
    }
    return temp;
}

int readMNISTIdxFile(const string &fileName, Tensor<unsigned char> *pTensor) {
    ifstream ifs(fileName, ifstream::in | ifstream::binary);
    ifs.seekg(ios_base::beg);
    if (!ifs.good()) {
        ifs.close();
        cout << "Error: read file error: " << fileName<<endl;
        return 1;
    }

    long numImages = 0;
    long rows = 0;
    long cols = 0;
    bool isImage = true;

    //read magic number and dimension
    char magicNum[4];
    char dim[4];
    ifs.read(magicNum, 4);
    if (0x00 == magicNum[0] && 0x00 == magicNum[1] && 0x08 == magicNum[2]) {
        if (0x03 == magicNum[3]) {//Image file
            ifs.read(dim, 4);
            numImages =  hexchar4ToLong(dim);
            ifs.read(dim, 4);
            rows = hexchar4ToLong(dim);
            ifs.read(dim, 4);
            cols = hexchar4ToLong(dim);
            isImage = true;

        }else if (0x01 == magicNum[3]) {// Label file
            ifs.read(dim, 4);
            numImages = hexchar4ToLong(dim);
            isImage = false;
        }
        else{
            cout << "Error: incorrect magic number in Idx file. Exit." << endl;
            return 2;
        }
    } else {
        cout << "Error: incorrect Idx file. Exit." << endl;
    }

    if (isImage){
        pTensor = new Tensor<unsigned char>({numImages,rows, cols});

    }
    else{
        pTensor = new Tensor<unsigned char>({numImages, 1});

    }




    ifs.close();
    return 0;
}
