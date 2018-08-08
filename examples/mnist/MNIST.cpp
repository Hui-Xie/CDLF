//
// Created by Sheen156 on 8/8/2018.
//

#include "MNIST.h"
#include <fstream>



MNIST::MNIST(const string& mnistDir){
    m_mnistDir = mnistDir;
    m_trainImageFile = m_mnistDir + "\\train-images.idx3-ubyte";
    m_trainLabelFile = m_mnistDir + "\\train-labels.idx1-ubyte";
    m_testImageFile =  m_mnistDir +"\\t10k-images.idx3-ubyte";
    m_testLabelFile = m_mnistDir + "\\t10k-labels.idx1-ubyte";

    m_pTrainImages = nullptr;
    m_pTrainLabels = nullptr;
    m_pTestImages = nullptr;
    m_pTestLabels = nullptr;
}
MNIST::~MNIST(){
    if (nullptr != m_pTrainImages){
        delete m_pTrainImages;
        m_pTrainImages = nullptr;
    }

    if (nullptr != m_pTrainLabels){
        delete m_pTrainLabels;
        m_pTrainLabels = nullptr;
    }

    if (nullptr != m_pTestImages){
        delete m_pTestImages;
        m_pTestImages = nullptr;
    }

    if (nullptr != m_pTestLabels){
        delete m_pTestLabels;
        m_pTestLabels = nullptr;
    }
}

long MNIST::hexChar4ToLong(char *buff){
    long  temp =0;
    for (int i=0; i<4; ++i){
        temp += ((unsigned char)buff[i])* pow(16,(3-i)*2);
    }
    return temp;
}

int MNIST::readIdxFile(const string &fileName, Tensor<unsigned char>* &pTensor){
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
    bool isImage = true; //False is a Label file

    //read magic number and dimension
    char magicNum[4];
    char dim[4];
    ifs.read(magicNum, 4);
    if (0x00 == magicNum[0] && 0x00 == magicNum[1] && 0x08 == magicNum[2]) {
        if (0x03 == magicNum[3]) {//Image file
            ifs.read(dim, 4);
            numImages = hexChar4ToLong(dim);
            ifs.read(dim, 4);
            rows = hexChar4ToLong(dim);
            ifs.read(dim, 4);
            cols = hexChar4ToLong(dim);
            isImage = true;

        }else if (0x01 == magicNum[3]) {// Label file
            ifs.read(dim, 4);
            numImages = hexChar4ToLong(dim);
            isImage = false;
        }
        else{
            cout << "Error: incorrect magic number in Idx file. Exit." << endl;
            ifs.close();
            return 2;
        }
    } else {
        cout << "Error: incorrect Idx file. Exit." << endl;
        ifs.close();
        return 3;
    }

    if (isImage){
        pTensor = new Tensor<unsigned char>({numImages,rows, cols});
    }
    else{
        pTensor = new Tensor<unsigned char>({numImages, 1});
    }
    long numBytes = pTensor->getLength()* sizeof(unsigned char);
    char* buff = new char[numBytes];
    ifs.read(buff, numBytes);
    pTensor->copyDataFrom(buff, numBytes);
    delete[] buff;

    ifs.close();
    return 0;
}

void MNIST::loadData(){
    readIdxFile(m_trainImageFile,m_pTrainImages);
    readIdxFile(m_trainLabelFile,m_pTrainLabels);
    readIdxFile(m_testImageFile,m_pTestImages);
    readIdxFile(m_testLabelFile,m_pTestLabels);
}

void MNIST::displayImage(Tensor<unsigned char>* pImages, const long index){
    Tensor<unsigned char> slice = pImages->slice(index);
    slice.printElements(true);
}

void MNIST::buildNet(){
  int layerID = 1;
  InputLayer* inputLayer = new InputLayer(layerID++, "Input Layer", {28,28}); //output size: 28*28
  m_net.addLayer(inputLayer);
  ConvolutionLayer* conv1 = new ConvolutionLayer(layerID++, "Conv1",{3,3},inputLayer, 1, 1); //output size: 26*26
  m_net.addLayer(conv1);
  ConvolutionLayer* conv2 = new ConvolutionLayer(layerID++, "Conv2",{5,5},conv1, 1, 1);//output size: 22*22
  m_net.addLayer(conv2);
  VectorizationLayer* vecLayer1 = new VectorizationLayer(layerID++, "Vec1", conv2); //output size: 484*1
  m_net.addLayer(vecLayer1);
  FCLayer *fcLayer1 = new FCLayer(layerID++, "fc1", {10,1}, vecLayer1); ////output size: 10*1
  m_net.addLayer(fcLayer1);
  SoftMaxLayer * softmaxLayer = new SoftMaxLayer(layerID++, "softmaxLayer",fcLayer1); //output size: 10*1
  m_net.addLayer(softmaxLayer);
  CrossEntropyLoss* crossEntropyLoss = new CrossEntropyLoss(layerID++, "CrossEntropy"); // output size: 1
  m_net.addLayer(crossEntropyLoss);

}



void MNIST::setNetParameters(){
    m_net.setLearningRate(0.01);
    m_net.setLossTolerance(0.02);
    m_net.setMaxIteration(60000);
    m_net.setBatchSize(200);
    m_net.initialize();
}

void MNIST::trainNet(){
    long nIter = 0;
    InputLayer* inputLayer = (InputLayer*)m_net.getInputLayer();
    CrossEntropyLoss* lossLayer = (CrossEntropyLoss* )m_net.getLossLayer();
    long maxIteration = m_net.getMaxIteration();
    int batchSize = m_net.getBatchSize();
    float learningRate = m_net.getLearningRate();
    long numBatch =  maxIteration / batchSize;
    if (0 !=  maxIteration % batchSize){
        numBatch += 1;
    }

    long nBatch = 0;
    while(nBatch < numBatch)
    {
        if (m_net.getJudgeLoss() && lossLayer->getLoss()< m_net.getLossTolerance()){
            break;
        }
        if (isinf(lossLayer->getLoss())) break;

        m_net.zeroParaGradient();
        int i=0;
        for(i=0; i< batchSize && nIter < maxIteration; ++i){
            inputLayer->setInputTensor(m_pTrainImages->slice(nIter));
            m_net.forwardPropagate();
            m_net.backwardPropagate();
            ++nIter;
        }
        m_net.sgd(learningRate,i);


        m_net.printIteration(lossLayer, nIter);
        ++nBatch;
    }
    lossLayer->printGroundTruth();
}