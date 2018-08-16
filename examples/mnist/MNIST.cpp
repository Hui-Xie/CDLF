//
// Created by Hui Xie on 8/8/2018.
//

#include "MNIST.h"
#include <fstream>


MNIST::MNIST(const string &mnistDir) {
    m_mnistDir = mnistDir;
    m_trainImageFile = m_mnistDir + "\\train-images.idx3-ubyte";
    m_trainLabelFile = m_mnistDir + "\\train-labels.idx1-ubyte";
    m_testImageFile = m_mnistDir + "\\t10k-images.idx3-ubyte";
    m_testLabelFile = m_mnistDir + "\\t10k-labels.idx1-ubyte";

    m_pTrainImages = nullptr;
    m_pTrainLabels = nullptr;
    m_pTestImages = nullptr;
    m_pTestLabels = nullptr;

    m_partDigits = {0,1}; //user can change number of elements to many, at least 2.

    m_pTrainImagesPart = nullptr;
    m_pTrainLabelsPart = nullptr;
    m_pTestImagesPart = nullptr;
    m_pTestLabelsPart = nullptr;
}

MNIST::~MNIST() {
    deleteWholeDataSet();
    deletePartDataSet();
}

void MNIST::deleteWholeDataSet(){
    if (nullptr != m_pTrainImages) {
        delete m_pTrainImages;
        m_pTrainImages = nullptr;
    }

    if (nullptr != m_pTrainLabels) {
        delete m_pTrainLabels;
        m_pTrainLabels = nullptr;
    }

    if (nullptr != m_pTestImages) {
        delete m_pTestImages;
        m_pTestImages = nullptr;
    }

    if (nullptr != m_pTestLabels) {
        delete m_pTestLabels;
        m_pTestLabels = nullptr;
    }
}

void MNIST::deletePartDataSet(){
    if (nullptr != m_pTrainImagesPart) {
        delete m_pTrainImagesPart;
        m_pTrainImagesPart = nullptr;
    }

    if (nullptr != m_pTrainLabelsPart) {
        delete m_pTrainLabelsPart;
        m_pTrainLabelsPart = nullptr;
    }

    if (nullptr != m_pTestImagesPart) {
        delete m_pTestImagesPart;
        m_pTestImagesPart = nullptr;
    }

    if (nullptr != m_pTestLabelsPart) {
        delete m_pTestLabelsPart;
        m_pTestLabelsPart = nullptr;
    }
}

long MNIST::hexChar4ToLong(char *buff) {
    long temp = 0;
    for (int i = 0; i < 4; ++i) {
        temp += ((unsigned char) buff[i]) * pow(16, (3 - i) * 2);
    }
    return temp;
}

bool MNIST::isDigitInVector(const unsigned char digit){
    const int N = m_partDigits.size();
    for (int i=0; i<N; ++i){
        if (digit == m_partDigits[i]) return true;
    }
    return false;
}

void MNIST::extractPart(const Tensor<unsigned char> * pWholeImages,  const Tensor<unsigned char> * pWholeLabels,
                 Tensor<unsigned char> * pPartImages,  Tensor<unsigned char> *  pPartLabels){
    long N = pWholeLabels->getLength();
    long NPart = 0;
    long imageSize = 28*28*sizeof(unsigned char);
    for(long i=0; i<N; ++i){
        if (isDigitInVector(pWholeLabels->e(i))) {
            pPartLabels->e(NPart) = pWholeLabels->e(i);
            memcpy(pPartImages->getData()+NPart*imageSize, pWholeImages->getData()+i*imageSize, imageSize );
            ++NPart;
        }
    }
}

int MNIST::readIdxFile(const string &fileName, Tensor<unsigned char> *&pTensor) {
    ifstream ifs(fileName, ifstream::in | ifstream::binary);
    ifs.seekg(ios_base::beg);
    if (!ifs.good()) {
        ifs.close();
        cout << "Error: read file error: " << fileName << endl;
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

        } else if (0x01 == magicNum[3]) {// Label file
            ifs.read(dim, 4);
            numImages = hexChar4ToLong(dim);
            isImage = false;
        } else {
            cout << "Error: incorrect magic number in Idx file. Exit." << endl;
            ifs.close();
            return 2;
        }
    } else {
        cout << "Error: incorrect Idx file. Exit." << endl;
        ifs.close();
        return 3;
    }

    if (isImage) {
        pTensor = new Tensor<unsigned char>({numImages, rows, cols});
    } else {
        pTensor = new Tensor<unsigned char>({numImages, 1});
    }
    long numBytes = pTensor->getLength() * sizeof(unsigned char);
    char *buff = new char[numBytes];
    ifs.read(buff, numBytes);
    pTensor->copyDataFrom(buff, numBytes);
    delete[] buff;

    ifs.close();
    return 0;
}

void MNIST::loadData() {
    readIdxFile(m_trainImageFile, m_pTrainImages);
    readIdxFile(m_trainLabelFile, m_pTrainLabels);
    readIdxFile(m_testImageFile, m_pTestImages);
    readIdxFile(m_testLabelFile, m_pTestLabels);
}

void MNIST::tailorData(){
    //get the total number of part train dataset
    long N = m_pTrainLabels->getLength();
    long NTrainPart = 0;
    for (long i=0;i<N;++i){
        if (isDigitInVector(m_pTrainLabels->e(i))){
            ++NTrainPart;
        }
    }
    cout<<"Info: MNIST tain part dataset has total "<<NTrainPart<<" elements."<<endl;
    m_pTrainImagesPart = new  Tensor<unsigned char> ({NTrainPart,28,28});
    m_pTrainLabelsPart = new  Tensor<unsigned char> ({NTrainPart,1});
    extractPart(m_pTrainImages,m_pTrainLabels,m_pTrainImagesPart, m_pTrainLabelsPart);

    //get the total number of part test dataset
    N = m_pTestLabels->getLength();
    long NTestPart = 0;
    for (long i=0;i<N;++i){
        if (isDigitInVector(m_pTestLabels->e(i))){
            ++NTestPart;
        }
    }
    cout<<"Info: MNIST test part dataset has total "<<NTestPart<<" elements."<<endl;
    m_pTestImagesPart = new  Tensor<unsigned char> ({NTestPart,28,28});
    m_pTestLabelsPart = new  Tensor<unsigned char> ({NTestPart,1});
    extractPart(m_pTestImages,m_pTestLabels,m_pTestImagesPart, m_pTestLabelsPart);
}

void MNIST::displayImage(Tensor<unsigned char> *pImages, const long index) {
    Tensor<unsigned char> slice = pImages->slice(index);
    slice.printElements(true);
}

void MNIST::buildNet() {
    // it is a good design if all numFilter is odd;
    int layerID = 1;
    InputLayer *inputLayer = new InputLayer(layerID++, "InputLayer", {28, 28}); //output size: 28*28
    m_net.addLayer(inputLayer);

    ConvolutionLayer *conv1 = new ConvolutionLayer(layerID++, "Conv1", {3, 3}, inputLayer, 5, 1); //output size: 5*26*26
    m_net.addLayer(conv1);
    ReLU *reLU1 = new ReLU(layerID++, "ReLU1", conv1);
    m_net.addLayer(reLU1);
    NormalizationLayer *norm1 = new NormalizationLayer(layerID++, "Norm1", reLU1);
    m_net.addLayer(norm1);

    ConvolutionLayer *conv2 = new ConvolutionLayer(layerID++, "Conv2", {5,3,3}, norm1, 5, 1);//output size: 5*24*24
    m_net.addLayer(conv2);
    ReLU *reLU2 = new ReLU(layerID++, "ReLU2", conv2);
    m_net.addLayer(reLU2);
    NormalizationLayer *norm2 = new NormalizationLayer(layerID++, "Norm2", reLU2);
    m_net.addLayer(norm2);

    ConvolutionLayer *conv3 = new ConvolutionLayer(layerID++, "Conv3", {3,3, 3}, norm2, 7, 1);//output size: 7*3*22*22
    m_net.addLayer(conv3);
    ReLU *reLU3 = new ReLU(layerID++, "ReLU3", conv3);
    m_net.addLayer(reLU3);
    NormalizationLayer *norm3 = new NormalizationLayer(layerID++, "Norm3", reLU3);
    m_net.addLayer(norm3);

    ConvolutionLayer *conv4 = new ConvolutionLayer(layerID++, "Conv4", {7,3,3,3}, norm3, 1, 1);//output size: 20*20
    m_net.addLayer(conv4);
    ReLU *reLU4 = new ReLU(layerID++, "ReLU4", conv4);
    m_net.addLayer(reLU4);
    NormalizationLayer *norm4 = new NormalizationLayer(layerID++, "Norm4", reLU4);
    m_net.addLayer(norm4);

    VectorizationLayer *vecLayer1 = new VectorizationLayer(layerID++, "Vector1", norm4); //output size: 400*1
    m_net.addLayer(vecLayer1);
    FCLayer *fcLayer1 = new FCLayer(layerID++, "FC1", {100, 1}, vecLayer1); //output size: 100*1
    m_net.addLayer(fcLayer1);
    ReLU *reLU5 = new ReLU(layerID++, "ReLU5", fcLayer1); //output size: 100*1
    m_net.addLayer(reLU5);
    NormalizationLayer *norm5 = new NormalizationLayer(layerID++, "Norm5", reLU5);
    m_net.addLayer(norm5);

    FCLayer *fcLayer2 = new FCLayer(layerID++, "FC2", {10, 1}, norm5); //output size: 10*1
    m_net.addLayer(fcLayer2);
    ReLU *reLU6 = new ReLU(layerID++, "ReLU6", fcLayer2);
    m_net.addLayer(reLU6);
    NormalizationLayer *norm6 = new NormalizationLayer(layerID++, "Norm6", reLU6);
    m_net.addLayer(norm6);

    //For 2 category case
    FCLayer *fcLayer3 = new FCLayer(layerID++, "FC3", {2, 1}, norm6); //output size: 2*1
    m_net.addLayer(fcLayer3);

    SoftMaxLayer *softmaxLayer = new SoftMaxLayer(layerID++, "Softmax1", fcLayer3); //output size: 2*1
    m_net.addLayer(softmaxLayer);
    CrossEntropyLoss *crossEntropyLoss = new CrossEntropyLoss(layerID++, "CrossEntropy",
                                                              softmaxLayer); // output size: 1
    m_net.addLayer(crossEntropyLoss);

}


void MNIST::setNetParameters() {
    m_net.setLearningRate(0.001);
    m_net.setLossTolerance(0.02);
    m_net.setBatchSize(100);
    m_net.initialize();

}

//construct a 10*1 one-hot vector
Tensor<float> MNIST::constructGroundTruth(Tensor<unsigned char> *pLabels, const long index) {
    Tensor<float> tensor({2, 1});// for 2 categories case
    tensor.zeroInitialize();
    tensor.e(pLabels->e(index)) = 1; //for {0,1} case
    return tensor;
}

void MNIST::trainNet() {
    InputLayer *inputLayer = (InputLayer *) m_net.getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) m_net.getFinalLayer();

    long maxIteration =m_pTrainLabelsPart->getLength();
    long NTrain = maxIteration;
    int batchSize = m_net.getBatchSize();
    float learningRate = m_net.getLearningRate();
    long numBatch = maxIteration / batchSize;
    if (0 != maxIteration % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long nBatch = 0;
    //random reshuffle data samples
    vector<long> randSeq = generateRandomSequence(NTrain);
    while (nBatch < numBatch) {
        m_net.zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < maxIteration; ++i) {
            inputLayer->setInputTensor(m_pTrainImagesPart->slice(randSeq[nIter]));
            lossLayer->setGroundTruth(constructGroundTruth(m_pTrainLabelsPart, randSeq[nIter]));
            m_net.forwardPropagate();
            m_net.backwardPropagate();
            ++nIter;
        }
        m_net.sgd(learningRate, i);
        ++nBatch;
    }
  }

float MNIST::testNet() {
    InputLayer *inputLayer = (InputLayer *) m_net.getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) m_net.getFinalLayer();
    long n = 0;
    long nSuccess = 0;
    const long Ntest = m_pTestLabelsPart->getLength();
    while (n < Ntest) {
        inputLayer->setInputTensor(m_pTestImagesPart->slice(n));
        lossLayer->setGroundTruth(constructGroundTruth(m_pTestLabelsPart, n));
        m_net.forwardPropagate();
        if (lossLayer->predictSuccess()) ++nSuccess;
        ++n;
    }
    m_accuracy = nSuccess * 1.0 / Ntest;
    return m_accuracy;
}