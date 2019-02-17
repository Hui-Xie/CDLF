
#include "HNSurvivalNet.h"


HNSurvivalNet::HNSurvivalNet(const string &name, const string &netDir) : FeedForwardNet(name, netDir) {
    m_pDataMgr = nullptr;
    m_pClinicalDataMgr = nullptr;
}

HNSurvivalNet::~HNSurvivalNet() {
    //null
}

void HNSurvivalNet::build() {
    //null: use csv file to create network
}

void HNSurvivalNet::setInput(const string &filename,const vector<int>& center) {
    InputLayer *inputLayer = getInputLayer();
    Tensor<float>* pImage = nullptr;
    m_pDataMgr->readImageFile(filename, pImage);
    Tensor<float>* pSubImage = new Tensor<float>(inputLayer->m_tensorSize);
    const vector<int> stride1 = vector<int>(inputLayer->m_tensorSize.size(),1);
    const vector<int> topLeft = m_pDataMgr->getTopLeftIndexFrom(pImage->getDims(), inputLayer->m_tensorSize, center);
    pImage->subTensorFromTopLeft(topLeft, pSubImage, stride1);
    inputLayer->setInputTensor(*pSubImage);
    if (nullptr != pImage) {
        delete pImage;
        pImage = nullptr;
    }
    if (nullptr != pSubImage) {
        delete pSubImage;
        pSubImage = nullptr;
    }
}

void HNSurvivalNet::setGroundtruth(const string &filename, const bool printPredict ) {
    CrossEntropyLoss* lossLayer = (CrossEntropyLoss *) getFinalLayer();
    const string patiendCode = m_pClinicalDataMgr->getPatientCode(filename);
    Survival survivalData = m_pClinicalDataMgr->getSurvivalData(patiendCode);
    if (printPredict){
        survivalData.print();
    }
    Tensor<float> gt = m_pClinicalDataMgr->getSurvivalTensor(survivalData);
    lossLayer->setGroundTruth(gt);
}


void HNSurvivalNet::train() {
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();

    m_loss = 0;

    int N =m_pDataMgr->m_NTrainFile;
    if (m_OneSampleTrain){
        N = 1;
    }
    const int batchSize = getBatchSize();
    const float learningRate = getLearningRate();
    const int numBatch = (N + batchSize -1) / batchSize;
    int n = 0;
    int batch = 0;
    vector<int> randSeq = generateRandomSequence(N);
    while (batch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && n < N; ++i) {
            const string imageFilePath = m_pDataMgr->m_trainImagesVector[randSeq[n]];
            const string labelFilePath = m_pDataMgr->getLabelPathFrom(imageFilePath);

            const vector<int> center = m_pDataMgr->getLabelCenter(labelFilePath, false, 0);
            setInput(imageFilePath, center);

            setGroundtruth(imageFilePath);

            forwardPropagate();
            m_loss += lossLayer->getLoss();
            backwardPropagate(true);
            ++n;
        }
        sgd(learningRate, i);
        ++batch;
    }
    m_loss /=n;
    printf("Train: loss = %f \n", m_loss);
}

float HNSurvivalNet::test() {
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();

    m_loss = 0.0;
    int n = 0;
    int N = m_pDataMgr->m_NTestFile;
    if (m_OneSampleTrain){
        N = 1;
    }
    while (n < N) {
        const string imageFilePath = m_pDataMgr->m_testImagesVector[n];
        const string labelFilePath = m_pDataMgr->getLabelPathFrom(imageFilePath);
        const vector<int> center = m_pDataMgr->getLabelCenter(labelFilePath, false , 0);
        setInput(imageFilePath, center);
        setGroundtruth(imageFilePath, m_printPredict);
        forwardPropagate();
        if (m_printPredict){
            printPredict();
        }
        m_loss += lossLayer->getLoss();
        ++n;

    }
    m_loss /=n;
    printf("Test: loss = %f \n", m_loss);
    return  m_loss;

}

void HNSurvivalNet::test(const bool printResult) {
     m_printPredict = printResult;
     test();
     m_printPredict = false;
}

void HNSurvivalNet::printPredict() {
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    SoftmaxLayer* pSoftmax = (SoftmaxLayer*)lossLayer->m_prevLayer;

    cout<<"10-year survival prediction:"<<endl;
    pSoftmax->m_pYTensor->row(1).print();
}


