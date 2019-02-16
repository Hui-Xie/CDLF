
#include "HNSegVNet.h"


HNSegVNet::HNSegVNet(const string &name, const string &netDir) : FeedForwardNet(name, netDir) {
    m_pDataMgr = nullptr;


}

HNSegVNet::~HNSegVNet() {
  //null
}

void HNSegVNet::build() {
   //null: use csv file to create network
}

/*
void HNSegVNet::defineAssemblyLoss() {
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();
    Layer* prevLayer = lossLayer->m_prevLayer;
    lossLayer->addLoss( new SquareLossLayer(-1, "SquareLoss", prevLayer, 1));
    lossLayer->addLoss( new CrossEntropyLoss(-2, "CrossEntropyLoss", prevLayer));
    lossLayer->addLoss( new DiceLossLayer(-3, "DiceLoss", prevLayer));
}
*/

void HNSegVNet::setInput(const string &filename,const vector<int>& center) {
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

void HNSegVNet::setGroundtruth(const string &filename, const vector<int>& center) {
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();

    Tensor<float>* pLabel = nullptr;
    m_pDataMgr->readLabelFile(filename, pLabel);
    Tensor<float>* pSubLabel = nullptr;
    pSubLabel = new Tensor<float>(lossLayer->m_prevLayer->m_tensorSize);


    //  for lossLayer->m_prevLayer is Softmax
    if (pLabel->getDims().size() +1  == lossLayer->m_prevLayer->m_tensorSize.size()){
        const int k = lossLayer->m_prevLayer->m_tensorSize[0];
        Tensor<float>* pOneHotLabel = nullptr;
        m_pDataMgr->oneHotEncodeLabel(pLabel, pOneHotLabel, k);
        const vector<int> strideOneHot = vector<int>(lossLayer->m_prevLayer->m_tensorSize.size(),1);

        //update topLeft index
        vector<int> subImageDims = lossLayer->m_prevLayer->m_tensorSize;
        subImageDims.erase(subImageDims.begin());
        vector<int> topLeft = m_pDataMgr->getTopLeftIndexFrom(pLabel->getDims(), subImageDims, center);
        topLeft.insert(topLeft.begin(), 0);

        pOneHotLabel->subTensorFromTopLeft(topLeft, pSubLabel, strideOneHot);
        if (nullptr != pOneHotLabel) {
            delete pOneHotLabel;
            pOneHotLabel = nullptr;
        }
    }
        // for lossLayer->m_prevLayer is Sigmoid
    else if (pLabel->getDims().size() == lossLayer->m_prevLayer->m_tensorSize.size()){
        const vector<int> stride1 = vector<int>(pLabel->getDims().size(),1);
        const vector<int> topLeft = m_pDataMgr->getTopLeftIndexFrom(pLabel->getDims(), lossLayer->m_prevLayer->m_tensorSize, center);
        pLabel->subTensorFromTopLeft(topLeft, pSubLabel, stride1);
    }
    else{
        cout<<"Error: lossLayer->prevLayer size does not match label image size."<<endl;
        std::exit(EXIT_FAILURE);
    }

    lossLayer->setGroundTruth(*pSubLabel);

    if (nullptr != pLabel) {
        delete pLabel;
        pLabel = nullptr;
    }
    if (nullptr != pSubLabel) {
        delete pSubLabel;
        pSubLabel = nullptr;
    }
}


void HNSegVNet::train() {
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();

    m_loss = 0;
    m_dice =0;
    m_TPR = 0;

    const int N =m_pDataMgr->m_NTrainFile;
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

            // support random translation in all axes direction within 15 pixels
            const vector<int> center = m_pDataMgr->getLabelCenter(labelFilePath, true, 15);
            setInput(imageFilePath, center);
            setGroundtruth(labelFilePath,center);

            forwardPropagate();
            m_loss += lossLayer->getLoss();
            // for softmax preceeds over loss layer
            if (m_isSoftmaxBeforeLoss){
                m_dice += lossLayer->diceCoefficient();
                m_TPR  += lossLayer->getTPR();
            }
            else{
                m_dice += lossLayer->diceCoefficient(0.5);
                m_TPR  += lossLayer->getTPR(0.5);
            }

            backwardPropagate(true);
            //debug
            //saveYTensor();
            //savedYTensor();
            ++n;
        }
        sgd(learningRate, i);
        ++batch;



    }
    m_loss /=n;
    m_dice /= n;
    m_TPR /= n;

    printf("Train: loss = %f, Dice = %f, TPR = %f; \n", m_loss, m_dice, m_TPR);
}

float HNSegVNet::test() {
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();

    m_loss = 0.0;
    m_dice = 0;
    m_TPR = 0;

    int n = 0;
    const int N = m_pDataMgr->m_NTestFile;
    while (n < N) {
        const string imageFilePath = m_pDataMgr->m_testImagesVector[n];
        const string labelFilePath = m_pDataMgr->getLabelPathFrom(imageFilePath);
        const vector<int> center = m_pDataMgr->getLabelCenter(labelFilePath, false , 0);
        setInput(imageFilePath, center);
        setGroundtruth(labelFilePath,center);
        forwardPropagate();
        m_loss += lossLayer->getLoss();

        // for softmax preceeds over loss layer
        if (m_isSoftmaxBeforeLoss){
            m_dice += lossLayer->diceCoefficient();
            m_TPR  += lossLayer->getTPR();
        }
        else{
            m_dice += lossLayer->diceCoefficient(0.5);
            m_TPR  += lossLayer->getTPR(0.5);
        }

        ++n;

    }
    m_loss /=n;
    m_dice /= n;
    m_TPR /= n;

    printf("Test: loss = %f, Dice = %f, TPR = %f; \n", m_loss, m_dice, m_TPR);

    return  m_loss;

}

float HNSegVNet::test(const string &imageFilePath, const string &labelFilePath, const vector<int>& center) {
    InputLayer *inputLayer = getInputLayer();
    DiceLossLayer *lossLayer = (DiceLossLayer *) getFinalLayer();

    m_loss = 0.0;
    m_dice = 0.0;
    m_TPR = 0.0;

    setInput(imageFilePath, center);
    if (!labelFilePath.empty()){
        setGroundtruth(labelFilePath, center);
    }
    forwardPropagate();

    vector<int> offset = m_pDataMgr->getOutputOffset(lossLayer->m_prevLayer->m_tensorSize);

    // output float image for debug
    if (!m_isSoftmaxBeforeLoss) {
        const string floatImageOutput = m_pDataMgr->generateFloatImagePath(imageFilePath);
        m_pDataMgr->saveImage2File(lossLayer->m_prevLayer->m_pYTensor, offset, floatImageOutput);
    }

    //Output network predicted label
    string outputLabelFilePath = m_pDataMgr->generateLabelFilePath(imageFilePath);
    if (m_isSoftmaxBeforeLoss){
       m_pDataMgr->saveOneHotCode2LabelFile(lossLayer->m_prevLayer->m_pYTensor, outputLabelFilePath, inputLayer->m_tensorSize);
    }
    else{
        Tensor<unsigned char> predictResult(lossLayer->m_prevLayer->m_tensorSize);
        lossLayer->getPredictTensor(predictResult, 0.5);
        m_pDataMgr->saveLabel2File(&predictResult, offset, outputLabelFilePath);
    }

    if (!labelFilePath.empty()){
        m_loss = lossLayer->getLoss();
        if (m_isSoftmaxBeforeLoss){
            m_dice = lossLayer->diceCoefficient();
            m_TPR  = lossLayer->getTPR();
        }
        else{
            m_dice = lossLayer->diceCoefficient(0.5);
            m_TPR  = lossLayer->getTPR(0.5);
        }
        printf("Test: loss = %f, Dice = %f, TPR = %f; \n", m_loss, m_dice, m_TPR);
    }
    return m_loss;
}

void HNSegVNet::detectSoftmaxBeforeLoss() {
    m_isSoftmaxBeforeLoss = true;
    if ("SigmoidLayer" == getFinalLayer()->m_prevLayer->m_type){
        m_isSoftmaxBeforeLoss = false;
    }
}

