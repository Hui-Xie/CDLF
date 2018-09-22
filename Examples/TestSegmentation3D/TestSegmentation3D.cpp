//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "TestSegmentation3D.h"
#include <iostream>
#include "Segmentation3DNet.h"
#include "SegmentGNet.h"
#include "SegmentDNet.h"

#include "StubNetForD.h"
#include "DataManager.h"

using namespace std;

void printUsage(char* argv0){
    cout<<"A Generative Adversarial Network for 3D Medical Images Segmentation: "<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <imageLabelDir>"<<endl;
    cout<<"Where the imageLabelDir must include 4 subdirectories: testImages  testLabels  trainImages  trainLabels" <<endl;
    cout<<"And the corresponding images file and label file should have same filename in different directory. "<<endl;
}

int main(int argc, char *argv[])
{
    printUsage(argv[0]);
    if (2 != argc){
        cout<<"Error: parameter error. Exit. "<<endl;
        return -1;
    }
    string dataSetDir = argv[1];
    DataManager dataMgr(dataSetDir);


    SegmentGNet Gnet("Generative Network");
    Gnet.setLearningRate(0.001);
    Gnet.setLossTolerance(0.02);
    Gnet.setBatchSize(3);
    Gnet.initialize();
    Gnet.printArchitecture();

    SegmentDNet Dnet("Discriminative Network");
    Dnet.setLearningRate(0.001);
    Dnet.setLossTolerance(0.02);
    Dnet.setBatchSize(3);
    Dnet.initialize();
    Dnet.printArchitecture();

    Segmentation3DNet gan("3DSegmentationGAN", &Gnet,&Dnet);
    long epochs = 1000;

    // pretrain DNet
    int epochsPretrainD = 100;
    StubNetForD stubNet("Stub Network for Discriminative Network");
    const long N = dataMgr.m_NTrainFile;
    const int batchSize = 3;
    long numBatch = N / batchSize;
    if (0 != N % batchSize) {
        numBatch += 1;
    }

    for (int epoch=0; epoch<epochsPretrainD; ++epoch){
        long nIter = 0;
        long batch = 0;
        vector<long> randSeq = generateRandomSequence(N);
        while (batch < numBatch) {
            Dnet.zeroParaGradient();
            int i = 0;
            for (i = 0; i < batchSize && nIter < N; ++i) {
                long index = randSeq[nIter];

                Tensor<float>* pImage;
                dataMgr.readTrainImageFile(index, pImage);
                Dnet.m_pInputXLayer->setInputTensor(*pImage);

                Tensor<unsigned char>* pLabel;
                dataMgr.readTrainLabelFile(index, pLabel);
                Dnet.m_pGTLayer->setInputTensor(*pLabel);
                Dnet.setAlpha(true);
                gan.switchDToGT();
                Dnet.forwardPropagate();
                Dnet.backwardPropagate(true);

                gan.setStubLayer(stubNet.getFinalLayer());
                stubNet.randomOutput();
                gan.switchDToStub();
                Dnet.forwardPropagate();
                Dnet.backwardPropagate(true);

                ++nIter;

                delete pImage;
                delete pLabel;

            }
            Dnet.sgd(Dnet.getLearningRate(), i);
            ++batch;
        }


    }




    // train G, D
    // quick alternative train
    float accuracy = 0;
    for (long i=0; i<epochs; ++i){
        gan.trainG(1);
        gan.trainD(1);
        accuracy = gan.testG();
        cout<<"Epoch_"<<i<<": "<<" accuracy = "<<accuracy<<endl;
    }
    //slow alternative train





    cout<<"==========End of Mnist Test==========="<<endl;
    return 0;
}