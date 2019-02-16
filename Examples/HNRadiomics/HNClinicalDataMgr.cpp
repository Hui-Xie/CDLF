
#include "HNClinicalDataMgr.h"
#include <fstream>
#include <iostream>

HNClinicalDataMgr::HNClinicalDataMgr() {
   m_survivalVector.clear();
}

HNClinicalDataMgr::~HNClinicalDataMgr() {

}


//TableHead: TCIACode, Age, Survival(months), AliveOrDead, CauseofDeath
//DataType:  string,   int, float,            string,      string
//Example: HNSCC-01-0006	62	118.9333333   	Dead	NonCancerRelated
void HNClinicalDataMgr::readSurvivalData(const string &filename) {
    ifstream ifs(filename.c_str());;
    char lineChar[80];
    char patientCode[20];
    char aliveOrDead[10];
    char causeofDeath[20];
    if (ifs.good()) {
        ifs.ignore(80, '\n'); // ignore the table head
        while (ifs.peek() != EOF) {
            ifs.getline(lineChar, 80, '\n');
            for (int i = 0; i < 80; ++i) {
                if (lineChar[i] == ',') lineChar[i] = ' ';
            }

            struct Survival survival;
            int nFills = sscanf(lineChar, "%s  %d  %f  %s  %s \r\n",
                                patientCode, &survival.m_age, &survival.m_survivalMonth, aliveOrDead, causeofDeath);
            if (5 != nFills) {
                cout << "Error: sscanf survival data in readSurvivalData at line: " << string(lineChar) << endl;
            } else {
                survival.m_patientCode = string(patientCode);
                survival.m_isAlive = ("Alive"== string(aliveOrDead))? true : false;
                survival.m_causeOfDeath = string(causeofDeath);
                m_survivalVector.push_back(survival);
            }
        }
    }
    ifs.close();

}

Tensor<float> HNClinicalDataMgr::getSurvivalTensor(struct Survival &survival) {
    Tensor<float> result({10,2});
    const int survivalYears = int(survival.m_survivalMonth/12.0 +0.5);
    for(int i=0; i< survivalYears && i<10; ++i){
        result.e({i,0}) = 0; //death
        result.e({i,1}) = 1; //alive
    }
    if (survival.m_isAlive){
        //use survival function sqrt(100-x)/10, where x \in [0, 100];
        for(int i=survivalYears; i<10; ++i){
            const int  age = survival.m_age+ i;
            const float s = sqrt(100.0-age)/10.0;
            result.e({i,0}) = 1-s; //death
            result.e({i,1}) = s; //alive
        }
    }
    else{
        for(int i=survivalYears; i<10; ++i){
            result.e({i,0}) = 1; //death
            result.e({i,1}) = 0; //alive
        }
    }
    return result;
}
