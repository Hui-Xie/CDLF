
#include "HNClinicalDataMgr.h"
#include <fstream>
#include <iostream>
#include "FileTools.h"


Survival::Survival() {

}

Survival::~Survival() {

}

void Survival::print() {
   printf("Patient=%s, Age=%d, SurvivalYeas=%f, Alive=%d, CauseOfDeath=%s \n", m_patientCode.c_str(), m_age, int(m_survivalMonth/12.0+0.5), m_isAlive, m_causeOfDeath.c_str());
}

//////////////////////////////////////////////////////

HNClinicalDataMgr::HNClinicalDataMgr(const string& clinicalFile) {
   m_survivalVector.clear();
   readSurvivalData(clinicalFile);
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

            Survival survival;
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

Tensor<float> HNClinicalDataMgr::getSurvivalTensor(const Survival &survival) {
    Tensor<float> result({2,10});
    const int survivalYears = int(survival.m_survivalMonth/12.0 +0.5);
    for(int i=0; i< survivalYears && i<10; ++i){
        result.e({0,i}) = 0; //death
        result.e({1,i}) = 1; //alive
    }
    if (survival.m_isAlive){
        //use survival function sqrt(100-x)/10, where x \in [0, 100];
        for(int i=survivalYears; i<10; ++i){
            const int  age = survival.m_age+ i;
            const float s = sqrt(100.0-age)/10.0;
            result.e({0,i}) = 1-s; //death
            result.e({1,i}) = s; //alive
        }
    }
    else{
        for(int i=survivalYears; i<10; ++i){
            result.e({0,i}) = 1; //death
            result.e({1,i}) = 0; //alive
        }
    }
    return result;
}

string HNClinicalDataMgr::getPatientCode(const string &imageFilename) {
    const string filename = getFileName(imageFilename);  // get like this: HNSCC-01-0039_CT.nrrd
    const string code = filename.substr(0, 13);
    return code;
}

struct Survival HNClinicalDataMgr::getSurvivalData(const string &patientCode) {
    const int N = m_survivalVector.size();
    for (int i=0; i<N; ++i){
        if (patientCode == m_survivalVector[i].m_patientCode){
            return m_survivalVector[i];
        }
    }
    cout<<"Error: program can not find suvival data of patientCode:"<<patientCode<<endl;
    return Survival();
}


