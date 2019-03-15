#include <iostream>
#include <vector>
using namespace std;

// the characters description of target file vector
/*
const vector<string> fileCharacVec= {
        "HNSCC-01-0001",
        "HNSCC-01-0002",
        "HNSCC-01-0003",
        "HNSCC-01-0004",
        "HNSCC-01-0005",
        "HNSCC-01-0007",
        "HNSCC-01-0008",
        "HNSCC-01-0009",
        "HNSCC-01-0013",
        "HNSCC-01-0014",
        "HNSCC-01-0015",
        "HNSCC-01-0016",
        "HNSCC-01-0017",
        "HNSCC-01-0018",
//"HNSCC-01-0019",
        "HNSCC-01-0020",
        "HNSCC-01-0021",
        "HNSCC-01-0022",
        "HNSCC-01-0026",
        "HNSCC-01-0027",
        "HNSCC-01-0029",
        "HNSCC-01-0032",
        "HNSCC-01-0033",
//"HNSCC-01-0034",
        "HNSCC-01-0035",
        "HNSCC-01-0039",
        "HNSCC-01-0040",
        "HNSCC-01-0041",
        "HNSCC-01-0046",
        "HNSCC-01-0047",
        "HNSCC-01-0048",
        "HNSCC-01-0055",
        "HNSCC-01-0063",
        "HNSCC-01-0064",
        "HNSCC-01-0065",
        "HNSCC-01-0067",
        "HNSCC-01-0070",
        "HNSCC-01-0071",
        "HNSCC-01-0072",
        "HNSCC-01-0077",
        "HNSCC-01-0081",
        "HNSCC-01-0082",
        "HNSCC-01-0083",
        "HNSCC-01-0084",
        "HNSCC-01-0085",
        "HNSCC-01-0087",
        "HNSCC-01-0090",
        "HNSCC-01-0091",
        "HNSCC-01-0092",
        "HNSCC-01-0093",
        "HNSCC-01-0094",
        "HNSCC-01-0097",
        "HNSCC-01-0098",
        "HNSCC-01-0099",
        "HNSCC-01-0100",
        "HNSCC-01-0101",
        "HNSCC-01-0102",
        "HNSCC-01-0104",
        "HNSCC-01-0106",
        "HNSCC-01-0107",
        "HNSCC-01-0110",
        "HNSCC-01-0120",
        "HNSCC-01-0121",
        "HNSCC-01-0123",
        "HNSCC-01-0125",
        "HNSCC-01-0127",
        "HNSCC-01-0128",
        "HNSCC-01-0129",
        "HNSCC-01-0131",
        "HNSCC-01-0132",
        "HNSCC-01-0133",
        "HNSCC-01-0134",
        "HNSCC-01-0135",
//"HNSCC-01-0136",
        "HNSCC-01-0137",
        "HNSCC-01-0138",
        "HNSCC-01-0139",
        "HNSCC-01-0140",
        "HNSCC-01-0141",
        "HNSCC-01-0142",
        "HNSCC-01-0144",
        "HNSCC-01-0146",
        "HNSCC-01-0147",
        "HNSCC-01-0149"
};
*/

const vector<string> fileCharacVec={
"HNSCC-01-0151",
"HNSCC-01-0155",
"HNSCC-01-0158",
//"HNSCC-01-0159",
"HNSCC-01-0160",
"HNSCC-01-0161",
//"HNSCC-01-0163",
"HNSCC-01-0166",
"HNSCC-01-0168",
"HNSCC-01-0171",
"HNSCC-01-0172",
"HNSCC-01-0173",
"HNSCC-01-0177",
"HNSCC-01-0178",
//"HNSCC-01-0181",
"HNSCC-01-0182",
"HNSCC-01-0192",
"HNSCC-01-0199",
"HNSCC-01-0201",
"HNSCC-01-0205",
"HNSCC-01-0210",
//"HNSCC-01-0211",
"HNSCC-01-0213",
"HNSCC-01-0214"
};

string getFilename(const string& characStr){
    return characStr+"_GTV.nrrd";
}



void printUsage(char* argv0){
    cout<<"============= Copy specific images from a directory to another ==========="<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <srcDir>  <dstDir>"<<endl;
}

int main(int argc, char *argv[]) {
    if (3 != argc) {
        cout << "Error: number of parameters error." << endl;
        printUsage(argv[0]);
        return -1;
    }

    const string srcDir = argv[1];
    const string dstDir = argv[2];

    const int N = fileCharacVec.size();
    for (int i=0; i<N; ++i){
        const string characStr = fileCharacVec[i];
        const string filename = getFilename(characStr);
        const string srcFullPath = srcDir + "/" + filename;
        const string cmdStr = "cp " + srcFullPath + " " + dstDir;
        int result = std::system(cmdStr.c_str());
        if (0 != result){
            cout<<"Error "<<cmdStr<<endl;
        }
    }

    cout<<"Info: total copied "<<N << " files to "<< dstDir<<endl;
}