#include <gflags/gflags.h>
#include <glog/logging.h>
#include <lmdb.h>
#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <sys/stat.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

using namespace std;
using namespace boost;
using namespace caffe;

typedef float Dtype;

void extract(const string& outname, const string& paramname,
             const string& protoname,
             const vector<string>& blobnames, int nBatch)
{
    boost::shared_ptr<Net<Dtype> > net
        (new Net<Dtype>(protoname, caffe::TEST));

    net->CopyTrainedLayersFrom(paramname);
    
    //std::vector<Blob<float>*> input_vec;

    ofstream outfile(outname.c_str());
    CHECK(!outfile.fail()) << "Failed to open file: " << outname;
    
    for(int i=0;i<nBatch;i++)
    {
        net->Forward();
        int batchSize = net->blob_by_name(blobnames[0])->num();
        
        for(int b=0;b<batchSize;b++)
        {
            for(int j=0;j<blobnames.size();j++)
            {
                if (j>0)
                {
                    outfile << ";";
                }
                const boost::shared_ptr<Blob<Dtype> > feature_blob = net
                    ->blob_by_name(blobnames[j]);
                int size = feature_blob->count()/batchSize;
                
                for(int k=0;k<size;k++)
                {
                    if (k>0)
                    {
                        outfile << ",";
                    }
                    outfile << feature_blob->cpu_data()[b*size+k];
                }
            }
            outfile << endl;
        }
    }
}

int main(int argn, char** argv)
{
    if (argn!=6)
    {
        cout << "usage: blobnames,outname,caffemodel,prototxt,nbatch" << endl;
    }
    vector<string> blobnames;
    split(blobnames, argv[1], boost::is_any_of(","));
    extract(argv[2], argv[3], argv[4], blobnames, atoi(argv[5]));
}
