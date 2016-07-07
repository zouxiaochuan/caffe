#ifndef CAFFE_SIGMOID_TOP_ONE_LOSS_LAYER_HPP_
#define CAFFE_SIGMOID_TOP_ONE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe 
{
    /*
    *@brief Compute the logistic loss from the 
        specific node among all the output nodes
    */
    template <typename Dtype>
    class ExSigmoidBinaryLossLayer : public LossLayer<Dtype> 
    {
        public:
            explicit ExSigmoidBinaryLossLayer(const LayerParameter& param)
                 : LossLayer<Dtype>(param),
            sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
            sigmoid_output_(new Blob<Dtype>()) 
            {
				//to do
            }
            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
            virtual void Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
            virtual inline const char* type() const
            {
                return "ExSigmoidBinaryLoss";
            }
            virtual inline int ExactNumBottomBlobs() const { return 3; }
			
        protected:

            virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom,const vector<Blob<Dtype>* >& top);
            virtual void Backward_cpu(const vector<Blob<Dtype>* >& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>* >& bottom);
            //virtual void Backward_gpu(const vector<Blob<Dtype>* >& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>* >& bottom);

            // this sigmoid layer used to implement logistic fun
            shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;

            // store the output of sigmoid_layer_
            shared_ptr<Blob<Dtype> > sigmoid_output_;

            // bottom vector to call the underlying SigmoidLayer::Forward
            vector<Blob<Dtype>* > sigmoid_bottom_vec_;
            // top vector
            vector<Blob<Dtype>* > sigmoid_top_vec_; 

    };

}  // namespace caffe

#endif // CAFFE_SIGMOID_TOP_ONE_LOSS_LAYER_HPP_

