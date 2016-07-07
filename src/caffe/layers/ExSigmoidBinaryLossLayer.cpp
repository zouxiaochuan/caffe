#include <vector>

#include "caffe/layers/ExSigmoidBinaryLossLayer.hpp"
#include "caffe/util/math_functions.hpp"
#include <sys/time.h>
/*
static struct timeval start1;
static struct timeval start2;
static struct timeval end1;
static struct timeval end2;
static int my_c1 = 0;
static int my_c2= 0;
static double deltatime1 = 0;
static double deltatime2 = 0;*/
namespace caffe 
{    
    template <typename Dtype>
    void ExSigmoidBinaryLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        LossLayer<Dtype>::LayerSetUp(bottom,top);
        sigmoid_bottom_vec_.clear();
        sigmoid_bottom_vec_.push_back(bottom[0]);
        sigmoid_top_vec_.clear();
        sigmoid_top_vec_.push_back(sigmoid_output_.get());
        sigmoid_layer_->SetUp(sigmoid_bottom_vec_,sigmoid_top_vec_);
    }
    
    template <typename Dtype>
    void ExSigmoidBinaryLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
    {
        LossLayer<Dtype>::Reshape(bottom,top);
        CHECK_EQ(bottom[0]->num(),bottom[1]->num()) << "bottom input blobs must have the same num.";
        sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
    }
    
    template <typename Dtype>
    void ExSigmoidBinaryLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,const vector<Blob<Dtype>* >& top)
    {
        /*gettimeofday(&start1,NULL);
        if (my_c1%100 == 0 && my_c1>0)
        {
            std::cout<<"my_count1 = "<<my_c1<<",deltatime1 = "<<deltatime1<<" us"<<std::endl;
        }
        my_c1 += 1;*/
        // compute forward of the sigmoid outputs
        sigmoid_bottom_vec_[0] = bottom[0];
        sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
        
        // compute the loss -1*(zi*log(p) + (1-zi)*log(1-p))
        // p = 1/(1+exp(-x)) if x >=0 and p = exp(x)/(1+exp(x)) if x<0
        const int count = bottom[0]->count();
        const int num = bottom[0]->num();
        const int channels = bottom[0]->channels();
        const int size = count / (num*channels);
        //CHECK_EQ(size,1) << "input data size(w*h) must eq to 1" <<"count:"<< count <<"num:"<< num <<"channels:"<< channels;
        const Dtype* input_data = bottom[0]->cpu_data();
        const Dtype* target_index =  bottom[1]->cpu_data();
        const Dtype* target = bottom[2]->cpu_data();
        
        CHECK_EQ(num,bottom[1]->count()) << "target_index and data num must have the same length";
        CHECK_EQ(bottom[1]->count(),bottom[2]->count()) << "target_index and target must have the same size";
        
        Dtype loss = 0;
        for (int i = 0; i < num; ++i)
        {
            const Dtype input_x = input_data[(i*channels + int(target_index[i]))*size];
            loss -= input_x * (target[i] - (input_x >= 0)) -
                log(1 + exp(input_x - 2*input_x*(input_x >= 0)));
        }
        top[0]->mutable_cpu_data()[0] = loss / num;
        //gettimeofday(&end1,NULL);
        //deltatime1 += (end1.tv_sec - start1.tv_sec)*1000000+(end1.tv_usec - start1.tv_usec);
    }
    
    template <typename Dtype>
    void ExSigmoidBinaryLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top,const vector<bool>& propagate_down,const vector<Blob<Dtype>* >& bottom)
    {
        /*gettimeofday(&start2,NULL);
        if (my_c2%100 == 0 && my_c2>0)
        {
            std::cout<<"my_count2 = "<<my_c2<<",deltatime2 = "<<deltatime2<<" us"<<std::endl;
            start2 = end2;
        }
        my_c2 += 1;*/
        if (propagate_down[1] | propagate_down[2])
        {
            LOG(FATAL) << this->type() << "Layer cannot backpropagate to target_index or target inputs.";
        }
        if (propagate_down[0])
        {
            // compute the diff
            //const int count = bottom[0]->count();
            const int num = bottom[0]->num();
            const int channels = bottom[0]->channels();
            //const int size = count / (num * channels);
            //CHECK_EQ(size,1) << "input data size(w*h) must eq to 1" <<"count:"<< count <<"num:"<< num <<"channels:"<< channels;
            const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
            const Dtype* target_index =  bottom[1]->cpu_data();
            const Dtype* target = bottom[2]->cpu_data();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            
            //CHECK_EQ(num,bottom[1]->count()) << "target_index and data num must have the same length";
            //CHECK_EQ(bottom[1]->count(),bottom[2]->count()) << "target_index and target must have the same size";
            
            // count = N*C*H*W
            // pos = ((n * channels() + c) * height() + h) * width() + w
            //caffe_scal(count,Dtype(0),bottom_diff);
            const Dtype loss_weight = top[0]->cpu_diff()[0];
            const Dtype scal_loss = loss_weight / num;
            for (int i=0; i<num; ++i)
            {
                const int target_pos = int(target_index[i]);
                //bottom_diff[i*channels+target_pos] = (sigmoid_output_data[i*channels+target_pos] - target[i])*scal_loss;
                
                for (int j=0; j<channels; ++j)
                {
                    if (j != target_pos)
                    {
                        bottom_diff[i * channels + j] = Dtype(0);
                    }
                    else
                    {
                        bottom_diff[i * channels + j] = (sigmoid_output_data[i * channels + j] - target[i])*scal_loss;
                    }
                }
            }
            //gettimeofday(&end2,NULL);
            //deltatime2 += (end2.tv_sec - start2.tv_sec)*1000000+(end2.tv_usec - start2.tv_usec);
        }
    }
        
        
    INSTANTIATE_CLASS(ExSigmoidBinaryLossLayer);
    REGISTER_LAYER_CLASS(ExSigmoidBinaryLoss);
        
        
}  // namespace caffe
    
    
    
    
    
    
    
    
    
    
