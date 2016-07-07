#include <vector>

#include "caffe/layers/ExSigmoidCrossEntropyLossLayer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ExSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  weightPos_ = this->layer_param_.ex_sigmoid_cross_entropy_loss_param().weight_pos();
}

template <typename Dtype>
void ExSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void ExSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype eta = (Dtype)(1e-6f);
  Dtype loss = 0;

  for (int i=0;i<count;++i) {
    Dtype p = std::min(std::max(sigmoid_output_data[i],eta),1-eta);
    loss -= this->weightPos_*target[i] * log(p) + (1-target[i])*log(1-p);
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void ExSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    //const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    Dtype n = 0;
    for (int i=0;i<count;++i) {
      //Dtype p = std::min(std::max(sigmoid_output_data[i],eta),1-eta);
      bottom_diff[i] = this->weightPos_*target[i] * (sigmoid_output_data[i]-1) +
          (1-target[i])*sigmoid_output_data[i];
      n += this->weightPos_*target[i] + (1-target[i]);
    }
    
    // Scale down gradient
    //const Dtype loss_weight = top[0]->cpu_diff()[0];
    const Dtype loss_weight = static_cast<Dtype>(1.0f);
    caffe_scal(count, loss_weight / n, bottom_diff);
  }
}



#ifdef CPU_ONLY
STUB_GPU_BACKWARD(ExSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(ExSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(ExSigmoidCrossEntropyLoss);

}  // namespace caffe
