#include <functional>
#include <utility>
#include <vector>
#include <iostream>

#include "caffe/layers/ExAccuracyLayer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ExAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ExAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ExAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  double accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  
  const int count = bottom[0]->count();
  
  for (int i=0;i<count;i++) {
    if (bottom_label[i]==bottom_data[i]) {
      accuracy++;
    }
  }
  top[0]->mutable_cpu_data()[0] = (Dtype) accuracy/count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(ExAccuracyLayer);
REGISTER_LAYER_CLASS(ExAccuracy);

}  // namespace caffe
