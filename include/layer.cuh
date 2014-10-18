/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef LAYER_CUH
#define	LAYER_CUH

#include <string>
#include <vector>
#include <map>
#include <assert.h>
#include <nvmatrix.cuh>

#include "convnet.cuh"
#include "cost.cuh"
#include "weights.cuh"
#include "neuron.cuh"

class Cost;
class ConvNet;
class CostLayer;
class DataLayer;

/*
 * Abstract layer.
 */
class Layer {
protected:
    ConvNet* _convNet;
    std::vector<Layer*> _prev, _next;
    int _rcvdFInputs, _rcvdBInputs;
    
    NVMatrixV _inputs;
    NVMatrix *_outputs; // TODO: make this a pointer so you can reuse previous layers' matrices
    NVMatrix *_actsGrad; // Layer activity gradients
    bool _gradConsumer, _foundGradConsumers, _trans;
    bool _conserveMem;
    int _numGradProducersNext;
    int _actsTarget, _actsGradTarget;
    std::string _name, _type;
    NVMatrix _dropout_mask;
    float _dropout_prob;
    void fpropNext(PASS_TYPE passType);
    virtual void truncBwdActs(); 
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) = 0;
    
    virtual void bpropCommon(NVMatrix& v, PASS_TYPE passType) {
        // Do nothing by default
    }
    virtual void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
        assert(!isGradProducer()); // Only do nothing if not grad producer
    }
public:    
    static bool _saveActsGrad, _saveActs;
    
    Layer(ConvNet* convNet, PyObject* paramsDict, bool trans);
    
    virtual void fprop(PASS_TYPE passType);
    void fprop(NVMatrix& v, PASS_TYPE passType);
    virtual void fprop(NVMatrixV& v, PASS_TYPE passType);
    virtual void bprop(PASS_TYPE passType);
    void bprop(NVMatrix& v, PASS_TYPE passType);
    virtual void reset();
    int incRcvdBInputs();
    int getRcvdFInputs();
    int getRcvdBInputs();
    bool isGradConsumer();
    virtual bool isGradProducer();
    std::string& getName();
    std::string& getType();
    void addNext(Layer* l);
    void addPrev(Layer* l);
    std::vector<Layer*>& getPrev();
    std::vector<Layer*>& getNext();
    virtual NVMatrix& getActs();
    virtual NVMatrix& getActsGrad();
    virtual void postInit();
    
    // Do nothing if this layer has no weights
    virtual void updateWeights() {
    }
    virtual void checkGradients() {
    }
    virtual void copyToCPU() {
    }
    virtual void copyToGPU()  {
    }
};

class NeuronLayer : public Layer {
protected:
    Neuron* _neuron;
    
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    virtual void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    NeuronLayer(ConvNet* convNet, PyObject* paramsDict);
};

class WeightLayer : public Layer {
protected:
    WeightList _weights;
    Weights *_biases;
    float _wStep, _bStep;
    
    void bpropCommon(NVMatrix& v, PASS_TYPE passType);
    virtual void bpropBiases(NVMatrix& v, PASS_TYPE passType) = 0;
    virtual void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) = 0;
public:
    WeightLayer(ConvNet* convNet, PyObject* paramsDict, bool trans, bool useGrad);
    virtual void updateWeights();
    virtual void copyToCPU();
    virtual void copyToGPU();
    void checkGradients();
    Weights& getWeights(int idx);
};

class FCLayer : public WeightLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
public:
    FCLayer(ConvNet* convNet, PyObject* paramsDict);
};

class SoftmaxLayer : public Layer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    SoftmaxLayer(ConvNet* convNet, PyObject* paramsDict);
};

class EltwiseSumLayer : public Layer {
protected:
    vector<float>* _coeffs;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    EltwiseSumLayer(ConvNet* convNet, PyObject* paramsDict);
};

class ForwardLayer: public Layer {
 protected:
  std::string _random_type;
  vector <float> _params;
  bool _pass_gradients;
  bool _add_noise;
  void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
  void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
 public:
  ForwardLayer(ConvNet* convNet, PyObject* paramsDict);
  // void set_params(const vector<float> &params);
  void set_params(const float* params, int len);
};

class EltwiseMulLayer : public Layer {
 protected:
  void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
  void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType); public:
  EltwiseMulLayer(ConvNet* convNet, PyObject* paramsDict);
};

class SliceLayer : public Layer {
 private:
  int _startX, _endX;
 protected:
  void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
  void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
 public:
  SliceLayer(ConvNet* convnet, PyObject* paramsDict);
  virtual ~SliceLayer();
};

class ConcatenationLayer : public Layer {
 private:
  int _numOutputs;
 protected:
    intv* _copyOffsets;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
 public:
    ConcatenationLayer(ConvNet* convnet, PyObject* paramsDict);
    virtual ~ConcatenationLayer();
};


class EltwiseMaxLayer : public Layer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    EltwiseMaxLayer(ConvNet* convNet, PyObject* paramsDict);
};

class DataLayer : public Layer {
private:
    int _dataIdx;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    DataLayer(ConvNet* convNet, PyObject* paramsDict);
    
    bool isGradProducer();
    void fprop(PASS_TYPE passType);
    void fprop(NVMatrixV& data, PASS_TYPE passType);
};

class LocalLayer : public WeightLayer {
protected:
    struct FilterConns {
        int* hFilterConns;
        int* dFilterConns;
    };
    vector<FilterConns>* _filterConns;
    
    intv* _padding, *_stride, *_filterSize, *_channels, *_imgSize, *_groups;
    intv* _imgPixels, *_filterPixels, *_filterChannels, *_overSample, *_randSparse;
    int _modulesX, _modules, _numFilters;

    void copyToGPU();
    
public:
    LocalLayer(ConvNet* convNet, PyObject* paramsDict, bool useGrad);
};

class ConvLayer : public LocalLayer {
protected:
    int _partialSum;
    bool _sharedBiases;
    
    NVMatrix _weightGradTmp, _actGradTmp;

    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
    void truncBwdActs();

public:
    ConvLayer(ConvNet* convNet, PyObject* paramsDict);
}; 

class LocalUnsharedLayer : public LocalLayer {
protected:
    NVMatrix _sexMask;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
public:
    LocalUnsharedLayer(ConvNet* convNet, PyObject* paramsDict);
}; 

class PoolLayer : public Layer {
protected:
    int _channels, _sizeX, _start, _stride, _outputsX;
    int _imgSize;
    string _pool;
public:
    PoolLayer(ConvNet* convNet, PyObject* paramsDict, bool trans);
    
    static PoolLayer& makePoolLayer(ConvNet* convNet, PyObject* paramsDict);
}; 

class AvgPoolLayer : public PoolLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    AvgPoolLayer(ConvNet* convNet, PyObject* paramsDict);
}; 

class MaxPoolLayer : public PoolLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    MaxPoolLayer(ConvNet* convNet, PyObject* paramsDict);
};

class NailbedLayer : public Layer {
protected:
    int _channels, _start, _stride, _outputsX;
    int _imgSize;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    
    NailbedLayer(ConvNet* convNet, PyObject* paramsDict);
};

class GaussianBlurLayer : public Layer {
protected:
    int _channels;
    Matrix* _hFilter;
    NVMatrix _filter;
    NVMatrix _actGradsTmp;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void copyToGPU();
    
    GaussianBlurLayer(ConvNet* convNet, PyObject* paramsDict);
};

class ResizeLayer : public Layer {
protected:
    int _channels;
    float _scale;
    int _imgSize, _tgtSize;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);

    ResizeLayer(ConvNet* convNet, PyObject* paramsDict);
};

class RGBToYUVLayer : public Layer {
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);

    RGBToYUVLayer(ConvNet* convNet, PyObject* paramsDict);
};

class RGBToLABLayer : public Layer {
protected:
    bool _center;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);

    RGBToLABLayer(ConvNet* convNet, PyObject* paramsDict);
};

class ResponseNormLayer : public Layer {
protected:
    int _channels, _size;
    float _scale, _pow;
    NVMatrix _denoms;

    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
public:
    ResponseNormLayer(ConvNet* convNet, PyObject* paramsDict);
}; 

class CrossMapResponseNormLayer : public ResponseNormLayer {
protected:
    bool _blocked;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    CrossMapResponseNormLayer(ConvNet* convNet, PyObject* paramsDict);
}; 

class ContrastNormLayer : public ResponseNormLayer {
protected:
    int _imgSize;
    NVMatrix _meanDiffs;
    
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
public:
    ContrastNormLayer(ConvNet* convNet, PyObject* paramsDict);
};

class CostLayer : public Layer {
protected:
    float _coeff;
    doublev _costv;
public:
    CostLayer(ConvNet* convNet, PyObject* paramsDict, bool trans);
    void bprop(PASS_TYPE passType); 
    virtual doublev& getCost();
    float getCoeff();
    bool isGradProducer();
    
    static CostLayer& makeCostLayer(ConvNet* convNet, string& type, PyObject* paramsDict);
};

/*
 * Input 0: labels
 * Input 1: softmax outputs
 */
class LogregCostLayer : public CostLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    LogregCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

class SumOfSquaresCostLayer : public CostLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    SumOfSquaresCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

class EltwiseLogregCostLayer : public CostLayer {
 protected:
  void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
  void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
 public:
  EltwiseLogregCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

class EltwiseL2SVMCostLayer : public CostLayer {
 protected:
  float _a, _b;
  void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
  void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
 public:
  EltwiseL2SVMCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

class LoglikeGaussianCostLayer : public CostLayer {
 protected:
  int _close_form_freq;
  int _close_form_update_count;
  WeightList _weights;
  float _wStep;
  float _ubound;
  float _lbound;
  bool _use_ubound;
  bool _use_lbound;
  bool _use_log_weights;
  void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
  void bpropCommon(NVMatrix& v, PASS_TYPE passType);
  void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
  virtual void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
 public:
  LoglikeGaussianCostLayer(ConvNet* convNet, PyObject* paramsDict);
  virtual void updateWeights();
  virtual void copyToCPU();
  virtual void copyToGPU();
  Weights& getWeights(int idx); // Maybe someone may want to call it in the future
};

class SSVMCostLayer: public CostLayer {
 protected:
  NVMatrix _act_max_ind;
  NVMatrix _act_max_value;
  int _num_groups;
  void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
  void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
 public:
  SSVMCostLayer(ConvNet* convNet, PyObject* paramsDict);
  virtual ~SSVMCostLayer();  
};

class LogisticCostLayer: public CostLayer {
 protected:
  float _a, _u;
  Neuron * _neuron;
  void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
  void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
 public:
  LogisticCostLayer(ConvNet* convNet, PyObject* paramsDict);
  virtual ~LogisticCostLayer();
};



#endif	/* LAYER_CUH */

