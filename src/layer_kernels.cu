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
/*
  Modified by Li Sijin, 2013  lisijin7@gmail.com
 */
#include <assert.h>

#include <layer_kernels.cuh>

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
__global__ void kLogregCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxp = maxProbs[tx];
        const float labelp = probs[label * numCases + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        /*
         * Compute the probability of guessing the correct case if you take the most-probable label.
         * 
         * This is done like this:
         * 
         * - If the most probable label is not equal to the true label, then the probability is zero.
         * - Otherwise, the probability is 1 / (number of labels whose probability is equal to the maximum).
         * 
         * This is certainly overkill -- in practice, it's just about impossible for two labels to get assigned
         * maximum probability. But it's a safety measure to prevent over-estimating your accuracy.
         * Though it could never happen in reality. Well it could. But it wouldn't. Cool?
         */
        if (labelp != maxp) {
            correctProbs[tx] = 0;
        } else {
            int numMax = 0;
            for (int i = 0; i < numOut; i++) {
                numMax += probs[i * numCases + tx] == maxp;
            }
            correctProbs[tx] = 1.0f / float(numMax);
        }
    }
}


/*
  Please don't use any special neuron (such as logistic, tanh) before this layer
  
  z_i is the input from the previous layer, y_i is the ground truth vectorized 1-of-K map
  x_i = logistic(z_i)
  what indlogpred calculate is y_i log( x_i) + (1-y_i) log(1-x_i)
  predmap:    (numTasks, numCases)
  indlogpred: (numTasks, numCases)
  correctprobs:(numTasks, numCases)

  each thread is responsible for per_thread_case position in one task
  blockIdx.x determines which task(indicator) to take
 */

__global__ void kEltwiseLogregCost(float* predmap, float* indmap, float*indlogpred, float* correctprobs, int numCases, int numTasks, int per_thread_case) {
  const int task_id = blockIdx.x;
  const int start_tx = threadIdx.x * per_thread_case;
  const int end_tx = min(start_tx + per_thread_case, numCases);
  if (task_id >= numTasks) {
    return;
  }
  for (int c_id = start_tx; c_id < end_tx; ++c_id) {
    int pos = task_id * numCases + c_id;
    float t = __fdividef(1.0f, 1.0f + __expf(-predmap[ pos ]));   
    if (indmap[pos] == 1) {
      indlogpred[pos] = __logf(t);
      correctprobs[pos] = t;
    } else {
      t = 1-t;
      indlogpred[pos] = __logf(t);
      correctprobs[pos] = t;
    }
  }
}

/*
  z_i is the input of previous layer
  x_i = logistic(z_i)
  Calculate the gradient of f(z_i) = y_i log x_i + (1-y_i) log(1-x_i)
  df_dz = [yi/xi + (yi-1)/(1-xi)]*(1-xi)(xi) = [yi * (1-xi) + (yi-1)*xi] = [yi - xi]
  predmap:       (numTasks, numCases)
  indmap:        (numCases, numCases)
  df_dz:    (numCases, numCases)

  each thread is responsible for per_thread_case cases in one task
  each block is responsible for one task  
 */
template <bool add>
__global__ void kEltwiseLogregGrad(float * predmap, float* indmap, float* df_dz, int numCases, int numTasks, int per_thread_case, float coeff ) {
  const int task_id = blockIdx.x;
  const int start_tx = threadIdx.x * per_thread_case;
  const int end_tx = min(start_tx + per_thread_case, numCases);
  if (task_id >= numTasks) {
    return;
  }
  for (int c_id = start_tx; c_id < end_tx; ++c_id) {
    int pos = task_id * numCases + c_id;
    float v = coeff * (indmap[pos] - __fdividef(1.0f, 1.0f + __expf(-predmap[ pos ])));   
    if (add) {
      df_dz[pos] += v;
    } else {
      df_dz[pos] = v;
    }
  }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dy_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregCostGrad(float* y_l, float* labels, float* dE_dy_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * (label == ty);
        v = __fdividef(v, y_l[tidx]);
        if (add) {
            dE_dy_l[tidx] += v;
        } else {
            dE_dy_l[tidx] = v;
        }
    }
}

/*
 * dE_dy_l: (numOut, numCases)
 * y_l:     (numOut, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kSoftmaxGrad(float* dE_dy_l, float* y_l, float* dE_dx_l, const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        float v = 0;
        for (int j = 0; j < numOut; j++) {
            v += dE_dy_l[j * numCases + tx] * ((j == ty) - y_l[j * numCases + tx]);
        }
        v *= y_l[tidx];
        
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregSoftmaxGrad(float* y_l, float* labels, float* dE_dx_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * ((label == ty) - y_l[tidx]);
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

template <int B_X, bool add>
__global__ void kEltwiseMaxGrad(float* actGrad, float* input, float* output, float* target,
                                const int numElements) {
    for (int i = B_X * blockIdx.x + threadIdx.x; i < numElements; i += B_X * gridDim.x) {
        if (add) {
            target[i] += actGrad[i] * (output[i] == input[i]);
        } else {
            target[i] = actGrad[i] * (output[i] == input[i]);
        }
    }
}

void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add) {
    assert(actGrad.isContiguous());
    assert(output.isContiguous());
    assert(input.isContiguous());
    assert(actGrad.isSameDims(input));
    assert(actGrad.isSameDims(output));
    
    dim3 blocks(DIVUP(actGrad.getNumElements(), 128));
    dim3 threads(128);
    if (add) {
        assert(actGrad.isSameDims(target));
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, true>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, true><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    } else {
        target.resize(actGrad);
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, false>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, false><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    }
    
    getLastCudaError("computeEltwiseMaxGrad: Kernel execution failed");
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    
    NVMatrix& maxProbs = probs.max(0);
    
    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaFuncSetCacheConfig(kLogregCost, cudaFuncCachePreferL1);
    kLogregCost<<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     labelLogProbs_out.getDevData(), correctProbs_out.getDevData(),
                                     numCases, numOut);
    getLastCudaError("computeLogregCost: Kernel execution failed");
//    cudaThreadSynchronize();
    delete &maxProbs;
}

void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregCostGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregCostGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    getLastCudaError("computeLogregGrad: Kernel execution failed");
}

void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, bool add) {
    int numCases = acts.getLeadingDim();
    int numOut = acts.getFollowingDim();

    assert(acts.isSameDims(actsGrad));
    assert(acts.isContiguous());
    assert(actsGrad.isContiguous());
    assert(target.isContiguous());
    assert(acts.isTrans());
    assert(actsGrad.isTrans());

    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(acts);
        kSoftmaxGrad<false><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    } else {
        kSoftmaxGrad<true><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    }
    getLastCudaError("computeSoftmaxGrad: Kernel execution failed");
}

void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregSoftmaxGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregSoftmaxGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    getLastCudaError("computeLogregSoftmaxGrad: Kernel execution failed");
}


void computeEltwiseLogregCost(NVMatrix& indmap, NVMatrix& predmap, NVMatrix & indlogpred, NVMatrix& correctprobs) {
  int numCases = predmap.getNumCols();
  int numTasks = predmap.getNumRows();
  assert(indmap.getNumCols() == numCases);
  assert(indmap.getNumRows() == numTasks);
  assert(!indmap.isTrans());
  assert(!predmap.isTrans());
  assert(indmap.isContiguous());
  assert(predmap.isContiguous());

  indlogpred.resize(numTasks, numCases);
  correctprobs.resize(numTasks, numCases);
  dim3 threads(ELTLOGREG_ERR_THREADS_X, 1);
  dim3 blocks(numTasks, 1); // Ensure the numTasks will not exceed GPU's capacity
  int per_thread_case = DIVUP( numCases, ELTLOGREG_ERR_THREADS_X); 
  cudaFuncSetCacheConfig(kEltwiseLogregCost, cudaFuncCachePreferL1);
  kEltwiseLogregCost<<<blocks, threads>>>(predmap.getDevData(), indmap.getDevData(), indlogpred.getDevData(), correctprobs.getDevData(), numCases, numTasks, per_thread_case);
  getLastCudaError("computeEltwiseLogregCost: Kernel execution failed");
}

void computeEltwiseLogregGrad(NVMatrix& indmap, NVMatrix& predmap, NVMatrix& target, bool add, float coeff) {
  int numCases = predmap.getLeadingDim();
  int numTasks = predmap.getFollowingDim();
  assert( indmap.getLeadingDim() == numCases);
  assert( indmap.getFollowingDim() == numTasks);
  assert(!indmap.isTrans());
  assert(!predmap.isTrans());
  assert(indmap.isContiguous());
  assert(predmap.isContiguous());

  dim3 threads(ELTLOGREG_ERR_THREADS_X, 1);
  dim3 blocks(numTasks, 1); // Ensure the numTasks will not exceed GPU's capacity
  int per_thread_case = DIVUP( numCases, ELTLOGREG_ERR_THREADS_X);
  if (!add) {
    target.resize(predmap);
    kEltwiseLogregGrad<false><<<blocks, threads>>>(predmap.getDevData(), indmap.getDevData(), target.getDevData(), numCases, numTasks, per_thread_case, coeff);
  } else {
    kEltwiseLogregGrad<true><<<blocks, threads>>>(predmap.getDevData(), indmap.getDevData(), target.getDevData(), numCases, numTasks, per_thread_case, coeff);
  }
  getLastCudaError("computeEltwiseLogregGrad: Kernel execution failed");
}
