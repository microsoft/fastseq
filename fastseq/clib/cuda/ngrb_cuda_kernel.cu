#include <torch/extension.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void getBannedTokens(long * __restrict__ tokens,
                float * __restrict__ lprobs, int cols,
                int lprob_cols, int no_repeat_ngram_size) {
    auto row = blockIdx.x;
    auto col = threadIdx.x;
    auto start = row*(cols)+ col;
    auto check_start_pos =  blockDim.x;
    auto lprob_start = row *lprob_cols;
    bool is_banned = true;
    extern __shared__ long tokens_shm[];
    tokens_shm[col] = tokens[start];
    if (col == blockDim.x -1) {
        tokens_shm[col+1] = tokens[start+1];
        tokens_shm[col+2] = tokens[start+2];
    }
    __syncthreads();

    for (int k = 0; k < no_repeat_ngram_size -1; k++) {
        if (tokens_shm[col+k] !=  tokens_shm[check_start_pos+k]) {
           is_banned = false;
        }
    }
    if (is_banned == true) {
        auto token_to_be_banned =  tokens_shm[col+no_repeat_ngram_size-1];
        lprobs[lprob_start + token_to_be_banned] = -INFINITY;
    }
    }

torch::Tensor ngrb_cuda_forward(
    torch::Tensor tokens,
    torch::Tensor lprobs,
    int bsz,
    int step,
    int beam_size,
    int no_repeat_ngram_size
    ) {
  int cols = tokens.size(1);
  int lprob_cols = lprobs.size(1);
  auto token_ptr = tokens.data_ptr<long>();
  auto lprob_ptr = lprobs.data_ptr<float>();
  int blocks  = bsz*beam_size;
  int no_repeat_ngram = 3;
  int threads = step - no_repeat_ngram +2;
  int shared_mem_size = (step+1) *sizeof(long);
  if (threads <=0) return lprobs;
  getBannedTokens<<<blocks, threads, shared_mem_size >>> (
           token_ptr, lprob_ptr, cols,
           lprob_cols, no_repeat_ngram_size);
  return lprobs;
}

