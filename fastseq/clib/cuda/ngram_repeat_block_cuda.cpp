/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

#include <torch/extension.h>
#include <vector>

/*
CPP Binding for CUDA OP
*/

// CUDA forward declarations
void ngram_repeat_block_cuda_forward(const int64_t* tokens, const int max_predict_len,
                                     float* lprobs, const int vocab_size,
                                     int bsz, int step, int beam_size,
                                     int no_repeat_ngram_size);

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// Input check and call to CUDA OP
// Backward method not required
torch::Tensor ngram_repeat_block_forward(torch::Tensor tokens,
                                         torch::Tensor lprobs, int bsz,
                                         int step, int beam_size,
                                         int no_repeat_ngram_size) {
  CHECK_INPUT(tokens);
  CHECK_INPUT(lprobs);
  assert(bsz > 0);
  assert(step >= 0);
  assert(beam_size > 0);
  assert(no_repeat_ngram_size > 0);

  const int max_predict_len = tokens.size(1);
  const int vocab_size = lprobs.size(1);

  ngram_repeat_block_cuda_forward(tokens.data_ptr<int64_t>(),
                                  max_predict_len,
                                  lprobs.data_ptr<float>(),
                                  vocab_size,
                                  bsz, step, beam_size,
                                  no_repeat_ngram_size);
  return lprobs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ngram_repeat_block_forward,
        "No Repeat Ngram Block forward (CUDA)");
}
