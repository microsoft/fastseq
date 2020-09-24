#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor ngrb_cuda_forward(
    torch::Tensor tokens,
    torch::Tensor lprobs,
    int bsz,
    int step,
    int beam_size,
    int no_repeat_ngram_size);

torch::Tensor ngrb_forward(
    torch::Tensor tokens,
    torch::Tensor lprobs,
    int bsz,
    int step,
    int beam_size,
    int no_repeat_ngram_size
    ) {
  return ngrb_cuda_forward(tokens, lprobs, bsz, step,
         beam_size, no_repeat_ngram_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ngrb_forward, "NGRB forward (CUDA)");
}
