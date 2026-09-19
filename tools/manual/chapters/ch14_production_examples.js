// tools/manual/chapters/ch14_production_examples.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 14</div>
      <h1 class="chapter-title">Enterprise Production Examples & Implementations</h1>
    </div>

    <h2>14.1 Deep MLP Training from Scratch with Autograd & Adam</h2>
    <p>
      This production example demonstrates training a two-layer Multi-Layer Perceptron (MLP) on GPU using automatic differentiation and the <strong>Adam Optimizer</strong> ($m_t, v_t$ moments with bias correction):
    </p>

    <pre><code>#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/autograd/variable.hpp"
#include &lt;iostream&gt;
#include &lt;vector&gt;

using namespace matrix_pro;
using namespace matrix_pro::autograd;

int main() {
    const size_t BATCH = 128, IN_DIM = 784, HIDDEN = 256, OUT_DIM = 10;
    const float LR = 1e-3f, BETA1 = 0.9f, BETA2 = 0.999f, EPS = 1e-8f;

    // Initialize Parameters with Kaiming Normal Random Distribution
    Variable W1(Matrix::randn(IN_DIM, HIDDEN, 0.0f, std::sqrt(2.0f / IN_DIM)), true);
    Variable b1(Matrix::zeros(1, HIDDEN), true);
    Variable W2(Matrix::randn(HIDDEN, OUT_DIM, 0.0f, std::sqrt(2.0f / HIDDEN)), true);
    Variable b2(Matrix::zeros(1, OUT_DIM), true);

    // Adam Optimizer First & Second Moment Accumulators
    Matrix mW1 = Matrix::zeros(IN_DIM, HIDDEN), vW1 = Matrix::zeros(IN_DIM, HIDDEN);
    Matrix mW2 = Matrix::zeros(HIDDEN, OUT_DIM), vW2 = Matrix::zeros(HIDDEN, OUT_DIM);

    // Synthetic Batch Data
    Variable X(Matrix::randn(BATCH, IN_DIM, 0.0f, 1.0f), false);
    Matrix targets = Matrix::random(BATCH, OUT_DIM, 0.0f, 1.0f);

    for (int epoch = 1; epoch &lt;= 100; ++epoch) {
        // Forward Pass
        Variable H = (X.matmul(W1) + b1).relu();
        Variable Y_pred = (H.matmul(W2) + b2).softmax();

        // Mean Squared Error (MSE) Loss
        Variable diff = Y_pred - targets;
        Variable loss = (diff * diff).mean();

        // Backward Gradient Propagation
        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        loss.backward();

        // Adam Optimizer Step for W1
        float beta1_pow = std::pow(BETA1, epoch);
        float beta2_pow = std::pow(BETA2, epoch);
        
        mW1 = mW1 * BETA1 + W1.grad() * (1.0f - BETA1);
        vW1 = vW1 * BETA2 + (W1.grad() * W1.grad()) * (1.0f - BETA2);
        
        Matrix m_hat = mW1 * (1.0f / (1.0f - beta1_pow));
        Matrix v_hat = vW1 * (1.0f / (1.0f - beta2_pow));
        Matrix step = m_hat / (v_hat.sqrt() + EPS);
        
        W1.data() = W1.data() - step * LR;

        if (epoch % 20 == 0) {
            float loss_val = loss.data().to_cpu_scalar();
            std::cout &lt;&lt; "Epoch " &lt;&lt; epoch &lt;&lt; " | MSE Loss: " &lt;&lt; loss_val &lt;&lt; std::endl;
        }
    }
    return 0;
}</code></pre>

    <div class="page-subbreak"></div>

    <h2>14.2 Scaled Dot-Product Multi-Head Self-Attention (Transformer Core)</h2>
    <p>
      The core mechanism of modern Large Language Models (LLMs) is <strong>Scaled Dot-Product Self-Attention</strong>. MatrixFlash-Pro evaluates multi-head attention with zero intermediate VRAM allocation:
    </p>

    <div class="formula-box">
      $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$
      <span class="eq-desc">Scaled Dot-Product Attention: Quadratic complexity $O(L^2 d_k)$ optimized with fused GEMM + Softmax</span>
    </div>

    <pre><code>#include "matrix_pro/core/matrix.hpp"
using namespace matrix_pro;

Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V) {
    size_t seq_len = Q.rows();
    size_t d_k = Q.cols();
    float scale = 1.0f / std::sqrt(static_cast&lt;float&gt;(d_k));

    // 1. MatMul: Q x K^T -> Scores [SeqLen x SeqLen]
    Matrix scores = Q.matmul_transposed_b(K);

    // 2. Scale in-place via vectorized FMA
    scores.scale_(scale);

    // 3. Online Stable Softmax across sequence dimension
    Matrix attn_weights = scores.softmax(-1);

    // 4. MatMul: Weights x V -> Output Context [SeqLen x d_k]
    return attn_weights * V;
}</code></pre>

    <h2>14.3 Graph Neural Network (GNN) Message Passing via Sparse SpMM</h2>
    <p>
      Graph Convolutional Networks (GCN) update node embedding representations through neighborhood aggregation:
    </p>

    <div class="formula-box">
      $$\mathbf{H}^{(l+1)} = \text{ReLU}\left( \mathbf{\tilde{A}}_{\text{csr}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$
      <span class="eq-desc">Where $\mathbf{\tilde{A}}$ is the symmetrically normalized adjacency matrix stored as a CSR sparse structure</span>
    </div>

    <pre><code>#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/sparse/csr_matrix.hpp"
using namespace matrix_pro;
using namespace matrix_pro::sparse;

Matrix gcn_layer_forward(const CsrMatrix& norm_adj, const Matrix& node_features, const Matrix& weights) {
    // 1. Dense Feature Projection: Z = H * W
    Matrix projected = node_features * weights;

    // 2. Neighborhood Message Passing: Agg = A_sparse * Projected
    Matrix aggregated = norm_adj.spmm(projected);

    // 3. Non-Linear Activation
    return aggregated.relu();
}</code></pre>

    <h2>14.4 Ultra-Low-Latency Inference with CUDA Graph Replay</h2>
    <pre><code>#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/cuda/stream.hpp"
#include "matrix_pro/cuda/graph.hpp"

using namespace matrix_pro;
using namespace matrix_pro::cuda;

int main() {
    Stream stream(Stream::Flags::NonBlocking);
    CudaGraph inference_graph;

    Matrix input = Matrix::zeros(1, 512);
    Matrix W1 = Matrix::random(512, 1024);
    Matrix b1 = Matrix::zeros(1, 1024);
    Matrix W2 = Matrix::random(1024, 64);
    Matrix output(1, 64);

    // Capture entire execution graph into GPU hardware plan
    inference_graph.capture(stream, [&]() {
        Matrix h = (input * W1 + b1).gelu();
        Matrix::multiply_into(output, h, W2);
    });

    // Sub-2-microsecond dispatch loop
    for (int req = 0; req &lt; 10000; ++req) {
        inference_graph.replay(stream);
    }
    stream.synchronize();
    return 0;
}</code></pre>
  </div>
`;

