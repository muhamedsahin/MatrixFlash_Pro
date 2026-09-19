// tools/manual/chapters/ch10_autograd_engine.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 10</div>
      <h1 class="chapter-title">Automatic Differentiation Engine (Dynamic Tape Autograd)</h1>
    </div>

    <h2>10.1 Reverse-Mode Automatic Differentiation Mechanics</h2>
    <p>
      Given a scalar loss objective $\mathcal{L} \in \mathbb{R}$ computed through a Directed Acyclic Graph (DAG) of tensor operations, <strong>Reverse-Mode Automatic Differentiation</strong> computes exact gradients of $\mathcal{L}$ with respect to all intermediate and leaf parameters $\mathbf{x}_j$ in a single backward sweep:
    </p>

    <div class="formula-box">
      $$\bar{\mathbf{x}}_j \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{x}_j} = \sum_{i \in \text{Children}(j)} \bar{\mathbf{y}}_i \frac{\partial \mathbf{y}_i}{\partial \mathbf{x}_j}$$
      <span class="eq-desc">Generalized Vector-Jacobian Product (VJP) Accumulation Formula</span>
    </div>

    <p>
      The computational complexity of computing all parameter gradients is bounded by $c \times \text{Cost}(\text{Forward})$, where $c \le 4$, regardless of whether the model contains 10 parameters or 100 billion parameters.
    </p>

    <h2>10.2 Dynamic Tape Graph & Topological Execution</h2>
    <p>
      MatrixFlash-Pro implements a lightweight, dynamic tape-based computation graph. As operations are executed in user C++ code, each operator appends a closure node containing its backward adjoint logic and saved input tensors:
    </p>

    <div class="arch-diagram">
FORWARD PASS (Eager Tape Recording):
  x1, x2 (Leafs) ---> [ MatMulNode ] ---> y1 = x1 * x2
                             |
                      [ ReLUNode ] ---> y2 = relu(y1)
                             |
                     [ SumReduceNode ] ---> Loss = sum(y2)

BACKWARD PASS (Reverse Topological Sweep):
  Loss.backward()
    |---> dLoss/dy2 = 1.0 (Seed Gradient)
    |---> ReLUNode::backward()      ===> dy1 = dy2 * (y1 > 0)
    |---> MatMulNode::backward()    ===> dx1 = dy1 * x2^T,  dx2 = x1^T * dy1
    |---> Gradients accumulated into x1.grad() and x2.grad()!
    </div>

    <div class="page-subbreak"></div>

    <h2>10.3 Vector-Jacobian Product (VJP) Mathematical Directory</h2>
    <p>
      Every differentiable operator in MatrixFlash-Pro implements an exact, GPU-vectorized Vector-Jacobian Product:
    </p>

    <table>
      <thead>
        <tr>
          <th>Forward Operator</th>
          <th>Forward Expression</th>
          <th>Adjoint Gradient Rule (Backward VJP)</th>
          <th>GPU Implementation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Matrix Multiply</td>
          <td>$\mathbf{Y} = \mathbf{A} \mathbf{B}$</td>
          <td>$\bar{\mathbf{A}} = \bar{\mathbf{Y}} \mathbf{B}^T, \quad \bar{\mathbf{B}} = \mathbf{A}^T \bar{\mathbf{Y}}$</td>
          <td>cuBLAS GEMM with transposed descriptors</td>
        </tr>
        <tr>
          <td>Element-Wise Add</td>
          <td>$\mathbf{Y} = \mathbf{A} + \mathbf{B}$</td>
          <td>$\bar{\mathbf{A}} = \bar{\mathbf{Y}}, \quad \bar{\mathbf{B}} = \bar{\mathbf{Y}}$</td>
          <td>Zero-cost pointer aliasing or FMA</td>
        </tr>
        <tr>
          <td>Hadamard Product</td>
          <td>$\mathbf{Y} = \mathbf{A} \odot \mathbf{B}$</td>
          <td>$\bar{\mathbf{A}} = \bar{\mathbf{Y}} \odot \mathbf{B}, \quad \bar{\mathbf{B}} = \bar{\mathbf{Y}} \odot \mathbf{A}$</td>
          <td>Vectorized 128-bit element-wise kernel</td>
        </tr>
        <tr>
          <td>ReLU Activation</td>
          <td>$\mathbf{Y} = \max(0, \mathbf{X})$</td>
          <td>$\bar{\mathbf{X}} = \bar{\mathbf{Y}} \odot \mathbb{I}(\mathbf{X} > 0)$</td>
          <td>Branchless conditional assignment</td>
        </tr>
        <tr>
          <td>Sigmoid Activation</td>
          <td>$\mathbf{Y} = \sigma(\mathbf{X})$</td>
          <td>$\bar{\mathbf{X}} = \bar{\mathbf{Y}} \odot \mathbf{Y} \odot (1 - \mathbf{Y})$</td>
          <td>Evaluated directly from saved output $\mathbf{Y}$</td>
        </tr>
        <tr>
          <td>Tanh Activation</td>
          <td>$\mathbf{Y} = \tanh(\mathbf{X})$</td>
          <td>$\bar{\mathbf{X}} = \bar{\mathbf{Y}} \odot (1 - \mathbf{Y}^2)$</td>
          <td>Single-pass FMA: $\bar{Y} \times (1 - Y^2)$</td>
        </tr>
        <tr>
          <td>Softmax (Row-wise)</td>
          <td>$\mathbf{Y}_i = \text{Softmax}(\mathbf{X}_i)$</td>
          <td>$\bar{\mathbf{X}}_i = \mathbf{Y}_i \odot \left( \bar{\mathbf{Y}}_i - \sum_k \bar{\mathbf{Y}}_{i, k} \mathbf{Y}_{i, k} \right)$</td>
          <td>Fused Warp Shuffle reduction pass</td>
        </tr>
      </tbody>
    </table>

    <pre><code>// Matrix Multiplication Backward Node Implementation
class MatMulBackwardNode : public AutogradNode {
public:
    MatMulBackwardNode(Matrix a, Matrix b) : saved_a_(std::move(a)), saved_b_(std::move(b)) {}

    void backward(const Matrix& grad_output) override {
        if (saved_a_.requires_grad()) {
            // dA = grad_output * B^T
            Matrix da = grad_output.matmul_transposed_b(saved_b_);
            saved_a_.accumulate_grad(da);
        }
        if (saved_b_.requires_grad()) {
            // dB = A^T * grad_output
            Matrix db = saved_a_.matmul_transposed_a(grad_output);
            saved_b_.accumulate_grad(db);
        }
    }

private:
    Matrix saved_a_;
    Matrix saved_b_;
};</code></pre>

    <h2>10.4 Activation Checkpointing (Rematerialization)</h2>
    <p>
      In deep models with hundreds of layers (such as deep Transformers), retaining all intermediate activation tensors in GPU memory during the forward pass leads to out-of-memory (OOM) failures. MatrixFlash-Pro implements <strong>Activation Checkpointing</strong>:
    </p>

    <div class="formula-box">
      $$\text{Standard Memory Complexity} = O(L) \quad \longrightarrow \quad O(\sqrt{L}) \quad (\text{with Checkpointing})$$
      <span class="eq-desc">Trading ~25% re-computation overhead for up to 80% reduction in peak VRAM consumption</span>
    </div>

    <p>
      Only boundary checkpoint tensors are stored on the dynamic tape. During the backward pass, sub-graphs between checkpoints are transparently recomputed on-demand in fast local CUDA streams, enabling training of models 4&times; larger on the same physical GPU hardware.
    </p>
  </div>
`;
