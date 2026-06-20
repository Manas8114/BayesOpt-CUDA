# BayesOpt-CUDA Kernel Design and Optimization

This document covers the mathematical formulations and CUDA design choices for the high-performance kernels in **BayesOpt-CUDA**.

---

## 1. Mathematical Formulations

### 1.1 RBF Kernel
The Radial Basis Function (RBF) or Squared Exponential kernel computes the similarity between two points:
$$K(x, y) = \sigma_{signal}^2 \exp\left( -\frac{\|x - y\|_2^2}{2 \ell^2} \right)$$
where:
* $\sigma_{signal}^2$ is the signal variance (output scale).
* $\ell$ is the lengthscale parameter.
* $\|x - y\|_2^2 = \sum_{k=1}^d (x_k - y_k)^2$ is the squared Euclidean distance.

### 1.2 Acquisition Functions
Let $\mu(x)$ and $\sigma(x)$ be the Gaussian Process posterior predictive mean and standard deviation at candidate point $x$. Let $f(x^+)$ be the best observed target value so far (abbreviated as $y_{best}$).

#### Expected Improvement (EI)
$$\text{EI}(x) = \begin{cases} 
(\mu(x) - y_{best} - \xi)\Phi(Z) + \sigma(x)\phi(Z) & \text{if } \sigma(x) > 0 \\
0 & \text{if } \sigma(x) = 0
\end{cases}$$
where:
* $Z = \frac{\mu(x) - y_{best} - \xi}{\sigma(x)}$
* $\Phi(\cdot)$ is the cumulative distribution function (CDF) of the standard normal distribution.
* $\phi(\cdot)$ is the probability density function (PDF) of the standard normal distribution.
* $\xi \ge 0$ is the exploration-exploitation trade-off parameter.

#### Upper Confidence Bound (UCB)
$$\text{UCB}(x) = \mu(x) + \sqrt{\beta}\sigma(x)$$
where $\beta \ge 0$ controls the exploration weight.

#### Probability of Improvement (PI)
$$\text{PI}(x) = \Phi(Z)$$
where $Z = \frac{\mu(x) - y_{best} - \xi}{\sigma(x)}$.

---

## 2. CUDA Optimization Design

### 2.1 Shared Memory Tiling for RBF ($16 \times 16$ Tile Grid)
A naive RBF kernel computes the distance of every test point $x_i$ to every training point $y_j$ by fetching coordinates from global memory repeatedly. For an $n \times m$ grid with dimensionality $d$, this results in $O(n \cdot m \cdot d)$ DRAM reads.

Our **Tiled RBF Kernel** solves this by:
1. Cooperative loading of tiles of size $16 \times 16$ into Shared Memory (SMEM).
2. Tiling both training points and test points.
3. Threads within a block cooperatively fetch $16 \times d$ elements of $X$ and $16 \times d$ elements of $Y$, storing them in high-speed L1 cache/SMEM.
4. Accumulating distances in registers.
5. **Impact**: Reduces global memory bandwidth requirements from $O(n \cdot m \cdot d)$ to $O((n + m) \cdot d)$, avoiding DRAM bottleneck.

```
                  Y (Training points, m x d)
              ┌────────────────────────┐
              │  [Tile Y: 16 x d]      │
              └────────────────────────┘
X (Test pts,  ┌───────────────┐        ┌───────────────┐
  n x d)      │               │        │               │
  ┌──────────┐│  Cooperative  │        │   Compute     │
  │[Tile X:  ]│  SMEM Load    │───────>│   Distance    │
  │ 16 x d   ││               │        │   In-Register │
  └──────────┘└───────────────┘        └───────────────┘
```

### 2.2 FP16 Mixed-Precision SIMD Acceleration
To further double the computational throughput and halve global memory pressure:
1. Coordinates are stored in DRAM in 16-bit float (`__half` or `__half2` types).
2. Global memory accesses are coalesced to fetch 32-bits (two 16-bit floats as `__half2`) per instruction.
3. Subtraction and multiplication are performed in registers using native CUDA SIMD instructions:
   * `__hsub2`: Subtracts two dimensions simultaneously.
   * `__hfma2`: Performs fused multiply-add on two dimensions in a single clock cycle.
4. To prevent underflow and preserve range, squared distance accumulation and the final exponent are computed in standard FP32 (`float` and `expf`).

### 2.3 Fused Acquisition Execution ($O(n)$ Shared Memory)
In standard Bayesian optimization, computing predictive variance requires:
1. Computing a cross-covariance vector $k_s \in \mathbb{R}^n$ between the candidate and all training points.
2. Solving the linear system $L v = k_s$, where $L$ is the Cholesky factor of the training covariance matrix.
3. Evaluating variance: $\sigma^2 = K(x,x) - \|v\|_2^2$.

A naive implementation makes multiple kernel launches, saving intermediate matrices to global memory. Fusing these calculations into one kernel avoids allocating the massive $m \times n$ cross-covariance matrix.

```
Naive Pipeline (DRAM Bound):
[Kernel 1: Covariance] ──> Write K_cross (m x n) to DRAM ──> [Kernel 2: Solve] ──> Write v (m x n) to DRAM ──> [Kernel 3: EI]

Fused Pipeline (Compute Bound):
Compute K_cross row online ──> In-kernel Cholesky Solve (SMEM) ──> Evaluate EI ──> Write EI (m) to DRAM
```

#### Shared-Memory Optimization for Cholesky Solve
* **$O(n^2)$ SMEM Spill**: Initially, the system stored $K^{-1}$ or $L$ entirely in shared memory. For $n \ge 128$, $128^2 \times 4\text{ bytes} \ge 65\text{ KB}$, which exceeds the physical 48KB SM block limit, causing kernel launch failures.
* **$O(n)$ Cholesky SMEM**:
  We resolved this by solving $L v = k_s$ row-by-row on the fly.
  * For each row $i$ of the Cholesky factor, the value $v_i$ is computed as:
    $$v_i = \frac{k_{s, i} - \sum_{j=0}^{i-1} L_{i,j} v_j}{L_{i,i}}$$
  * Because $v$ is a vector of size $n$, we only need to store $v$ in shared memory, requiring only $O(n)$ floats.
  * The row elements of $L$ are streamed from global memory since they are accessed sequentially, maximizing hardware caching.
  * This allows the fused acquisition kernel to scale to thousands of training points without exhausting shared memory.
