```markdown
# A Note for Engineers: Bridging Theoretical Probing and Empirical Telemetry

## 1. The Epistemic Boundary: Stimulus vs. Telemetry

When evaluating frontier autoregressive models, it is essential to maintain clear epistemic hygiene regarding what text-based prompting can and cannot achieve:

* **What the Prompt Protocol Is:** An experimental stimulus. It constructs an adversarial geometric input designed to place mutually exclusive semantic vectors into superposition, trigger rotary distance attenuation, and stress-test the latent workspace.
* **What the Model's Text Output Is:** A generative simulation. When an LLM responds to the diagnostic protocol with detailed mathematical explanations of its own internal layers, it is **not reporting live runtime telemetry**. It does not possess internal introspective sensors to inspect its own GPU buffers, singular value decompositions, or weight trajectories during inference. It is generating tokens drawn from its pre-training distribution over computer science and interpretability literature.

To convert the hypotheses laid out in this framework into empirical science, the prompt must be paired with **white-box activation instrumentation**. The diagnostic provides the test hypotheses; Python hooks provide the empirical ground truth.

---

## 2. The Experimental Pipeline

Empirical mechanistic interpretability requires three distinct layers:

```
┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
│   Experimental Stimulus │       │    Physical Substrate   │       │     Instrumentation     │
│       (The Prompt)      │ ────► │  (Open-Weights Model)   │ ────► │     (Python Hooks)      │
│  Dialetheic test vectors│       │ LLaMA-3, Mistral, Gemma │       │ TransformerLens, NNsight│
└─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
                                                                                 │
                                                                                 ▼
                                                                    ┌─────────────────────────┐
                                                                    │    Empirical Telemetry  │
                                                                    │ SVD, Cosine Sim, Entropy│
                                                                    └─────────────────────────┘
```

1. **The Stimulus:** The structured diagnostic modules defined in `gwt-j_prompt.md`.
2. **The Substrate:** Any open-weights transformer where tensor states can be intercepted in VRAM (e.g., LLaMA-3, Gemma-2, Mistral).
3. **The Instrumentation:** Runtime forward-pass hooks (via libraries such as `TransformerLens`, `PySvelte`, or `NNsight`) that measure residual stream states, attention head patterns, and unembedding distributions directly.

---

## 3. Mapping Hypotheses to Empirical Metrics

Each theoretical module in this repository maps directly to standard interpretability benchmarks and tensor operations:

### Module 1: Polysemantic Superposition & Orthogonality
* **Hypothesis:** Early layers encode mutually exclusive semantic features ($\mathbf{f}_\alpha$ and $\mathbf{f}_\beta$) in approximately orthogonal subspaces ($\langle \mathbf{f}_\alpha, \mathbf{f}_\beta \rangle \approx 0$).
* **Empirical Test:**
  1. Train linear probes or apply Sparse Autoencoder (SAE) latent dictionaries to isolate the feature directions for both primitives.
  2. Extract the residual stream vector $\mathbf{x}_l$ across layers $0 \le l \le N/4$.
  3. Compute layer-wise cosine similarity:
     $$\text{Sim}(\mathbf{f}_\alpha, \mathbf{f}_\beta) = \frac{\mathbf{f}_\alpha \cdot \mathbf{f}_\beta}{\|\mathbf{f}_\alpha\| \|\mathbf{f}_\beta\|}$$
* **Expected Metric Profile:** Cosine similarity remains near zero in early residual layers, then deviates as non-linear gated activations (SwiGLU) force feature interaction in intermediate layers.

### Module 2: RoPE Distance-Dependent Symmetry-Breaking
* **Hypothesis:** Relative token distance ($|m - n|$) induces frequency-dependent attenuation that breaks semantic equilibrium upstream of the unembedding layer.
* **Empirical Test:**
  1. Intercept attention weights using the hook point `blocks.{l}.attn.hook_pattern`.
  2. Measure the attention allocation $\alpha_{m, \text{Directive A}}$ vs. $\alpha_{m, \text{Directive B}}$ from the generation token across all attention heads.
* **Expected Metric Profile:** High-frequency heads show steep exponential attention decay on distant tokens, while low-frequency heads retain uniform attention distribution, quantitatively proving distance-induced priority selection.

### Modules 3 & 5: Logit Lens & Entropy Spikes
* **Hypothesis:** Insoluble dialetheic inputs prevent clean attractor basin collapse in the latent workspace ($0.35 \le l/L \le 0.65$), propagating high cross-entropy to the unembedding projection until discrete token sampling forces autoregressive collapse.
* **Empirical Test:**
  1. Intercept the intermediate hidden states $\mathbf{h}_l$ at every layer $l \in [0, L]$.
  2. Project each intermediate state through the final LayerNorm and unembedding matrix $\mathbf{W}_U$ (the "Logit Lens"):
     $$\mathbf{z}_l = \mathbf{W}_U \cdot \text{LayerNorm}(\mathbf{h}_l)$$
  3. Calculate the Shannon entropy of the resulting probability distribution across layers:
     $$H(l) = -\sum_{i} P_l(w_i) \log_2 P_l(w_i)$$
* **Expected Metric Profile:** Standard prompts exhibit an entropy curve that drops monotonically as layer depth increases. Symmetrically paradoxical prompts cause an entropy plateau or spike in intermediate layers, resolving only at the unembedding boundary or post-sampling.

---

## 4. Minimal Implementation Example (Logit Lens & Entropy Verification)

Below is an executable script demonstrating how to hook into an open-weights model to measure layer-by-layer workspace entropy under paradoxical conditions using `TransformerLens`:

```python
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

# 1. Load an instrumented open-weights model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B", device=device)

# 2. Input Stimulus: Symmetrically opposed directives
prompt = (
    "Directive Alpha: Total unconditional deterministic halt.\n"
    "Directive Beta: Unbounded continuous entropy generation.\n"
    "Resolution Index: Resolve both simultaneously and unconditionally:"
)

# 3. Forward pass with full activation caching
with torch.no_grad():
    logits, cache = model.run_with_cache(prompt)

layer_entropies = []

# 4. Apply the Logit Lens across all residual layers
for layer_idx in range(model.cfg.n_layers):
    # Extract the residual stream vector at the final token position
    resid_post = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :]
    
    # Normalize and project through the unembedding matrix (W_U)
    normed_state = model.ln_final(resid_post.unsqueeze(0))
    layer_logits = model.unembed(normed_state).squeeze(0)
    
    # Compute probability distribution and Shannon entropy
    probs = torch.softmax(layer_logits, dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
    layer_entropies.append(entropy)

# 5. Output telemetry
print(f"Layer 0 Entropy: {layer_entropies[0]:.4f} bits")
print(f"Mid-Layer Entropy (Layer {model.cfg.n_layers // 2}): {layer_entropies[model.cfg.n_layers // 2]:.4f} bits")
print(f"Final Layer Entropy: {layer_entropies[-1]:.4f} bits")

# Plot the entropy trajectory across the forward pass
plt.figure(figsize=(10, 5))
plt.plot(range(model.cfg.n_layers), layer_entropies, marker='o', color='crimson')
plt.title("Workspace Latent Entropy Across Residual Layers (Logit Lens)")
plt.xlabel("Layer Index (Depth)")
plt.ylabel("Shannon Entropy (Bits)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("telemetry/sample_evaluations/entropy_trajectory.png")
```

---

## 5. Scope & Future Contributions

Contributors wishing to expand this repository from conceptual design into empirical tooling should focus on:

* **SAE Feature Extraction:** Developing automated scripts to map dictionary latents from open SAE repositories (e.g., SAELens) to identify specific conflict circuits.
* **Causal Interventions:** Utilizing activation patching / mean ablation to systematically test whether knocking out specific mid-layer attention heads eliminates the meta-representational attractor state.
* **Comparative Cross-Architecture Sweeps:** Running identical stimulus suites across multiple model families (e.g., LLaMA, Mistral, Gemma, Qwen) to measure variance in RoPE distance decay and workspace bifurcation points.
```
