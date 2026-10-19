# LLM4Decompile reading notes and Forge decisions

Reviewed 2026-10-20. Sources are versioned because the early paper abstract,
later paper, and current repository describe different experiments.

## LLM4Decompile (2024)

[Tan et al., arXiv:2403.05286v3](https://arxiv.org/html/2403.05286v3), especially
Sections 3.1, 4.1.1, 4.2, and Appendix D:

- End uses disassembly extracted from binaries as the condition and C as the
  target. Ref instead conditions on Ghidra pseudocode.
- The models initialize from DeepSeek-Coder checkpoints. The conditional
  objective scores source tokens, not the assembly prompt.
- O0–O3 compilation variants augment the data; MinHash/LSH removes duplicates.
- End first uses compilable object-level examples, then executable examples.
  Linked and unlinked functions have materially different context.
- The paper reports 15B training tokens for its large End experiments; its
  1.3B run takes 12 days on eight A100-80GB GPUs. This is not the repository's
  small replication recipe or evidence for scratch training on one A100.
- Evaluation checks reconstruction against test assertions. Passing those
  assertions is evidence on the tested cases, not a universal equivalence proof.

## Decompile-Bench (2025)

[Tan et al., arXiv:2505.12668v2](https://arxiv.org/html/2505.12668v2), especially
Sections 3 and 5.1:

The follow-up describes two million binary/source function pairs selected from
a much larger collection. Its Compile–Trace–Filter pipeline addresses build
configuration, inlining, source ownership, and duplicate mappings. Its reported
fine-tuning experiment uses a tenth of the pairs and initializes from an
existing LLM4Decompile checkpoint. Neither that result nor a synthetic function
pair establishes clean binary/source correspondence for a new corpus.

**Decompile-Bench is not Noelo-Lab's DecBench.** Forge's agreed external scoring
framework remains [DecBench](../evaluation.md). Decompile-Bench is a candidate
training-data/reference source, subject to provenance, licensing, and overlap
review before ingestion. No external training corpus has been downloaded here.

## Decisions for Forge

These are our experimental choices, not claims made by either paper:

1. Train from random weights in Rust/Candle. Start with a small causal decoder;
   do not reuse the papers' fine-tuning learning rates without a scratch-training
   learning-rate experiment.
2. Implement a source-completion-only objective with explicit prompt/EOS/padding
   boundaries. Preserve the input/output token protocol in checkpoints.
3. Use deterministic fixed-batch smoke probes to validate training first. The
   initial four handcrafted pairs demonstrate mechanics only; they are neither
   a compiler-verified corpus nor a quality/generalization benchmark.
4. Keep model weights, AdamW moments, step, config, seed, and dataset hash when
   resuming. Once dropout or sampling exists, checkpoint their state too.
5. Build the real dataset from traced compiler outputs and disassembly, not
   compiler-generated assembly listings with privileged metadata. Split projects
   before generating optimization/architecture variants and audit overlap with
   DecBench. Deduplication must not replace project-level splitting.
6. Research IR-conditioned training alongside the assembly baseline; see
   [input representation experiments](input-representations.md). This changes
   the conditioning representation, not the decision to train our own weights.
7. Run source-semantic checks only on a vetted, separate execution corpus.
   DecBench's external kit stays static-analysis-only under its own contract.

## Initial architecture and limits

The implemented first decoder uses pre-layer normalization, multi-head causal
self-attention, GELU feed-forward blocks, learned absolute positions, and an
untied output projection. It uses a reversible byte codec (256 bytes plus PAD,
BOS, SEP, EOS) instead of a trained BPE tokenizer. There is no claim that this
matches DeepSeek-Coder, Ventris, or the optimal architecture for decompilation.

The approximately 100M byte-level configuration is an infrastructure experiment,
not a useful trained model. F32 and ordinary attention establish a baseline;
mixed precision, optimized attention, data streaming, generation, a learned
tokenizer, and actual DecBench scoring remain separate work.
