# Slide Deck Outline & Speaker Notes (15 Slides)

**Topic:** Beyond Self-Attention in Audio AI: Masked Spectrograms, Mamba & xLSTM  
**Primary Paper:** Yadav, S., Theodoridis, S., & Tan, Z.-H. (2025). *An overview of neural architectures for self-supervised audio representation learning from masked spectrograms* ([arXiv:2509.18691](https://arxiv.org/abs/2509.18691))  
**Synthesized Surveys:** Liu et al. (2022), Ericsson et al. (2022), Triantafyllopoulos et al. (2025), Su et al. (2025), Mousavi et al. (2025)

---

## Slide-by-Slide Plan & Speaking Script

1. **Title Slide — Beyond Self-Attention in Audio AI**
   - *Key Stat Cards:* 4.8M SSAM-Tiny vs. 85.7M SSAST-Base; 10 HEAR Tasks; O(N) vs. O(N²).
   - *Notes:* Introduce the core question—how self-supervised audio models learn from masked spectrograms, and why linear-time Mamba and xLSTM models challenge Transformer dominance.

2. **Why Sound Breaks Standard Deep Learning Assumptions**
   - *Takeaway:* General audio demands both local transient precision and global harmonic context, yet Transformers pay O(N²) attention cost.

3. **Multi-Survey Landscape: Where Masked Spectrograms Fit** (`01_audio_ssl_survey_taxonomy.png`)
   - *Takeaway:* Connects Liu et al. (2022) contrastive/speech SSL, Triantafyllopoulos et al. (2025) Tier-1 vs. Tier-2 Audio-LLMs, Mousavi et al. (2025) discrete audio tokens, and Yadav et al. (2025) MSM backbones.

4. **Masked Spectrogram Modeling: Sound as a 2D Puzzle**
   - *Takeaway:* Non-overlapping $t \times f$ patches, 50–80% masking (due to high spectrogram redundancy vs. BERT's 15%), grouped masking in Dasheng (2024), and masked-patch MSE loss.

5. **The Hidden Architectural Catch: MAE-Style vs. SSAST-Style** (`02_ssast_vs_mae_architecture.png`)
   - *Takeaway:* MAE's token-dropping shortcut only works for unordered-set attention (SDPA); SSAST's in-place `[MASK]` tokens preserve sequence order for a fair Transformer vs. Mamba vs. xLSTM test.

6. **Under the Hood: Transformer vs. Mamba vs. xLSTM** (`03_transformers_vs_mamba_vs_xlstm.png`)
   - *Takeaway:* Side-by-side comparison of complexity, KV cache vs. fixed state, sequence view, and audio strengths/weaknesses.

7. **How Mamba and xLSTM Fix Classical Recurrence**
   - *Takeaway:* Intuition without heavy math—Mamba's selective $\Delta$ behaves as an adaptive focus/forget knob with SRAM parallel scan; xLSTM's mLSTM replaces scalar memory with $d \times d$ outer-product matrix memory and exponential gates.

8. **Experimental Setup: Fair, Depth-Matched, and Reproducible**
   - *Takeaway:* AudioSet-2M pretraining (2s crops, $4 \times 16$ patches, 50% mask, $l=12$ depth-matched blocks, $E_f=3$), frozen 10-task HEAR MLP evaluation, and min-max normalized score $s(m) \in [0, 100]$.

9. **Headline Results: Mamba & xLSTM Outperform Comparable Transformers** (`04_hear_benchmark_overall_scores.png`)
   - *Takeaway:* `SSAM-Tiny` (4.8M, `75.0`) beats `SSAST-Base` (85.7M, `68.2`); `SSAM-Base` (`87.9`) beats `AudioMAE`, `MSM-MAE-208`, and `BEATs-iter3`. Only local-global `MWMAE` scores higher.

10. **Different Architectures, Different Ears: 10-Task Breakdown** (`05_task_domain_heatmap.png`)
    - *Takeaway:* `SSAM` leads Speech & General Audio (`CD`, `VL`, `SC-5h`, `E50`); `AxLSTM` leads Music & Pitch (`NS-5h` `71.4`, `Mri-T` `97.5`); speech-only `HuBERT` collapses on music & tagging.

11. **Ablations: Resolution Sensitivity & Sequence-Length Scaling** (`06_ablation_patch_and_duration.png`)
    - *Takeaway:* Coarsening time ($t=4 \rightarrow 8$) hits Mamba hard (−17.6 pts); finer frequency ($f=8$) boosts xLSTM (+2.1 pts); 10s clips degrade SSAST (`46.4`) and trigger OOM in AxLSTM.

12. **Parameter Efficiency Frontier** (`07_efficiency_vs_accuracy_tradeoff.png`)
    - *Takeaway:* Log-scale parameter curves show Mamba & xLSTM shifting the accuracy-per-parameter frontier sharply up and left.

13. **Cross-Survey Synthesis: Pretraining Targets Matter Too**
    - *Takeaway:* Continuous patch MSE (`Dasheng`, `SSAM`) vs. hybrid InfoNCE (`SSAST`) vs. discrete codebook tokens (`BEATs`)—and the opportunity for Tier-2 Audio-LLMs.

14. **Critical Reflections ("My Two Cents")**
    - *Takeaway:* (1) Frozen HEAR probing vs. fine-tuning bias; (2) MAE's 4× pretraining compute lock-in; (3) Unidirectional beat bidirectional; (4) Hybrid local-global SSM + attention is the endgame.

15. **Practitioner Decision Guide & Key References**
    - *Takeaway:* Clear selection matrix by compute/domain constraints + full multi-survey bibliography.
