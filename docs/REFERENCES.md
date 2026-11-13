# Research references (checked 2026-11-13)

1. Takehi, Teoh, Zhang & Khattab (MIT CSAIL). “Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights,” November 2, 2026. Research blog, not peer-reviewed proceedings. https://rikiyat.github.io/blog/weight-space-programming/
2. Original experiment logs, Experiment 8: https://github.com/RikiyaT/lm_understand-logs
3. Official model card and weights, Qwen3.5-0.8B (pin the revision): https://huggingface.co/Qwen/Qwen3.5-0.8B
4. Hu et al. “Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations,” arXiv:2611.09382 (preprint). https://arxiv.org/abs/2611.09382
5. Math-Verify evaluation package: https://github.com/huggingface/Math-Verify
6. Independent replication with different reported and reproduced scores; inspect exact protocols and versions: https://github.com/openwsp/openwsp

Research motivation: a stronger agent should use mathematical structure to reduce experimentation, not just arrive at a patch after many trials. No prior result is treated as confirmed unless exact model, decoding, sampling, scorer and budget are matched.