# Use L2 SFT cold start before L2 GRPO

The training paper will include a small but explicit L2 mini-system slice in SFT before increasing L2 weight during GRPO. L2 SFT is not expected to solve system-level correctness by itself; it provides module-composition structure and nonzero reward variance so verifier-decomposed GRPO can optimize L2 behavior.
