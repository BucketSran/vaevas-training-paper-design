# Use EVAS as an audited training accelerator

The training paper will treat Spectre as the reference oracle and EVAS Rust as a parity-certified accelerator for training-time verifier feedback. EVAS-derived rewards may drive SFT/GRPO experiments only under continuous Spectre shadow audit, with zero tolerance for EVAS PASS / Spectre FAIL false positives; Spectre PASS / EVAS FAIL false negatives may be batched into a deferred EVAS repair backlog after fixed audit intervals. This preserves Spectre-grounded credibility while making verifier-in-the-loop training computationally feasible.
