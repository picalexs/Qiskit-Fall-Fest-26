# Hardware experiments

These notebooks hold Alexandru Picioroaga's hardware and noise-mitigation tests. The shared setup is [hardware_setup.ipynb](hardware_setup.ipynb); its defaults preserve the five-SNP example, angles, cost function and FakeManilaV2 simulator snapshot used by the pilots.

For a shared input, call `configure_problem(reads, gammas, betas, evidence=(A, D))` after loading setup. The angles are frozen, locally trained standard-QAOA angles; the optional evidence matrices retain fractional weights and allow pruning. Alternatively supply a `hardware_inputs` dictionary before `%run -i hardware_setup.ipynb`. A backend override must have enough qubits. No training or remote jobs occur during configuration. Warm-start/multi-angle circuits should be compiled directly rather than forced into the standard gamma/beta interface.

[The combined notebook](../../bench/all_together.ipynb) reuses this setup, compares ideal and raw noisy sampling with local search, and rescores all samples against full pairwise and full-read MEC references. It does not rerun the optional mitigation sweeps or execute real-QPU jobs. Install the repository's `requirements.txt` for the combined notebook; standalone optional methods may need their own dependencies (for example M3).

The results below come from saved notebook runs. They are small simulator studies, not real-device results. Read each result within its own notebook: compilation seeds and circuits differ between tests. Lower mean cost is better; higher optimum frequency is better.

| Variant | What changed | Saved result |
|---|---|---|
| [SNP ordering and layout](hardware_placement.ipynb) | Tried alternative logical SNP orders and compared them with automatic layout. | The selected order kept the median two-qubit gate count at 28 and reduced two-qubit depth from 23 to 21. No clear noisy-quality gain was established. |
| [Calibration-aware layout](hardware_noise_aware_layout.ipynb) | Chose a physical layout using the backend's gate and readout error estimates. | Two-qubit gates and depth tied automatic layout; total depth and the error proxy were slightly worse. No clear noisy advantage. |
| [Routing](hardware_routing.ipynb) | Compared SABRE with Lookahead on the same circuit and layout. | Both had median 28 two-qubit gates and depth 23; noisy results were close. |
| [Cost-gate order](hardware_gate_order.ipynb) | Compared original, reverse and largest-coupling-first RZZ orders. | Original used fewer two-qubit gates; magnitude-first had lower two-qubit depth. No clear noisy-quality winner. |
| [Commuting RZZ schedule](hardware_commuting_schedule.ipynb) | Grouped non-overlapping interactions into rounds before compilation. | Depth fell 57→52, but two-qubit gates rose 28→36. Mean cost and optimum frequency worsened slightly in the pilot. |
| [Local readout correction](hardware_readout.ipynb) | Calibrated each measured bit and corrected the mean cost. | The corrected mean cost was closer to the simulator's pre-readout reference in all five repeats, including an equal-total-shot comparison. It corrects the estimate, not individual bitstrings. |
| [TREX-style readout correction](hardware_trex.ipynb) | Randomized measurement flips and corrected pair correlations. | TREX-style and local correction both moved the mean toward the reference; TREX showed no clear advantage over local correction. |
| [M3 readout correction](hardware_m3_readout.ipynb) | Compared matrix-free M3 with local correction and raw sampling. | The saved table reports 8.761 ± 0.111 for both corrected methods, versus 9.381 ± 0.093 raw; M3 did not beat local correction in this test. |
| [Dynamical decoupling](hardware_dynamical_decoupling.ipynb) | Added X–X pulses during scheduled idle gaps. | The pulses did not produce a clear change in mean cost or optimum frequency in this simulator run. |
| [Pauli twirling](hardware_pauli_twirling.ipynb) | Compared the baseline with an equal-shot ensemble of twirled circuits. | Mean cost and optimum frequency were close to baseline; twirling added single-qubit gates and increased median depth from 57 to 71. |
| [Zero-noise extrapolation](hardware_zne.ipynb) | Folded two-qubit gates and extrapolated mean cost from noise factors 1 and 3. | Equal-budget mean cost moved from 9.63 raw to 9.15 after extrapolation, closer to the ideal 7.80. This was only three repeats with two noise factors. |

Two older notebook sections mention an archived line-specific commuting-block router and synthetic edge-deletion tests. Their archive files are not present in this repository, so their details are not included here. The current commuting RZZ-order test is linked above.
