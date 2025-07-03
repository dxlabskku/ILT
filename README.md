# Explainable Inverse Lithography Technology:
A Data-Driven Framework for Transparent Mask Fabrication

## 1. Abstract
Advanced lithography is struggling with escalating variable-shaped-beam (VSB) shot counts and rising mask costs as layouts grow more intricate.
Although deep-learning Inverse Lithography Technology (DL-ILT) can synthesize masks in seconds, its black-box nature limits industrial trust.
The paper fuses the state-of-the-art DAMO-ILT generator with a lightweight Adaptive-CAM (AD-CAM) explainer, achieving a mean pixel-level correlation R^2_{sal}=0.79 between saliency maps and correction-needed regions on ICCAD-13 patterns, and R^2_{reg}=0.63 between saliency intensity and post-simulation edge-placement error (EPE) across 1,647 MetalSet tiles. Patterns with stronger AD-CAM activation also demand more VSB shots, linking model attention to manufacturing cost.

## 2. Introduction
•	Shrinking channel lengths to the single-digit-nanometer regime have made OPC and ILT mandatory to preserve PPA, but at the cost of combinatorial growth in candidate mask fragments and shot counts.
•	DL-ILT accelerates mask optimization yet provides no explanation of which layout regions drive cost inflation, hampering fab adoption.
•	Contributions:
1.	First CAM-based interpretation framework for DL-ILT.
2.	New quantitative metric R^2_{sal} to score explainability.
3.	Empirical link between saliency, EPE, and shot count.

## 3. Related Work
A. Traditional Mask Correction – Rule-based OPC is fast but inaccurate; model-based OPC (MB-OPC) improves fidelity yet explodes in runtime and shot count below 7 nm nodes.
B. Deep-Learning ILT – GAN-OPC, DAMO-ILT, A2-ILT, Neural-ILT and others cut runtimes from hours to seconds while retaining accuracy by learning a direct layout-to-mask mapping or embedding physics in the network.
C. Explainable AI in Lithography – Prior work (e.g., LithoExp) used Grad-CAM for hotspot classification but no study quantified interpretability for mask synthesis until now.
D. Datasets & Benchmarks – The 2023 LithoBench benchmark unifies 133,496 tiles and five metrics (L2, PVB, EPE, shot count, runtime), enabling head-to-head comparison; this study adopts that platform.

## 4. Method
1 Experimental Setup – Single workstation: NVIDIA RTX 4090 (24 GB) + Intel Xeon Silver 4310.
2 Target Network (DAMO-ILT) – U-Net-like generator: 1 × 7×7 stem → 5 down-sampling conv blocks (32→1 024 ch) → 9 residual blocks → 5 deconv blocks back to 32 ch → 7×7 head producing a binary mask (Fig. 2).
<img width="782" alt="image" src="https://github.com/user-attachments/assets/93daa39f-08cb-4ff9-bcf7-2588ef043527" />

3 CAM Extraction Pipeline –
	1)	Capture activations from every encoder, residual, and decoder block.
	2)	Compute Grad-CAM, Score-CAM, and AD-CAM per block.
	3)	Min–max normalise, resize to 128×128, and analyse layer-wise attention shift from coarse to fine details.
4 Correction-Needed Map (∆) – Pixel-wise Hopkins simulation difference between non-optimised and ground-truth masks serves as the gold-standard region to correct ￼.
5 Quantifying Explainability – Define R^2_{sal} between each CAM map and ∆; higher scores denote better localisation. AD-CAM delivers best mean R^2_{sal}=0.79 across ten source patterns.
6 Adaptive-CAM Implementation – Gradient-free, three-stage process: upsample & mask activations, forward re-inject to compute channel weights, aggregate weighted activations through ReLU to form high-resolution saliency.

## 5. Results
•	Explainability – AD-CAM outperforms Grad-/Score-CAM across ten ICCAD-13 patterns with R^2_{sal}=0.79 (Fig. 5).

<img width="429" alt="image" src="https://github.com/user-attachments/assets/528d8777-00fb-4ac7-ae78-014d444914ec" />

•	Accuracy Correlation – For 1,647 MetalSet tiles, every 0.1 rise in R^2_{sal} yields significant drops in L2, PVB, and EPE; regressions achieve R^2_{reg} 0.63 (Fig. 6).

<img width="297" alt="image" src="https://github.com/user-attachments/assets/233ab482-276d-4cc7-8097-89bb15615b53" />

•	Manufacturing Impact – Binned analysis shows shot count and EPE increase monotonically with mean AD-CAM intensity (p < 0.05) (Fig. 7).

<img width="511" alt="image" src="https://github.com/user-attachments/assets/7bf5414c-030e-4f3b-b40b-be340bd07e3e" />

## 6. Discussion & Conclusions
Visualising DAMO-ILT with AD-CAM aligns model attention with lithographic hotspots, providing:
	1.	Transparency – engineers can verify model focus on known critical geometries;
	2.	Targeted QA – inspection resources concentrate on salient areas;
	3.	Cost Reduction – local OPC/ILT tweaks on CAM-flagged regions trim EPE and shot counts without global redesign.

## 7. Limitation & Future Work
Current analysis is confined to a single CNN-based engine (DAMO-ILT); transformer-style generators (e.g., SwinT-ILT) require architecture-agnostic interpretability methods. The authors propose merging perturbation-based explainers (LIME/SHAP) with adaptive spatial sampling to extend the framework across diverse ILT back-ends.



