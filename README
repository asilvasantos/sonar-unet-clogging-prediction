# 🌊 Automated Cyber-Physical Decision Support System (DSS) for Trash Rack Blockage Monitoring Using Deep Learning and Modern CHIRP Sonar

This repository contains the source code, methodology, and implementation framework for an end-to-end automated cyber-physical **Decision Support System (DSS)** designed to monitor obstructions and forecast predictive maintenance windows in run-of-river Hydroelectric Power Plant (HPP) trash racks.

By combining modern **CHIRP (Compressed High-Intensity Radiated Pulse)** acoustic imagery, deep semantic segmentation via **U-Net**, and a seasonal hybrid predictive model, this framework eliminates the need for destructive digital spatial filtering, preserving high-frequency deterministic textures to ensure high-accuracy operational diagnostics.

---

## 📌 Project Overview & Core Proposal

In run-of-river hydroelectric systems, the accumulation of underwater debris (logs, branches, and sediment blocks) on trash racks causes head loss, structural stress, and significant active power degradation. Traditional inspection methods rely heavily on periodic scheduled outages or subjective human operator analysis of raw sonar scans.

**This project proposes a fully integrated, cyber-physical pipeline that:**
1. **Leverages Modern Acoustic Hardware:** Utilizes raw matrices from modern CHIRP sonars which mitigate background noise natively at the hardware/firmware level, delivering high-contrast, sharp geometric boundaries.
2. **Automates Image Interpretation:** Employs a customized 6-class deep learning architecture (`UNet_6Classes`) to perform pixel-by-pixel semantic segmentation directly on raw intensity signals.
3. **Translates Vision into Hydroenergetic Metrics:** Isolates a user-defined **Region of Interest (ROI)** over the generator unit gates, calculates the Instantaneous Blockage Rate ($T_{obs}$), and maps it to a **Dynamic Debris Rate ($V_{obs}$)** calibrated according to hydrological seasonality (dry, regular flood, and extreme/sudden flash flood events).
4. **Delivers an Actionable UI:** Wraps the entire inference engine in an interactive **Streamlit web application** for real-time risk assessment, customizable class binarization, and predictive maintenance scheduling ($T_{man}$).

---

## 🏗️ Repository Architecture

The project is logically structured into modular components to support open-science reproducibility:

```text
├── data/                  # Standardized samples of acoustic patches and palettized masks (CVAT format)
├── models/                # Structural definitions and layers for the UNet_6Classes architecture
├── scripts/               # Core execution files for the engineering pipeline
│   ├── train_unet.py      # Data pipeline, 85/15 train/val split, and optimization loop
│   └── calculate_blockage.py # Mathematical post-processing, ROI filtering, and maintenance forecasting
├── app.py                 # Interactive Streamlit Web Application interface and core inference logic
└── requirements.txt       # Environment dependencies for rapid virtual environment assembly
