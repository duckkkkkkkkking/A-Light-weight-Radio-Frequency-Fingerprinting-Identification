# A-Light-weight-Radio-Frequency-Fingerprinting-Identification
## What is it?

With the rapid advancement of deep learning techniques, numerous neural networks have been successfully developed for radio frequency (RF) fingerprinting identification. In this work, we propose a lightweight yet reliable neural network framework featuring a 9-layer architecture based on the long short-term memory (LSTM) strategy, designed for efficient open-set fingerprinting identification. To simulate real-world conditions, the simulated beacon frames incorporate random modulation, power amplifier nonlinearity, multi-path fading, radio impairments, and additive noise. Compared to the well-known ResNet and GoogleNet, which have $144$ and $177$ layers, respectively, the accuracy and efficiency of our LSTM-based neural network are extensively evaluated across a wide parameter space, including transmitter variability ($s$), device number ($N$), frame per transmitter ($\textup{FPT}$), and signal-to-noise ratio (SNR). Our results demonstrate that the LSTM-based neural network consistently achieves an accuracy above $96\%$ when $\textup{SNR}\ge{20}$ and $\textup{FPT}=200$, regardless of the number of transmitters, which reaches up to $1000$ in this study. For lower values of $\textup{FPT}\le{100}$ and $6\le\textup{SNR}\le{15}$, our LSTM-based network still outperforms GoogleNet and performs comparably to ResNet. Notably, a significant training acceleration of up to $116.7$ times is achieved when $N=100$ and $\textup{FPT}=200$, and the inference time falls below $2$ seconds. Additionally, VRAM usage is reduced by a factor of up to $22.4$, and the disk storage requirement for the trained neural network is less than $1$ MB. Experiments conducted on various devices, including a high-performance computing (HPC) node, a personal computer (PC), and smartphones, suggest that the optimal approach depends on the scale of the task: local training and inference are suitable for $N=12$, remote training combined with local inference is more viable for $N=100$, and remote training and inference become the preferred option for $N=1000$.

From more detailed technical information, you can read our article: A Lightweight LSTM-Based Open-Set RF Fingerprinting Identification for Edge Deployment


## Who will need this code?


Researchers around the world are working on improving RFFI technoligy. It is too expensive to collect sample signals from a large amount of transmitters so generated sample signals are often used. However, current open source program for data generation is too slow. Here we propose a light weight neural network for RFFI and a method that allow us to generate L-LTF sample up to 500 times faster(it depends on the number of cores in the processor). We took nonlinear amplification into consideration, making the signals more realistic.


## How to use these code?

The code runs on MATLABR2023b or later version. Unnormal errors happened on MATLABR2024b at late February 2025, so R2023b is suggested. Parallel computing toolbox is also required.
The files need to stay in the same folder.
