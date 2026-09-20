# Matched Filtering, ISI and Eye Diagrams

## Experiment 8 - Matched Filtering, ISI and Eye Diagrams

### Objectives

- Demonstrate matched-filter reception.
- Analyze BPSK transmission through an RRC pulse-shaped channel.
- Study the effect of AWGN noise.
- Analyze eye diagrams and timing sensitivity.
- Study the effect of SNR and timing offset.
- Introduce a multipath ISI channel.

### Implementation

The experiment transmits BPSK symbols using Root Raised Cosine (RRC) pulse shaping.

AWGN noise is added to the transmitted signal, and a matched RRC filter is applied at the receiver.

The total transmitter and receiver filter delay is calculated and compensated before symbol detection.

The experiment also investigates:

- Different SNR values
- Eye height versus SNR
- BER versus timing offset
- Multipath ISI
- BER versus SNR

### Required Visualizations

- Transmit and matched-filter output
- Eye diagram after matched filtering
- Eye height versus SNR
- BER versus timing offset
- Eye diagram with multipath ISI
- BER versus SNR

### Mandatory Validation

The total cascade delay of the transmitter and receiver RRC filters is calculated.

Symbol detection is performed only after removing the total cascade delay.

The detected symbols are compared with the transmitted symbols to verify correct delay compensation and detection.

### Observations

Increasing SNR generally improves the quality of the received signal and increases the eye opening.

Timing offset can increase the BER when sampling is performed away from the optimum sampling instant.

Multipath propagation introduces inter-symbol interference (ISI), which can reduce the eye opening and affect symbol detection.

Matched filtering improves the receiver output by maximizing the signal-to-noise ratio at the sampling instant.

### Files Included

- `matched_filtering_isi_eye_diagrams.py` - Python source code
- `Experiment_8_Matched_Filtering_ISI_Eye_Diagrams.ipynb` - Google Colab notebook with code and outputs
- `Experiment_8_Matched_Filtering_ISI_Eye_Diagrams.pdf` - Experiment report
