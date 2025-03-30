import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def generate_single_beat(fs=500):
    # Time vector for a single beat
    t = np.linspace(0, 1, fs)

    # Generate a synthetic PQRST complex
    p_wave = 0.1 * np.sin(2 * np.pi * 5 * t) * np.exp(-30 * (t - 0.1)**2)
    q_wave = -0.15 * np.exp(-1000 * (t - 0.18)**2)
    r_wave = 1.0 * np.exp(-1000 * (t - 0.2)**2)
    s_wave = -0.25 * np.exp(-1000 * (t - 0.22)**2)
    t_wave = 0.3 * np.sin(2 * np.pi * 1 * t) * np.exp(-20 * (t - 0.35)**2)

    beat = p_wave + q_wave + r_wave + s_wave + t_wave
    return beat

def assemble_ecg(rhythm_type="Sinus", length_sec=10, bpm=75, fs=500):
    num_beats = int(length_sec * bpm / 60)
    beat = generate_single_beat(fs)
    silence = np.zeros(int(fs * (60 / bpm) - len(beat)))
    ecg = []

    for i in range(num_beats):
        if rhythm_type == "AV Block Type 2" and i % 4 == 3:
            # Simulate dropped beat
            ecg.extend(np.zeros(len(beat) + len(silence)))
        elif rhythm_type == "Atrial Fibrillation":
            # Irregular RR intervals
            irr_gap = np.random.randint(int(0.6 * fs), int(1.2 * fs))
            gap = np.zeros(irr_gap)
            ecg.extend(beat.tolist() + gap.tolist())
        elif rhythm_type == "VTach":
            # Sharp sawtooth wave for VTach
            vt_wave = np.tile([1, -1], int(fs / bpm))
            ecg.extend(vt_wave[:len(beat)])
        else:
            ecg.extend(beat.tolist() + silence.tolist())

    return np.array(ecg)

# Streamlit UI
st.set_page_config(page_title="Telemetry Trainer Pro", layout="wide")
st.title("🚑 Telemetry Trainer Pro")
st.caption("Realistic ECG simulator for telemetry tech training")

rhythm_type = st.sidebar.selectbox("Rhythm Type", ["Sinus", "Atrial Fibrillation", "VTach", "AV Block Type 2"])
bpm = st.sidebar.slider("Heart Rate (bpm)", 40, 180, 75)
duration = st.sidebar.slider("Simulation Duration (seconds)", 5, 30, 10)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.05)

# Generate waveform
fs = 500
ecg = assemble_ecg(rhythm_type, duration, bpm, fs)
time = np.linspace(0, duration, len(ecg))

# Add noise
noise = np.random.normal(0, noise_level, len(ecg))
ecg_noisy = ecg + noise

# Plot
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(time, ecg_noisy, lw=1)
ax.set_title(f"Simulated ECG: {rhythm_type}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage")
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.info("Use the sidebar to simulate different rhythms and noise levels.")
