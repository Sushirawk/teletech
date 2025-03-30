# app.py (or your preferred filename)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Rhythm generator functions (Keep these exactly the same as before) ---
def generate_sinus(length, hr=75, noise_level=0.02):
    """Generates a normal sinus rhythm ECG signal."""
    t = np.linspace(0, length, int(length * 500)) # Ensure integer length for linspace
    beat_interval = 60 / hr
    ecg = np.zeros_like(t)
    # Simple PQRST approximation
    p_duration = 0.08
    qrs_duration = 0.10
    t_duration = 0.18
    pr_interval = 0.16
    qt_interval = 0.40

    for beat_start_time in np.arange(0, length, beat_interval):
        # P wave
        p_start = beat_start_time
        p_end = p_start + p_duration
        p_indices = (t >= p_start) & (t < p_end)
        p_time = t[p_indices] - p_start
        ecg[p_indices] += 0.1 * np.sin(np.pi / p_duration * p_time)**2 # Smoother P

        # QRS complex
        qrs_start = beat_start_time + pr_interval - qrs_duration / 2 # Center QRS
        qrs_end = qrs_start + qrs_duration
        qrs_indices = (t >= qrs_start) & (t < qrs_end)
        qrs_time = t[qrs_indices] - qrs_start
        # Simple QRS shape (adjust amplitude/shape as needed)
        ecg[qrs_indices] += 1.0 * (1 - 30 * (qrs_time - qrs_duration/2)**2) * np.exp(-15 * (qrs_time - qrs_duration/2)**2)


        # T wave
        t_start = beat_start_time + pr_interval + qrs_duration / 2 + 0.05 # Start T after QRS
        t_end = t_start + t_duration
        # Ensure T wave ends before the next P wave theoretically starts
        t_end = min(t_end, beat_start_time + beat_interval - 0.05)
        t_indices = (t >= t_start) & (t < t_end)
        t_time = t[t_indices] - t_start
        if len(t_time) > 0: # Ensure t_time is not empty
             ecg[t_indices] += 0.3 * np.sin(np.pi / (t_end - t_start) * t_time)**2 # Smoother T

    ecg += np.random.normal(0, noise_level, len(t))
    return t, ecg

def generate_afib(length, avg_hr=110, noise_level=0.08):
    """Generates an atrial fibrillation ECG signal with irregular R-R intervals."""
    t = np.linspace(0, length, int(length * 500))
    ecg = np.zeros_like(t)

    # Base fibrillatory waves (more continuous noise)
    ecg += np.random.normal(0, noise_level * 0.8, len(t)) # Base noise
    # Add some slightly larger, slower oscillations for f-waves
    fibrillation = 0.05 * np.sin(2 * np.pi * 7 * t + np.random.randn()) + \
                   0.04 * np.sin(2 * np.pi * 11 * t + np.random.randn())
    ecg += fibrillation


    # Irregular QRS complexes
    current_time = 0
    while current_time < length:
        # Highly variable R-R interval around the average HR
        rr_interval = np.random.normal(60 / avg_hr, 0.15) # Mean, Std Dev
        rr_interval = max(0.3, rr_interval) # Prevent excessively short intervals
        current_time += rr_interval

        if current_time > length:
            break

        # Add QRS complex at current_time
        qrs_duration = 0.10
        qrs_start = current_time - qrs_duration / 2
        qrs_end = current_time + qrs_duration / 2
        qrs_indices = (t >= qrs_start) & (t < qrs_end)
        qrs_time = t[qrs_indices] - qrs_start
        if len(qrs_time) > 0:
            # Simple QRS shape (can be more varied in real Afib)
             ecg[qrs_indices] += 0.8 * (1 - 30 * (qrs_time - qrs_duration/2)**2) * np.exp(-15 * (qrs_time - qrs_duration/2)**2)


    ecg += np.random.normal(0, noise_level, len(t)) # Add overall noise
    return t, ecg


def generate_vtach(length, hr=150, noise_level=0.05):
    """Generates a ventricular tachycardia ECG signal."""
    t = np.linspace(0, length, int(length * 500))
    beat_interval = 60 / hr
    # More sinusoidal VTach shape
    ecg = 0.6 * np.sin(2 * np.pi * (1 / beat_interval) * t + np.pi/4) # Amplitude 0.6
    # Add some harmonics or shape variation if desired
    ecg += 0.1 * np.sin(2 * np.pi * 2 * (1 / beat_interval) * t)

    ecg += np.random.normal(0, noise_level, len(t))
    return t, ecg

def generate_block(length, block_type='2nd Degree Mobitz I', hr=70, noise_level=0.02):
    """Generates an AV block ECG signal."""
    t = np.linspace(0, length, int(length * 500))
    ecg = np.zeros_like(t)
    p_interval = 60 / hr # Atrial rate (P waves)
    ventricular_hr = hr # Ventricular rate starts same as atrial unless blocked

    if block_type == '3rd Degree':
        ventricular_hr = 40 # Independent, slow ventricular escape rhythm
        # P waves occur regularly at 'hr'
        for p_time in np.arange(0, length, p_interval):
             # P wave
            p_duration = 0.08
            p_start = p_time
            p_end = p_start + p_duration
            p_indices = (t >= p_start) & (t < p_end)
            p_time_local = t[p_indices] - p_start
            ecg[p_indices] += 0.1 * np.sin(np.pi / p_duration * p_time_local)**2

        # Independent QRS complexes occur regularly at 'ventricular_hr'
        qrs_interval = 60 / ventricular_hr
        for qrs_beat_time in np.arange(random.uniform(0, qrs_interval), length, qrs_interval): # Random start phase
            # QRS complex (often wider in 3rd degree)
            qrs_duration = 0.12
            qrs_start = qrs_beat_time - qrs_duration / 2
            qrs_end = qrs_start + qrs_duration
            qrs_indices = (t >= qrs_start) & (t < qrs_end)
            qrs_time_local = t[qrs_indices] - qrs_start
            ecg[qrs_indices] += 0.9 * (1 - 25 * (qrs_time_local - qrs_duration/2)**2) * np.exp(-12 * (qrs_time_local - qrs_duration/2)**2)
            # No T-wave simulation here for simplicity, focus on dissociation

    else: # Mobitz I or II
        beat_num = 0
        pr_prolongation = 0 # For Mobitz I
        pr_base = 0.16

        for beat_time in np.arange(0, length, p_interval):
            beat_num += 1
            conducted = True

            # Determine if beat is dropped
            if block_type == '2nd Degree Mobitz I':
                 # Drop every 4th beat (3:2 conduction shown here, adjust ratio as needed)
                if beat_num % 4 == 0:
                    conducted = False
                    pr_prolongation = 0 # Reset after dropped beat
                else:
                    # Progressively prolong PR before the dropped beat
                    pr_prolongation += 0.04 * ( (beat_num % 4) -1) if (beat_num % 4) != 1 else 0


            elif block_type == '2nd Degree Mobitz II':
                # Drop every 3rd beat (e.g., 2:1 or 3:1 conduction)
                if beat_num % 3 == 0: # Constant PR, intermittent drop
                     conducted = False
                pr_prolongation = 0 # PR is constant in Mobitz II


            # --- Add P wave ---
            p_duration = 0.08
            p_start = beat_time
            p_end = p_start + p_duration
            p_indices = (t >= p_start) & (t < p_end)
            p_time_local = t[p_indices] - p_start
            ecg[p_indices] += 0.1 * np.sin(np.pi / p_duration * p_time_local)**2

            # --- Add QRS and T wave IF conducted ---
            if conducted:
                current_pr = pr_base + pr_prolongation
                qrs_duration = 0.10
                t_duration = 0.18

                # QRS complex
                qrs_start = beat_time + current_pr - qrs_duration / 2
                qrs_end = qrs_start + qrs_duration
                qrs_indices = (t >= qrs_start) & (t < qrs_end)
                qrs_time_local = t[qrs_indices] - qrs_start
                ecg[qrs_indices] += 1.0 * (1 - 30 * (qrs_time_local - qrs_duration/2)**2) * np.exp(-15 * (qrs_time_local - qrs_duration/2)**2)


                # T wave
                t_start = beat_time + current_pr + qrs_duration / 2 + 0.05
                t_end = t_start + t_duration
                t_end = min(t_end, beat_time + p_interval - 0.05) # Ensure T ends before next P
                t_indices = (t >= t_start) & (t < t_end)
                t_time_local = t[t_indices] - t_start
                if len(t_time_local) > 0:
                    ecg[t_indices] += 0.3 * np.sin(np.pi / (t_end - t_start) * t_time_local)**2 # Smoother T


    ecg += np.random.normal(0, noise_level, len(t))
    return t, ecg


# --- Streamlit UI ---
st.set_page_config(layout="wide") # Use wider layout
st.title(" ECG Rhythm Generator")

# --- User Inputs ---
st.sidebar.header("Rhythm Parameters")

rhythm_options = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Ventricular Tachycardia",
    "2nd Degree AV Block (Mobitz I)",
    "2nd Degree AV Block (Mobitz II)",
    "3rd Degree AV Block"
]
selected_rhythm = st.sidebar.selectbox("Select Rhythm:", rhythm_options)

# Duration Slider
duration = st.sidebar.slider("Duration (seconds):", min_value=5, max_value=20, value=10, step=1)

# Heart Rate Slider (conditional)
hr = None
if selected_rhythm in ["Normal Sinus Rhythm", "Ventricular Tachycardia"]:
    hr = st.sidebar.slider("Heart Rate (bpm):", min_value=40, max_value=180, value=75, step=5)
elif selected_rhythm in ["2nd Degree AV Block (Mobitz I)", "2nd Degree AV Block (Mobitz II)", "3rd Degree AV Block"]:
     # For blocks, this is the underlying atrial rate (or escape rate for 3rd degree P waves)
     hr_label = "Atrial Rate (bpm):" if "2nd Degree" in selected_rhythm else "P Wave Rate (bpm):"
     hr = st.sidebar.slider(hr_label, min_value=40, max_value=100, value=70, step=5)
elif selected_rhythm == "Atrial Fibrillation":
     hr = st.sidebar.slider("Average Ventricular Rate (bpm):", min_value=50, max_value=180, value=110, step=5)


# Noise Checkbox
add_noise = st.sidebar.checkbox("Add Noise", value=True)
noise_amount = st.sidebar.slider("Noise Level:", min_value=0.0, max_value=0.2, value=0.03, step=0.01, disabled=not add_noise)

# Generate Button
generate_button = st.sidebar.button("Generate ECG")

# --- Plotting Area ---
st.header("Generated ECG Waveform")
plot_placeholder = st.empty() # Create a placeholder to update the plot

if generate_button:
    # Determine noise level based on checkbox
    actual_noise_level = noise_amount if add_noise else 0.0

    # Generate the selected rhythm
    t, ecg = None, None # Initialize
    if selected_rhythm == "Normal Sinus Rhythm":
        if hr is None: hr = 75 # Default if somehow None
        t, ecg = generate_sinus(duration, hr, actual_noise_level)
    elif selected_rhythm == "Atrial Fibrillation":
        if hr is None: hr = 110 # Default avg rate
        t, ecg = generate_afib(duration, avg_hr=hr, noise_level=actual_noise_level)
    elif selected_rhythm == "Ventricular Tachycardia":
        if hr is None: hr = 150 # Default VT rate
        t, ecg = generate_vtach(duration, hr, actual_noise_level)
    elif selected_rhythm == "2nd Degree AV Block (Mobitz I)":
         if hr is None: hr = 70 # Default atrial rate
         t, ecg = generate_block(duration, '2nd Degree Mobitz I', hr, actual_noise_level)
    elif selected_rhythm == "2nd Degree AV Block (Mobitz II)":
        if hr is None: hr = 70
        t, ecg = generate_block(duration, '2nd Degree Mobitz II', hr, actual_noise_level)
    elif selected_rhythm == "3rd Degree AV Block":
        if hr is None: hr = 70 # P wave rate default
        t, ecg = generate_block(duration, '3rd Degree', hr, actual_noise_level)

    # --- Plotting Logic ---
    if t is not None and ecg is not None:
        fig, ax = plt.subplots(figsize=(15, 4)) # Adjusted figsize for wider layout
        ax.plot(t, ecg, linewidth=1.0, color='blue') # Thinner line, specific color

        # Customize plot appearance
        title_hr_part = f"HR: {hr} bpm" if hr is not None else "Avg HR: {hr} bpm" if selected_rhythm == "Atrial Fibrillation" else "P Rate: {hr} bpm" if selected_rhythm == "3rd Degree AV Block" else f"Atrial Rate: {hr} bpm" if "2nd Degree" in selected_rhythm else ""
        ax.set_title(f"{selected_rhythm} ({duration}s) {title_hr_part}", fontsize=14)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)") # More standard label
        ax.grid(True, linestyle=':', linewidth=0.5, color='gray') # Lighter grid

        # Set reasonable Y-axis limits (adjust as needed based on signal amplitudes)
        min_val = np.min(ecg) - 0.2
        max_val = np.max(ecg) + 0.2
        ax.set_ylim(min_val, max_val)

        # Set X-axis limits
        ax.set_xlim(0, duration)

        # Improve layout
        plt.tight_layout()

        # Display the plot in Streamlit
        plot_placeholder.pyplot(fig)
    else:
        plot_placeholder.warning("Could not generate ECG. Please check parameters.")

else:
    # Show a message initially or when parameters change before clicking Generate
    plot_placeholder.info("Adjust parameters in the sidebar and click 'Generate ECG'")
