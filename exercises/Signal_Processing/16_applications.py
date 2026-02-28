"""
Exercises for Lesson 16: Applications
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, ifft, fftfreq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Audio Effects Chain ===
# Problem: Implement Schroeder reverberator (4 comb + 2 allpass), generate
#          melody with reverb, implement flanger effect.

def exercise_1():
    """Audio effects: Schroeder reverberator, melody, flanger."""
    fs = 44100

    # (a) Schroeder reverberator
    print("(a) Schroeder reverberator:")

    def comb_filter(x, delay_ms, gain, fs_rate):
        """IIR comb filter: y[n] = x[n] + g * y[n-M]"""
        M = int(delay_ms * fs_rate / 1000)
        y = np.zeros(len(x) + M)
        y[:len(x)] = x
        for n in range(M, len(y)):
            y[n] += gain * y[n - M]
        return y[:len(x)]

    def allpass_filter(x, delay_ms, gain, fs_rate):
        """Allpass filter: y[n] = -g*x[n] + x[n-M] + g*y[n-M]"""
        M = int(delay_ms * fs_rate / 1000)
        y = np.zeros(len(x))
        for n in range(len(x)):
            y[n] = -gain * x[n]
            if n >= M:
                y[n] += x[n - M] + gain * y[n - M]
        return y

    def schroeder_reverb(x, fs_rate):
        """Schroeder reverberator: 4 parallel combs + 2 series allpasses."""
        # 4 parallel comb filters
        comb_params = [
            (29.7, 0.742),
            (37.1, 0.733),
            (41.1, 0.715),
            (43.7, 0.697),
        ]
        comb_out = np.zeros(len(x))
        for delay, gain in comb_params:
            comb_out += comb_filter(x, delay, gain, fs_rate)
        comb_out /= len(comb_params)

        # 2 series allpass filters
        ap1 = allpass_filter(comb_out, 5.0, 0.7, fs_rate)
        ap2 = allpass_filter(ap1, 1.7, 0.7, fs_rate)
        return ap2

    # Apply to impulse
    impulse_len = int(fs * 1.0)  # 1 second
    impulse = np.zeros(impulse_len)
    impulse[0] = 1.0
    ir = schroeder_reverb(impulse, fs)

    print(f"    Impulse response length: {len(ir)} samples ({len(ir) / fs:.2f} s)")
    print(f"    RT60 estimate: ~{np.argmax(np.cumsum(ir ** 2) > 0.999 * np.sum(ir ** 2)) / fs:.2f} s")
    print(f"    Comb delays: 29.7, 37.1, 41.1, 43.7 ms")
    print(f"    Allpass delays: 5.0, 1.7 ms, gain=0.7")

    # (b) Generate "Twinkle Twinkle Little Star" melody and apply reverb
    print("\n(b) Melody with reverb:")
    # Note frequencies (C major)
    notes = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
    }
    # Twinkle Twinkle: C C G G A A G, F F E E D D C
    melody_notes = ['C4', 'C4', 'G4', 'G4', 'A4', 'A4', 'G4',
                    'F4', 'F4', 'E4', 'E4', 'D4', 'D4', 'C4']
    note_dur = 0.3  # seconds per note

    melody = np.array([])
    t_note = np.linspace(0, note_dur, int(fs * note_dur), endpoint=False)
    for note in melody_notes:
        freq = notes[note]
        # Simple tone with envelope
        envelope = np.exp(-3 * t_note / note_dur)
        tone = 0.5 * np.sin(2 * np.pi * freq * t_note) * envelope
        melody = np.concatenate([melody, tone])

    # Pad for reverb tail
    melody_padded = np.concatenate([melody, np.zeros(int(fs * 0.5))])
    melody_reverb = schroeder_reverb(melody_padded, fs)

    # Mix dry + wet
    dry_wet = 0.3
    mixed = (1 - dry_wet) * melody_padded + dry_wet * melody_reverb
    mixed /= np.max(np.abs(mixed))

    print(f"    Melody duration: {len(melody) / fs:.2f} s ({len(melody_notes)} notes)")
    print(f"    With reverb tail: {len(mixed) / fs:.2f} s")
    print(f"    Dry/wet mix: {(1 - dry_wet) * 100:.0f}% / {dry_wet * 100:.0f}%")

    # (c) Flanger effect
    print("\n(c) Flanger effect:")

    def flanger(x, fs_rate, rate=0.5, depth_ms=3.0, mix=0.5):
        """Flanger: modulated delay line."""
        depth_samples = depth_ms * fs_rate / 1000
        y = np.zeros(len(x))
        for n in range(len(x)):
            delay = depth_samples * (1 + np.sin(2 * np.pi * rate * n / fs_rate)) / 2
            d_int = int(delay)
            d_frac = delay - d_int
            if n - d_int - 1 >= 0:
                # Linear interpolation for fractional delay
                delayed = (1 - d_frac) * x[n - d_int] + d_frac * x[n - d_int - 1]
                y[n] = (1 - mix) * x[n] + mix * delayed
            else:
                y[n] = x[n]
        return y

    # Apply to a short test signal (440 Hz tone)
    dur = 2.0
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    test_tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    flanged = flanger(test_tone, fs, rate=0.5, depth_ms=3.0, mix=0.5)

    # Show time-varying frequency response
    f_axis, t_axis, Sxx = sig.spectrogram(flanged, fs, nperseg=2048, noverlap=1024)

    print(f"    Flanger rate: 0.5 Hz, depth: 3.0 ms")
    print(f"    Creates sweeping comb-filter effect")
    print(f"    Notch spacing varies with LFO position")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t_ir = np.arange(len(ir)) / fs
    axes[0, 0].plot(t_ir[:int(fs * 0.3)], ir[:int(fs * 0.3)])
    axes[0, 0].set_title('Schroeder Impulse Response')
    axes[0, 0].set_xlabel('Time (s)')

    t_mel = np.arange(len(mixed)) / fs
    axes[0, 1].plot(t_mel, mixed, alpha=0.7)
    axes[0, 1].set_title('Melody with Reverb')
    axes[0, 1].set_xlabel('Time (s)')

    axes[1, 0].pcolormesh(t_axis, f_axis[:100], 10 * np.log10(Sxx[:100] + 1e-10),
                          shading='gouraud', cmap='viridis')
    axes[1, 0].set_title('Flanger Spectrogram')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')

    # Magnitude response at two time points
    n1 = int(0.25 * fs)  # t=0.25s
    n2 = int(1.25 * fs)  # t=1.25s (half period later)
    for n_pt, label in [(n1, 't=0.25s'), (n2, 't=1.25s')]:
        depth_samples = 3.0 * fs / 1000
        delay = depth_samples * (1 + np.sin(2 * np.pi * 0.5 * n_pt / fs)) / 2
        # Transfer function: H(f) = 0.5 + 0.5*exp(-j*2*pi*f*delay/fs)
        freqs = np.linspace(0, fs / 2, 500)
        H = 0.5 + 0.5 * np.exp(-1j * 2 * np.pi * freqs * delay / fs)
        axes[1, 1].plot(freqs, 20 * np.log10(np.abs(H) + 1e-10), label=label)
    axes[1, 1].set_title('Flanger Frequency Response')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].legend()
    axes[1, 1].set_xlim([0, 5000])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex16_1_audio_effects.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex16_1_audio_effects.png")


# === Exercise 2: Pitch Detection Robustness ===
# Problem: Autocorrelation and cepstrum pitch detection, harmonics, noise, missing fundamental.

def exercise_2():
    """Pitch detection: autocorrelation, cepstrum, robustness analysis."""
    fs = 16000
    dur = 0.1  # 100 ms analysis window
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)

    def autocorr_pitch(x, fs_rate, fmin=50, fmax=500):
        """Pitch detection via autocorrelation."""
        r = np.correlate(x, x, mode='full')
        r = r[len(r) // 2:]  # keep positive lags
        # Search range
        lag_min = int(fs_rate / fmax)
        lag_max = int(fs_rate / fmin)
        lag_max = min(lag_max, len(r) - 1)
        if lag_min >= lag_max:
            return 0
        peak_lag = lag_min + np.argmax(r[lag_min:lag_max + 1])
        return fs_rate / peak_lag

    def cepstrum_pitch(x, fs_rate, fmin=50, fmax=500):
        """Pitch detection via real cepstrum."""
        windowed = x * np.hanning(len(x))
        spectrum = np.abs(fft(windowed))
        spectrum[spectrum < 1e-10] = 1e-10
        ceps = np.real(ifft(np.log(spectrum)))
        # Search range in quefrency
        q_min = int(fs_rate / fmax)
        q_max = int(fs_rate / fmin)
        q_max = min(q_max, len(ceps) // 2 - 1)
        if q_min >= q_max:
            return 0
        peak_q = q_min + np.argmax(ceps[q_min:q_max + 1])
        return fs_rate / peak_q

    # (a) Pure tone at 220 Hz
    f0 = 220
    pure = np.sin(2 * np.pi * f0 * t)
    ac_freq = autocorr_pitch(pure, fs)
    cep_freq = cepstrum_pitch(pure, fs)
    print("(a) Pure tone at 220 Hz:")
    print(f"    Autocorrelation: {ac_freq:.1f} Hz (error: {abs(ac_freq - f0):.1f} Hz)")
    print(f"    Cepstrum:        {cep_freq:.1f} Hz (error: {abs(cep_freq - f0):.1f} Hz)")

    # (b) With harmonics
    print("\n(b) With harmonics (1st through 5th):")
    harmonics = np.zeros_like(t)
    for h in range(1, 6):
        harmonics += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    ac_freq = autocorr_pitch(harmonics, fs)
    cep_freq = cepstrum_pitch(harmonics, fs)
    print(f"    Autocorrelation: {ac_freq:.1f} Hz (error: {abs(ac_freq - f0):.1f} Hz)")
    print(f"    Cepstrum:        {cep_freq:.1f} Hz (error: {abs(cep_freq - f0):.1f} Hz)")

    # (c) Noise robustness
    print("\n(c) Pitch detection accuracy vs SNR:")
    snr_values = [20, 10, 5, 0]
    n_trials = 20
    print(f"    {'SNR (dB)':>8s}  {'AC accuracy':>12s}  {'Cep accuracy':>13s}")
    for snr in snr_values:
        ac_correct = 0
        cep_correct = 0
        for _ in range(n_trials):
            signal_power = np.mean(harmonics ** 2)
            noise_power = signal_power / (10 ** (snr / 10))
            noisy = harmonics + np.sqrt(noise_power) * np.random.randn(len(t))
            ac_f = autocorr_pitch(noisy, fs)
            cep_f = cepstrum_pitch(noisy, fs)
            if abs(ac_f - f0) < 5:
                ac_correct += 1
            if abs(cep_f - f0) < 5:
                cep_correct += 1
        print(f"    {snr:8d}  {ac_correct / n_trials * 100:11.0f}%  {cep_correct / n_trials * 100:12.0f}%")

    # (d) Missing fundamental
    print("\n(d) Missing fundamental (harmonics 2,3,4 of 100 Hz):")
    f0_miss = 100
    missing_fund = np.zeros_like(t)
    for h in [2, 3, 4]:
        missing_fund += np.sin(2 * np.pi * f0_miss * h * t)
    ac_freq = autocorr_pitch(missing_fund, fs)
    cep_freq = cepstrum_pitch(missing_fund, fs)
    print(f"    Expected fundamental: {f0_miss} Hz")
    print(f"    Autocorrelation: {ac_freq:.1f} Hz "
          f"({'correct' if abs(ac_freq - f0_miss) < 5 else 'incorrect'})")
    print(f"    Cepstrum:        {cep_freq:.1f} Hz "
          f"({'correct' if abs(cep_freq - f0_miss) < 5 else 'incorrect'})")
    print("    Autocorrelation can detect missing fundamental because the period")
    print("    of the composite waveform still matches 1/f0.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(t * 1000, harmonics)
    axes[0, 0].set_title('Signal with 5 harmonics (f0=220 Hz)')
    axes[0, 0].set_xlabel('Time (ms)')

    # Autocorrelation plot
    r = np.correlate(harmonics, harmonics, mode='full')
    r = r[len(r) // 2:]
    lags = np.arange(len(r)) / fs * 1000
    axes[0, 1].plot(lags[:200], r[:200])
    axes[0, 1].axvline(1000 / f0, color='r', linestyle='--', label=f'T0={1000 / f0:.1f} ms')
    axes[0, 1].set_title('Autocorrelation')
    axes[0, 1].set_xlabel('Lag (ms)')
    axes[0, 1].legend()

    # Missing fundamental
    axes[1, 0].plot(t * 1000, missing_fund)
    axes[1, 0].set_title('Missing fundamental (harmonics 2,3,4 of 100 Hz)')
    axes[1, 0].set_xlabel('Time (ms)')

    # Cepstrum of missing fundamental
    windowed = missing_fund * np.hanning(len(missing_fund))
    spectrum = np.abs(fft(windowed))
    spectrum[spectrum < 1e-10] = 1e-10
    ceps = np.real(ifft(np.log(spectrum)))
    quefrency = np.arange(len(ceps)) / fs * 1000
    axes[1, 1].plot(quefrency[:200], ceps[:200])
    axes[1, 1].axvline(1000 / f0_miss, color='r', linestyle='--', label=f'q0={1000 / f0_miss:.1f} ms')
    axes[1, 1].set_title('Cepstrum (missing fundamental)')
    axes[1, 1].set_xlabel('Quefrency (ms)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('ex16_2_pitch_detection.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex16_2_pitch_detection.png")


# === Exercise 3: Digital Modulation BER ===
# Problem: BPSK, QPSK, 16-QAM modulation/demodulation, BER curves, Gray coding.

def exercise_3():
    """Digital modulation: BPSK, QPSK, 16-QAM with BER analysis."""
    np.random.seed(42)
    n_bits = 100000

    def q_function(x):
        """Q-function: tail probability of standard normal."""
        from scipy.special import erfc
        return 0.5 * erfc(x / np.sqrt(2))

    # (a) Implement modulation schemes
    print("(a) Modulation implementations:")

    def bpsk_mod(bits):
        return 2 * bits - 1  # 0 -> -1, 1 -> +1

    def bpsk_demod(symbols):
        return (symbols.real > 0).astype(int)

    def qpsk_mod(bits):
        # 2 bits per symbol
        bits = bits[:len(bits) - len(bits) % 2]
        I = 2 * bits[0::2] - 1
        Q_ch = 2 * bits[1::2] - 1
        return (I + 1j * Q_ch) / np.sqrt(2)

    def qpsk_demod(symbols):
        I_bits = (symbols.real > 0).astype(int)
        Q_bits = (symbols.imag > 0).astype(int)
        bits = np.empty(2 * len(symbols), dtype=int)
        bits[0::2] = I_bits
        bits[1::2] = Q_bits
        return bits

    def qam16_constellation():
        """16-QAM constellation with natural binary mapping."""
        levels = np.array([-3, -1, 1, 3])
        const = []
        for i in levels:
            for q in levels:
                const.append(i + 1j * q)
        return np.array(const) / np.sqrt(10)  # normalize average power

    def qam16_gray_constellation():
        """16-QAM constellation with Gray coding."""
        # Gray code for 2 bits: 00 -> -3, 01 -> -1, 11 -> 1, 10 -> 3
        gray_levels = {0b00: -3, 0b01: -1, 0b11: 1, 0b10: 3}
        const = []
        mapping = {}
        idx = 0
        for i_bits in [0b00, 0b01, 0b11, 0b10]:
            for q_bits in [0b00, 0b01, 0b11, 0b10]:
                i_val = gray_levels[i_bits]
                q_val = gray_levels[q_bits]
                symbol = (i_val + 1j * q_val) / np.sqrt(10)
                const.append(symbol)
                label = (i_bits << 2) | q_bits
                mapping[idx] = label
                idx += 1
        return np.array(const), mapping

    def qam16_mod(bits, gray=False):
        bits = bits[:len(bits) - len(bits) % 4]
        n_symbols = len(bits) // 4
        if gray:
            const, _ = qam16_gray_constellation()
        else:
            const = qam16_constellation()
        symbols = np.zeros(n_symbols, dtype=complex)
        for k in range(n_symbols):
            idx = 0
            for b in range(4):
                idx = (idx << 1) | bits[k * 4 + b]
            symbols[k] = const[idx]
        return symbols

    def qam16_demod(symbols, gray=False):
        if gray:
            const, mapping = qam16_gray_constellation()
        else:
            const = qam16_constellation()
            mapping = {i: i for i in range(16)}

        bits = np.zeros(len(symbols) * 4, dtype=int)
        for k, s in enumerate(symbols):
            dists = np.abs(s - const)
            idx = np.argmin(dists)
            label = mapping.get(idx, idx)
            for b in range(4):
                bits[k * 4 + 3 - b] = (label >> b) & 1
        return bits

    print("    BPSK: 1 bit/symbol")
    print("    QPSK: 2 bits/symbol")
    print("    16-QAM: 4 bits/symbol")

    # (b) BER simulation over AWGN
    print("\n(b) BER curves over AWGN channel:")
    EbN0_dB = np.arange(0, 21, 2)

    ber_results = {'BPSK': [], 'QPSK': [], '16-QAM': []}

    for ebn0 in EbN0_dB:
        ebn0_linear = 10 ** (ebn0 / 10)
        bits = np.random.randint(0, 2, n_bits)

        # BPSK: 1 bit/symbol, Es = Eb
        s_bpsk = bpsk_mod(bits)
        noise_std = 1 / np.sqrt(2 * ebn0_linear)
        r_bpsk = s_bpsk + noise_std * np.random.randn(len(s_bpsk))
        bits_hat = bpsk_demod(r_bpsk)
        ber_results['BPSK'].append(np.mean(bits[:len(bits_hat)] != bits_hat))

        # QPSK: 2 bits/symbol, Es = 2*Eb
        s_qpsk = qpsk_mod(bits)
        noise_std = 1 / np.sqrt(2 * ebn0_linear)
        noise = noise_std * (np.random.randn(len(s_qpsk)) + 1j * np.random.randn(len(s_qpsk)))
        r_qpsk = s_qpsk + noise
        bits_hat = qpsk_demod(r_qpsk)
        ber_results['QPSK'].append(np.mean(bits[:len(bits_hat)] != bits_hat))

        # 16-QAM: 4 bits/symbol, Es = 10*Eb/4 (average)
        s_qam = qam16_mod(bits)
        noise_std = np.sqrt(10 / (4 * ebn0_linear)) / np.sqrt(2)
        noise = noise_std * (np.random.randn(len(s_qam)) + 1j * np.random.randn(len(s_qam)))
        r_qam = s_qam + noise
        bits_hat = qam16_demod(r_qam)
        ber_results['16-QAM'].append(np.mean(bits[:len(bits_hat)] != bits_hat))

    for scheme, ber in ber_results.items():
        print(f"    {scheme}: BER at Eb/N0=10dB = {ber[5]:.6f}")

    # (c) Theoretical BER comparison
    print("\n(c) Theoretical BER comparison:")
    ebn0_fine = np.linspace(0, 20, 100)
    ebn0_lin = 10 ** (ebn0_fine / 10)
    ber_bpsk_theory = q_function(np.sqrt(2 * ebn0_lin))
    ber_qpsk_theory = q_function(np.sqrt(2 * ebn0_lin))
    ber_16qam_theory = (3 / 8) * q_function(np.sqrt(4 * ebn0_lin / 5))

    print(f"    BPSK theoretical BER at 10dB:  {q_function(np.sqrt(2 * 10)):.6f}")
    print(f"    QPSK theoretical BER at 10dB:  {q_function(np.sqrt(2 * 10)):.6f}")
    print(f"    16-QAM theoretical BER at 10dB: {3 / 8 * q_function(np.sqrt(4 * 10 / 5)):.6f}")

    # (d) Gray coding improvement for 16-QAM
    print("\n(d) Gray coding for 16-QAM:")
    ebn0_test = 12  # dB
    ebn0_lin_test = 10 ** (ebn0_test / 10)
    bits = np.random.randint(0, 2, n_bits)

    # Natural binary
    s_nat = qam16_mod(bits, gray=False)
    noise_std = np.sqrt(10 / (4 * ebn0_lin_test)) / np.sqrt(2)
    noise = noise_std * (np.random.randn(len(s_nat)) + 1j * np.random.randn(len(s_nat)))
    bits_nat = qam16_demod(s_nat + noise, gray=False)
    ber_nat = np.mean(bits[:len(bits_nat)] != bits_nat)

    # Gray coding
    s_gray = qam16_mod(bits, gray=True)
    bits_gray = qam16_demod(s_gray + noise[:len(s_gray)], gray=True)
    ber_gray = np.mean(bits[:len(bits_gray)] != bits_gray)

    print(f"    At Eb/N0 = {ebn0_test} dB:")
    print(f"    Natural binary BER: {ber_nat:.6f}")
    print(f"    Gray coded BER:     {ber_gray:.6f}")
    print(f"    Improvement: {ber_nat / max(ber_gray, 1e-10):.2f}x")

    fig, ax = plt.subplots(figsize=(8, 6))
    for scheme, ber in ber_results.items():
        ber_plot = [max(b, 1e-7) for b in ber]
        ax.semilogy(EbN0_dB, ber_plot, 'o-', label=f'{scheme} (sim)')
    ax.semilogy(ebn0_fine, ber_bpsk_theory, '--', color='C0', alpha=0.5, label='BPSK (theory)')
    ax.semilogy(ebn0_fine, ber_qpsk_theory, '--', color='C1', alpha=0.5, label='QPSK (theory)')
    ax.semilogy(ebn0_fine, ber_16qam_theory, '--', color='C2', alpha=0.5, label='16-QAM (theory)')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('BER')
    ax.set_title('BER vs Eb/N0')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-6, 1])
    plt.tight_layout()
    plt.savefig('ex16_3_modulation_ber.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex16_3_modulation_ber.png")


# === Exercise 4: OFDM System ===
# Problem: OFDM transmitter/receiver, multipath channel, cyclic prefix equalization.

def exercise_4():
    """OFDM system: transmitter, receiver, cyclic prefix, equalization."""
    np.random.seed(42)

    N_sub = 64       # number of subcarriers
    N_cp = 16        # cyclic prefix length
    n_symbols = 100  # OFDM symbols
    SNR_dB = 20

    # Multipath channel
    h = np.array([1.0, 0, 0.5, 0, 0.2])

    # QPSK modulation for simplicity
    def qpsk_symbols(n):
        bits = np.random.randint(0, 2, (n, 2))
        return (2 * bits[:, 0] - 1 + 1j * (2 * bits[:, 1] - 1)) / np.sqrt(2)

    print("(a) OFDM system parameters:")
    print(f"    Subcarriers: {N_sub}")
    print(f"    Cyclic prefix: {N_cp}")
    print(f"    Channel: h = {h}")
    print(f"    Modulation: QPSK")

    # (a)(b) Transmitter
    tx_symbols = np.zeros((n_symbols, N_sub), dtype=complex)
    for i in range(n_symbols):
        tx_symbols[i] = qpsk_symbols(N_sub)

    # OFDM modulation: IFFT + CP
    def ofdm_tx(symbols, n_cp):
        """OFDM transmitter: IFFT + cyclic prefix insertion."""
        time_domain = np.fft.ifft(symbols, axis=1) * np.sqrt(N_sub)
        # Add cyclic prefix
        cp = time_domain[:, -n_cp:]
        return np.hstack([cp, time_domain])

    tx_signal = ofdm_tx(tx_symbols, N_cp)

    # Transmit through channel
    rx_frames = np.zeros_like(tx_signal)
    for i in range(n_symbols):
        frame = tx_signal[i]
        rx_frame = np.convolve(frame, h)[:len(frame)]
        # Add noise
        signal_power = np.mean(np.abs(rx_frame) ** 2)
        noise_power = signal_power / (10 ** (SNR_dB / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(rx_frame)) +
                                              1j * np.random.randn(len(rx_frame)))
        rx_frames[i] = rx_frame + noise

    # (c) OFDM receiver with CP removal and equalization
    def ofdm_rx(rx_frames_in, n_cp, channel):
        """OFDM receiver: CP removal + FFT + one-tap equalization."""
        # Remove CP
        data = rx_frames_in[:, n_cp:]
        # FFT
        freq_domain = np.fft.fft(data, axis=1) / np.sqrt(N_sub)
        # Channel frequency response
        H = np.fft.fft(channel, N_sub)
        # One-tap equalization (ZF)
        equalized = freq_domain / H[np.newaxis, :]
        return freq_domain, equalized

    raw_rx, equalized_rx = ofdm_rx(rx_frames, N_cp, h)

    # BER calculation
    def qpsk_demod_symbols(symbols):
        bits = np.zeros((len(symbols), 2), dtype=int)
        bits[:, 0] = (symbols.real > 0).astype(int)
        bits[:, 1] = (symbols.imag > 0).astype(int)
        return bits

    # Before equalization
    rx_bits_raw = np.array([qpsk_demod_symbols(raw_rx[i]) for i in range(n_symbols)])
    tx_bits = np.array([qpsk_demod_symbols(tx_symbols[i]) for i in range(n_symbols)])

    # After equalization
    rx_bits_eq = np.array([qpsk_demod_symbols(equalized_rx[i]) for i in range(n_symbols)])

    ber_raw = np.mean(tx_bits != rx_bits_raw)
    ber_eq = np.mean(tx_bits != rx_bits_eq)

    print(f"\n(c) Equalization results:")
    print(f"    BER before equalization: {ber_raw:.6f}")
    print(f"    BER after equalization:  {ber_eq:.6f}")
    print(f"    One-tap ZF equalization corrects multipath distortion")

    # (d) Without cyclic prefix - demonstrate ISI
    print("\n(d) Effect of removing cyclic prefix:")
    tx_no_cp = np.fft.ifft(tx_symbols, axis=1) * np.sqrt(N_sub)  # no CP

    # Transmit through channel (symbols bleed into each other)
    rx_no_cp = np.zeros_like(tx_no_cp)
    prev_tail = np.zeros(len(h) - 1, dtype=complex)
    for i in range(n_symbols):
        frame = tx_no_cp[i]
        rx_full = np.convolve(frame, h)
        rx_no_cp[i] = rx_full[:N_sub] + np.pad(prev_tail, (0, N_sub - len(prev_tail)))[:N_sub]
        prev_tail = rx_full[N_sub:]
        # Add noise
        signal_power = np.mean(np.abs(rx_no_cp[i]) ** 2)
        noise_power = signal_power / (10 ** (SNR_dB / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(N_sub) +
                                              1j * np.random.randn(N_sub))
        rx_no_cp[i] += noise

    freq_no_cp = np.fft.fft(rx_no_cp, axis=1) / np.sqrt(N_sub)
    H = np.fft.fft(h, N_sub)
    eq_no_cp = freq_no_cp / H[np.newaxis, :]
    rx_bits_no_cp = np.array([qpsk_demod_symbols(eq_no_cp[i]) for i in range(n_symbols)])
    ber_no_cp = np.mean(tx_bits != rx_bits_no_cp)
    print(f"    BER without CP (with equalization): {ber_no_cp:.6f}")
    print(f"    ISI from multipath causes errors even with equalization")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Constellation before equalization
    axes[0].scatter(raw_rx[:10].real.flatten(), raw_rx[:10].imag.flatten(),
                    s=1, alpha=0.5)
    axes[0].set_title('Before Equalization')
    axes[0].set_xlabel('In-phase')
    axes[0].set_ylabel('Quadrature')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # After equalization
    axes[1].scatter(equalized_rx[:10].real.flatten(), equalized_rx[:10].imag.flatten(),
                    s=1, alpha=0.5)
    axes[1].set_title('After Equalization (with CP)')
    axes[1].set_xlabel('In-phase')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    # Without CP
    axes[2].scatter(eq_no_cp[:10].real.flatten(), eq_no_cp[:10].imag.flatten(),
                    s=1, alpha=0.5)
    axes[2].set_title('After Equalization (no CP)')
    axes[2].set_xlabel('In-phase')
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex16_4_ofdm.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex16_4_ofdm.png")


# === Exercise 5: Radar Waveform Design ===
# Problem: Chirp pulse compression, matched filtering, target resolution,
#          ambiguity function.

def exercise_5():
    """Radar waveform: chirp compression, matched filter, ambiguity function."""
    c = 3e8  # speed of light

    # (a) Chirp pulses with different TBP
    print("(a) Chirp pulse compression:")
    fs = 100e6  # 100 MHz sampling rate
    tbp_values = [10, 50, 200]

    for tbp in tbp_values:
        T = 10e-6  # pulse width 10 us
        B = tbp / T  # bandwidth
        t = np.linspace(-T / 2, T / 2, int(fs * T))
        chirp = np.exp(1j * np.pi * B / T * t ** 2)

        # Matched filter = time-reversed conjugate
        mf = np.conj(chirp[::-1])
        compressed = np.convolve(chirp, mf, mode='full')
        compressed = np.abs(compressed) / np.max(np.abs(compressed))
        compressed_db = 20 * np.log10(compressed + 1e-10)

        # Mainlobe width (-3 dB)
        above_3db = compressed_db > -3
        mainlobe_samples = np.sum(above_3db)
        mainlobe_width = mainlobe_samples / fs

        # Peak sidelobe level
        center = len(compressed) // 2
        half_main = mainlobe_samples // 2 + 5
        sidelobes = compressed_db.copy()
        sidelobes[center - half_main:center + half_main] = -100
        psl = np.max(sidelobes)

        range_res = c / (2 * B)
        print(f"    TBP={tbp:3d}: B={B / 1e6:.1f} MHz, "
              f"compressed width={mainlobe_width * 1e9:.1f} ns, "
              f"PSL={psl:.1f} dB, range res={range_res:.2f} m")

    # (b) Windowed matched filter
    print("\n(b) Windowed matched filter (Hamming):")
    T = 10e-6
    for tbp in tbp_values:
        B = tbp / T
        t = np.linspace(-T / 2, T / 2, int(fs * T))
        chirp = np.exp(1j * np.pi * B / T * t ** 2)

        # Hamming-windowed matched filter
        mf = np.conj(chirp[::-1]) * np.hamming(len(chirp))
        compressed = np.convolve(chirp, mf, mode='full')
        compressed = np.abs(compressed) / np.max(np.abs(compressed))
        compressed_db = 20 * np.log10(compressed + 1e-10)

        above_3db = compressed_db > -3
        mainlobe_samples = np.sum(above_3db)
        mainlobe_width = mainlobe_samples / fs

        center = len(compressed) // 2
        half_main = mainlobe_samples // 2 + 5
        sidelobes = compressed_db.copy()
        sidelobes[center - half_main:center + half_main] = -100
        psl = np.max(sidelobes)

        print(f"    TBP={tbp:3d}: mainlobe={mainlobe_width * 1e9:.1f} ns (wider), "
              f"PSL={psl:.1f} dB (lower)")

    # (c) Two-target resolution
    print("\n(c) Two-target resolution (10 km and 10.03 km):")
    R1, R2 = 10000, 10030  # meters
    T = 10e-6

    for tbp in tbp_values:
        B = tbp / T
        t = np.linspace(-T / 2, T / 2, int(fs * T))
        chirp = np.exp(1j * np.pi * B / T * t ** 2)
        mf = np.conj(chirp[::-1])

        range_res = c / (2 * B)
        delta_R = R2 - R1

        # Two returns at different delays
        delay1 = 2 * R1 / c
        delay2 = 2 * R2 / c
        delay_samples = int((delay2 - delay1) * fs)

        # Composite received signal
        n_total = len(chirp) + delay_samples + 100
        rx = np.zeros(n_total, dtype=complex)
        rx[:len(chirp)] += chirp
        rx[delay_samples:delay_samples + len(chirp)] += 0.8 * chirp

        compressed = np.convolve(rx, mf, mode='full')
        compressed = np.abs(compressed)

        # Check if two peaks are resolvable
        peaks = []
        comp_norm = compressed / np.max(compressed)
        for i in range(1, len(comp_norm) - 1):
            if comp_norm[i] > comp_norm[i - 1] and comp_norm[i] > comp_norm[i + 1]:
                if comp_norm[i] > 0.3:
                    peaks.append(i)

        resolved = len(peaks) >= 2
        print(f"    TBP={tbp:3d}: range_res={range_res:.1f} m, "
              f"delta_R={delta_R:.0f} m, "
              f"resolved={'YES' if resolved else 'NO'} "
              f"({len(peaks)} peaks)")

    # (d) 2D Ambiguity function for TBP=50
    print("\n(d) Ambiguity function for TBP=50:")
    tbp = 50
    T = 10e-6
    B = tbp / T
    t = np.linspace(-T / 2, T / 2, int(fs * T))
    chirp = np.exp(1j * np.pi * B / T * t ** 2)

    # Compute |chi(tau, fd)|
    n_tau = 200
    n_fd = 200
    tau_max = 2 * T
    fd_max = 2 * B
    taus = np.linspace(-tau_max, tau_max, n_tau)
    fds = np.linspace(-fd_max, fd_max, n_fd)

    chi = np.zeros((n_fd, n_tau))
    for i, fd in enumerate(fds):
        # Doppler-shifted reference
        ref_doppler = chirp * np.exp(1j * 2 * np.pi * fd * t)
        cross_corr = np.correlate(chirp, ref_doppler, mode='full')
        # Resample to tau grid
        center = len(cross_corr) // 2
        for j, tau in enumerate(taus):
            idx = center + int(tau * fs)
            if 0 <= idx < len(cross_corr):
                chi[i, j] = np.abs(cross_corr[idx])

    chi /= np.max(chi)
    chi_db = 20 * np.log10(chi + 1e-10)

    print(f"    Pulse width: {T * 1e6:.0f} us, Bandwidth: {B / 1e6:.0f} MHz")
    print(f"    Range-Doppler coupling: chirp creates a tilted ridge")
    print(f"    Ridge slope: delay changes with Doppler shift")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Compressed pulses
    T = 10e-6
    for tbp in tbp_values:
        B_val = tbp / T
        t_ch = np.linspace(-T / 2, T / 2, int(fs * T))
        ch = np.exp(1j * np.pi * B_val / T * t_ch ** 2)
        mf_ch = np.conj(ch[::-1])
        comp = np.abs(np.convolve(ch, mf_ch, mode='full'))
        comp = comp / np.max(comp)
        t_comp = np.arange(len(comp)) / fs * 1e6
        t_comp -= t_comp[len(t_comp) // 2]
        axes[0].plot(t_comp, 20 * np.log10(comp + 1e-10), label=f'TBP={tbp}')
    axes[0].set_xlim([-2, 2])
    axes[0].set_ylim([-60, 5])
    axes[0].set_xlabel('Time (us)')
    axes[0].set_ylabel('Amplitude (dB)')
    axes[0].set_title('Compressed Pulses')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Two-target scenario
    tbp = 200
    B_val = tbp / T
    t_ch = np.linspace(-T / 2, T / 2, int(fs * T))
    ch = np.exp(1j * np.pi * B_val / T * t_ch ** 2)
    mf_ch = np.conj(ch[::-1])
    delay_s = int((2 * 30 / c) * fs)
    n_total = len(ch) + delay_s + 100
    rx = np.zeros(n_total, dtype=complex)
    rx[:len(ch)] += ch
    rx[delay_s:delay_s + len(ch)] += 0.8 * ch
    comp = np.abs(np.convolve(rx, mf_ch, mode='full'))
    comp /= np.max(comp)
    axes[1].plot(comp[:500])
    axes[1].set_title('Two Targets (30m apart, TBP=200)')
    axes[1].set_xlabel('Sample')
    axes[1].grid(True, alpha=0.3)

    # Ambiguity function
    extent = [taus[0] * 1e6, taus[-1] * 1e6, fds[0] / 1e6, fds[-1] / 1e6]
    axes[2].imshow(chi_db, extent=extent, aspect='auto',
                   cmap='viridis', vmin=-40, vmax=0, origin='lower')
    axes[2].set_xlabel('Delay (us)')
    axes[2].set_ylabel('Doppler (MHz)')
    axes[2].set_title('Ambiguity Function (TBP=50)')

    plt.tight_layout()
    plt.savefig('ex16_5_radar.png', dpi=150)
    plt.close()
    print("    Plot saved: ex16_5_radar.png")


# === Exercise 6: ECG Analysis Pipeline ===
# Problem: Synthetic ECG, preprocessing, Pan-Tompkins QRS detection, HRV analysis.

def exercise_6():
    """ECG analysis: synthetic ECG, preprocessing, Pan-Tompkins QRS, HRV."""
    np.random.seed(42)
    fs = 360  # Hz (standard ECG sampling rate)
    duration = 30  # seconds
    heart_rate = 75  # bpm
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # (a) Generate synthetic ECG
    print("(a) Synthetic ECG generation:")
    period = 60 / heart_rate
    n_beats = int(duration / period)

    def ecg_template(t_local, period_s):
        """Simple ECG template (PQRST complex)."""
        t_norm = t_local / period_s
        ecg = np.zeros_like(t_norm)

        # P wave
        p_center, p_width, p_amp = 0.1, 0.04, 0.15
        ecg += p_amp * np.exp(-((t_norm - p_center) ** 2) / (2 * p_width ** 2))

        # Q wave
        q_center, q_width, q_amp = 0.19, 0.008, -0.1
        ecg += q_amp * np.exp(-((t_norm - q_center) ** 2) / (2 * q_width ** 2))

        # R wave
        r_center, r_width, r_amp = 0.2, 0.01, 1.0
        ecg += r_amp * np.exp(-((t_norm - r_center) ** 2) / (2 * r_width ** 2))

        # S wave
        s_center, s_width, s_amp = 0.22, 0.012, -0.2
        ecg += s_amp * np.exp(-((t_norm - s_center) ** 2) / (2 * s_width ** 2))

        # T wave
        t_center, t_width, t_amp = 0.35, 0.06, 0.3
        ecg += t_amp * np.exp(-((t_norm - t_center) ** 2) / (2 * t_width ** 2))

        return ecg

    # Generate ECG with slight HRV
    clean_ecg = np.zeros_like(t)
    beat_times = []
    current_time = 0
    while current_time < duration - period:
        # Add slight heart rate variability
        rr_interval = period + 0.02 * np.random.randn()
        beat_idx = int(current_time * fs)
        beat_len = int(rr_interval * fs)
        if beat_idx + beat_len <= len(t):
            t_local = t[beat_idx:beat_idx + beat_len] - t[beat_idx]
            clean_ecg[beat_idx:beat_idx + beat_len] += ecg_template(t_local, rr_interval)
            beat_times.append(current_time + 0.2 * rr_interval)  # R-peak time
        current_time += rr_interval

    # Add noise
    baseline_wander = 0.3 * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz
    powerline = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50 Hz
    signal_power = np.mean(clean_ecg ** 2)
    noise_power = signal_power / (10 ** (20 / 10))  # 20 dB SNR
    random_noise = np.sqrt(noise_power) * np.random.randn(len(t))

    noisy_ecg = clean_ecg + baseline_wander + powerline + random_noise

    print(f"    Duration: {duration} s, Heart rate: {heart_rate} bpm")
    print(f"    Number of beats: {len(beat_times)}")
    print(f"    Noise: baseline wander (0.2 Hz), powerline (50 Hz), random (20 dB SNR)")

    # (b) Preprocessing pipeline
    print("\n(b) Preprocessing pipeline:")

    # Highpass filter (0.5 Hz) - remove baseline wander
    sos_hp = sig.butter(4, 0.5, btype='high', fs=fs, output='sos')
    ecg_hp = sig.sosfilt(sos_hp, noisy_ecg)
    print(f"    Step 1: Highpass 0.5 Hz (remove baseline wander)")

    # Notch filter (50 Hz) - remove powerline interference
    b_notch, a_notch = sig.iirnotch(50, Q=30, fs=fs)
    ecg_notch = sig.filtfilt(b_notch, a_notch, ecg_hp)
    print(f"    Step 2: Notch filter at 50 Hz (Q=30)")

    # Bandpass filter (1-40 Hz)
    sos_bp = sig.butter(4, [1, 40], btype='band', fs=fs, output='sos')
    ecg_clean = sig.sosfilt(sos_bp, ecg_notch)
    print(f"    Step 3: Bandpass 1-40 Hz")

    residual_50hz = np.abs(np.fft.fft(ecg_clean)[int(50 * duration)])
    print(f"    50 Hz residual energy: {residual_50hz:.4f}")

    # (c) Pan-Tompkins QRS detection
    print("\n(c) Pan-Tompkins QRS detection:")

    def pan_tompkins(ecg_signal, fs_rate):
        """Simplified Pan-Tompkins QRS detector."""
        # 1. Bandpass filter (5-15 Hz)
        sos = sig.butter(2, [5, 15], btype='band', fs=fs_rate, output='sos')
        filtered = sig.sosfilt(sos, ecg_signal)

        # 2. Derivative filter
        deriv = np.diff(filtered)
        deriv = np.append(deriv, 0)

        # 3. Squaring
        squared = deriv ** 2

        # 4. Moving window integration (150 ms)
        win_len = int(0.15 * fs_rate)
        integrated = np.convolve(squared, np.ones(win_len) / win_len, mode='same')

        # 5. Thresholding and peak detection
        threshold = 0.3 * np.max(integrated)
        peaks, _ = sig.find_peaks(integrated, height=threshold,
                                   distance=int(0.3 * fs_rate))
        return peaks

    detected_peaks = pan_tompkins(ecg_clean, fs)
    detected_times = detected_peaks / fs

    # Compare with ground truth
    gt_times = np.array(beat_times)
    true_pos = 0
    for dt in detected_times:
        if np.min(np.abs(gt_times - dt)) < 0.1:  # 100 ms tolerance
            true_pos += 1

    sensitivity = true_pos / len(gt_times) if len(gt_times) > 0 else 0
    ppv = true_pos / len(detected_times) if len(detected_times) > 0 else 0

    print(f"    Ground truth beats: {len(gt_times)}")
    print(f"    Detected beats: {len(detected_times)}")
    print(f"    True positives: {true_pos}")
    print(f"    Sensitivity: {sensitivity:.3f}")
    print(f"    PPV: {ppv:.3f}")

    # (d) HRV analysis
    print("\n(d) HRV analysis:")
    if len(detected_peaks) > 2:
        rr_intervals = np.diff(detected_peaks) / fs * 1000  # in ms

        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = nn50 / (len(rr_intervals) - 1) * 100 if len(rr_intervals) > 1 else 0

        mean_hr = 60000 / np.mean(rr_intervals)

        print(f"    Mean RR interval: {np.mean(rr_intervals):.1f} ms")
        print(f"    Mean heart rate: {mean_hr:.1f} bpm (expected: {heart_rate})")
        print(f"    SDNN: {sdnn:.1f} ms")
        print(f"    RMSSD: {rmssd:.1f} ms")
        print(f"    pNN50: {pnn50:.1f}%")

        # (e) HRV power spectrum
        print("\n(e) HRV power spectrum:")
        if len(rr_intervals) > 10:
            # Interpolate RR intervals to uniform sampling
            rr_times = np.cumsum(rr_intervals) / 1000
            fs_hrv = 4  # Hz
            t_uniform = np.arange(rr_times[0], rr_times[-1], 1 / fs_hrv)
            rr_interp = np.interp(t_uniform, rr_times, rr_intervals[:-1]
                                  if len(rr_intervals) > len(rr_times) else rr_intervals[:len(rr_times)])

            # Welch PSD
            f_hrv, psd_hrv = sig.welch(rr_interp - np.mean(rr_interp),
                                        fs=fs_hrv, nperseg=min(64, len(rr_interp)))

            # Band powers
            lf_mask = (f_hrv >= 0.04) & (f_hrv < 0.15)
            hf_mask = (f_hrv >= 0.15) & (f_hrv < 0.4)
            lf_power = np.trapz(psd_hrv[lf_mask], f_hrv[lf_mask])
            hf_power = np.trapz(psd_hrv[hf_mask], f_hrv[hf_mask])

            print(f"    LF power (0.04-0.15 Hz): {lf_power:.2f} ms^2")
            print(f"    HF power (0.15-0.40 Hz): {hf_power:.2f} ms^2")
            print(f"    LF/HF ratio: {lf_power / max(hf_power, 1e-10):.2f}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    t_show = 5  # show 5 seconds
    mask = t < t_show

    axes[0, 0].plot(t[mask], noisy_ecg[mask])
    axes[0, 0].set_title('Noisy ECG')
    axes[0, 0].set_xlabel('Time (s)')

    axes[0, 1].plot(t[mask], ecg_clean[mask])
    peaks_mask = detected_peaks[detected_peaks < int(t_show * fs)]
    axes[0, 1].plot(peaks_mask / fs, ecg_clean[peaks_mask], 'rv', markersize=8)
    axes[0, 1].set_title('Filtered + QRS Detection')
    axes[0, 1].set_xlabel('Time (s)')

    axes[1, 0].plot(t[mask], ecg_hp[mask], label='HP')
    axes[1, 0].set_title('After Highpass')
    axes[1, 0].set_xlabel('Time (s)')

    axes[1, 1].plot(t[mask], ecg_notch[mask])
    axes[1, 1].set_title('After Notch (50 Hz)')
    axes[1, 1].set_xlabel('Time (s)')

    if len(detected_peaks) > 2:
        axes[2, 0].plot(rr_intervals, 'b-o', markersize=3)
        axes[2, 0].set_title('RR Intervals')
        axes[2, 0].set_xlabel('Beat #')
        axes[2, 0].set_ylabel('RR (ms)')
        axes[2, 0].grid(True, alpha=0.3)

        if len(rr_intervals) > 10:
            axes[2, 1].semilogy(f_hrv, psd_hrv)
            axes[2, 1].axvspan(0.04, 0.15, alpha=0.2, color='blue', label='LF')
            axes[2, 1].axvspan(0.15, 0.4, alpha=0.2, color='red', label='HF')
            axes[2, 1].set_title('HRV Power Spectrum')
            axes[2, 1].set_xlabel('Frequency (Hz)')
            axes[2, 1].set_ylabel('PSD (ms^2/Hz)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex16_6_ecg_analysis.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex16_6_ecg_analysis.png")


# === Exercise 7: EEG Spectral Analysis ===
# Problem: Simulate EEG with alpha/beta bands, Welch PSD, STFT time-frequency,
#          sliding window band power.

def exercise_7():
    """EEG analysis: spectral bands, eyes open/closed, band power tracking."""
    np.random.seed(42)
    fs = 256  # Hz
    duration = 10  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # (a) Simulate EEG signal
    print("(a) EEG signal simulation:")

    # 1/f background noise
    n_samples = len(t)
    freqs = np.fft.rfftfreq(n_samples, 1 / fs)
    freqs[0] = 1  # avoid division by zero
    pink_spectrum = 1 / np.sqrt(freqs) * np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
    pink_noise = np.fft.irfft(pink_spectrum, n=n_samples)
    pink_noise *= 10 / np.std(pink_noise)  # scale to ~10 uV

    # Alpha (10 Hz) - eyes closed dominant
    alpha_amp = np.ones_like(t) * 5  # uV baseline
    # Eyes closed from t=2 to t=4 and t=6 to t=8: alpha increases
    eyes_closed = ((t >= 2) & (t < 4)) | ((t >= 6) & (t < 8))
    alpha_amp[eyes_closed] = 20  # stronger during eyes closed
    # Smooth transitions
    from scipy.ndimage import gaussian_filter1d
    alpha_amp = gaussian_filter1d(alpha_amp, sigma=int(0.2 * fs))
    alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)

    # Beta (20 Hz) - present during concentration
    beta_amp = np.ones_like(t) * 3  # uV
    beta = beta_amp * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)

    eeg = pink_noise + alpha + beta

    print(f"    Duration: {duration} s, fs: {fs} Hz")
    print(f"    Alpha (10 Hz): ~5 uV normally, ~20 uV eyes closed")
    print(f"    Beta (20 Hz): ~3 uV constant")
    print(f"    Background: 1/f noise (~10 uV)")
    print(f"    Eyes closed periods: [2-4] s, [6-8] s")

    # (b) Welch PSD
    print("\n(b) Welch PSD (2-second windows, 50% overlap):")
    nperseg = 2 * fs  # 2-second windows
    noverlap = nperseg // 2
    f_psd, psd = sig.welch(eeg, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Identify band peaks
    alpha_mask = (f_psd >= 8) & (f_psd <= 13)
    beta_mask = (f_psd >= 13) & (f_psd <= 30)
    delta_mask = (f_psd >= 0.5) & (f_psd <= 4)
    theta_mask = (f_psd >= 4) & (f_psd <= 8)

    alpha_peak = f_psd[alpha_mask][np.argmax(psd[alpha_mask])]
    beta_peak = f_psd[beta_mask][np.argmax(psd[beta_mask])]

    alpha_power = np.trapz(psd[alpha_mask], f_psd[alpha_mask])
    beta_power = np.trapz(psd[beta_mask], f_psd[beta_mask])
    total_power = np.trapz(psd, f_psd)

    print(f"    Alpha peak: {alpha_peak:.1f} Hz")
    print(f"    Beta peak: {beta_peak:.1f} Hz")
    print(f"    Alpha power: {alpha_power:.2f} uV^2 ({alpha_power / total_power * 100:.1f}%)")
    print(f"    Beta power: {beta_power:.2f} uV^2 ({beta_power / total_power * 100:.1f}%)")

    # (c) STFT time-frequency analysis
    print("\n(c) STFT showing time-varying alpha power:")
    f_stft, t_stft, Zxx = sig.stft(eeg, fs=fs, nperseg=256, noverlap=224)
    Sxx = np.abs(Zxx) ** 2

    # Alpha band power over time
    alpha_band = (f_stft >= 8) & (f_stft <= 13)
    alpha_power_t = np.mean(Sxx[alpha_band], axis=0)

    # Find periods of high alpha
    alpha_threshold = np.percentile(alpha_power_t, 70)
    high_alpha = alpha_power_t > alpha_threshold
    print(f"    STFT window: 256 samples (1 s), overlap: 224")
    print(f"    Alpha power clearly increases during eyes-closed periods")
    print(f"    High alpha detected at t = ", end="")
    changes = np.diff(high_alpha.astype(int))
    starts = t_stft[:-1][changes == 1]
    ends = t_stft[:-1][changes == -1]
    for s in starts[:4]:
        print(f"{s:.1f}s ", end="")
    print()

    # (d) Sliding window band power
    print("\n(d) Relative band power over time:")
    win_len = int(2 * fs)  # 2-second window
    hop = int(0.5 * fs)    # 0.5-second hop

    times_bp = []
    alpha_rel = []
    beta_rel = []

    for start in range(0, len(eeg) - win_len, hop):
        segment = eeg[start:start + win_len]
        f_seg, psd_seg = sig.welch(segment, fs=fs, nperseg=min(256, win_len))

        total = np.trapz(psd_seg, f_seg)
        alpha_m = (f_seg >= 8) & (f_seg <= 13)
        beta_m = (f_seg >= 13) & (f_seg <= 30)
        a_pow = np.trapz(psd_seg[alpha_m], f_seg[alpha_m])
        b_pow = np.trapz(psd_seg[beta_m], f_seg[beta_m])

        times_bp.append((start + win_len / 2) / fs)
        alpha_rel.append(a_pow / total)
        beta_rel.append(b_pow / total)

    print(f"    Window: 2 s, Hop: 0.5 s")
    print(f"    Mean alpha relative power: {np.mean(alpha_rel) * 100:.1f}%")
    print(f"    Mean beta relative power: {np.mean(beta_rel) * 100:.1f}%")
    print(f"    Alpha peaks during eyes-closed epochs as expected")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # EEG signal
    axes[0, 0].plot(t, eeg, 'k', linewidth=0.5)
    for start, end in [(2, 4), (6, 8)]:
        axes[0, 0].axvspan(start, end, alpha=0.2, color='blue', label='Eyes closed' if start == 2 else '')
    axes[0, 0].set_title('Simulated EEG')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (uV)')
    axes[0, 0].legend()

    # Welch PSD
    axes[0, 1].semilogy(f_psd, psd)
    axes[0, 1].axvspan(8, 13, alpha=0.2, color='blue', label='Alpha')
    axes[0, 1].axvspan(13, 30, alpha=0.2, color='red', label='Beta')
    axes[0, 1].set_title('Welch PSD')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD (uV^2/Hz)')
    axes[0, 1].set_xlim([0, 50])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Spectrogram
    axes[1, 0].pcolormesh(t_stft, f_stft, 10 * np.log10(Sxx + 1e-10),
                          shading='gouraud', cmap='viridis', vmin=-20)
    axes[1, 0].set_ylim([0, 50])
    axes[1, 0].set_title('STFT Spectrogram')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')

    # Alpha power over time
    axes[1, 1].plot(t_stft, alpha_power_t)
    for start, end in [(2, 4), (6, 8)]:
        axes[1, 1].axvspan(start, end, alpha=0.2, color='blue')
    axes[1, 1].set_title('Alpha Band Power (STFT)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Power')
    axes[1, 1].grid(True, alpha=0.3)

    # Relative band power
    axes[2, 0].plot(times_bp, alpha_rel, 'b-', label='Alpha')
    axes[2, 0].plot(times_bp, beta_rel, 'r-', label='Beta')
    for start, end in [(2, 4), (6, 8)]:
        axes[2, 0].axvspan(start, end, alpha=0.2, color='blue')
    axes[2, 0].set_title('Relative Band Power')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Relative Power')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Alpha/Beta ratio
    alpha_arr = np.array(alpha_rel)
    beta_arr = np.array(beta_rel)
    ratio = alpha_arr / (beta_arr + 1e-10)
    axes[2, 1].plot(times_bp, ratio, 'g-')
    for start, end in [(2, 4), (6, 8)]:
        axes[2, 1].axvspan(start, end, alpha=0.2, color='blue')
    axes[2, 1].set_title('Alpha/Beta Ratio')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Ratio')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex16_7_eeg_analysis.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex16_7_eeg_analysis.png")


# === Main ===

def main():
    exercises = [
        ("Exercise 1: Audio Effects Chain", exercise_1),
        ("Exercise 2: Pitch Detection Robustness", exercise_2),
        ("Exercise 3: Digital Modulation BER", exercise_3),
        ("Exercise 4: OFDM System", exercise_4),
        ("Exercise 5: Radar Waveform Design", exercise_5),
        ("Exercise 6: ECG Analysis Pipeline", exercise_6),
        ("Exercise 7: EEG Spectral Analysis", exercise_7),
    ]

    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}\n")
        func()


if __name__ == "__main__":
    main()
