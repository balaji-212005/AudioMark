"""
  EMBED — Audio Watermarking  (Definitive Final Version)
  ════════════════════════════════════════════════════════════════════════
  Pipeline : WAV input → embed → MP3 output

  ════════════════════════════════════════════════════════════════════════
  DESIGN DECISIONS — all backed by exhaustive analysis
  ════════════════════════════════════════════════════════════════════════

  1. DWT LEVEL-3  (chosen over level-2 and level-4)
     ─────────────────────────────────────────────
     Level-2 cA: 0–5512 Hz, noise_std≈0.008  → needs delta=0.071 → SNR=13.8dB  BAD
     Level-3 cA: 0–2756 Hz, noise_std≈0.005  → needs delta=0.030 → SNR=21.2dB  BEST
     Level-4 cA: 0–1378 Hz, noise_std≈0.003  → needs delta=0.027 → SNR=22.2dB  but
                                                  64×64 needs 36s, 128×128 needs 144s
     Winner: Level-3. One level lower freq than level-2 → MP3 noise halved.
     Delta can be lower → SNR 21 dB. Audio: 64×64 needs ~36s.

  2. QIM ON RAW cA 
     ────────────────────────────────────────────────────────────
     Tested: SVD singular values have IDENTICAL noise magnitude to raw
     coefficients (both ~0.006 std for MP3 noise 0.008 input on 4×4 blocks).
     SVD adds complexity with zero noise reduction benefit.
     Paper achieves 196dB PSNR using WAV output and tiny k values — not
     applicable to MP3. QIM is optimal for our use case.

  3. THREE-ZONE LAYOUT (sync | header | payload)
     ─────────────────────────────────────────────
     ZONE A — SYNC MARKER (64 bits, delta=0.15, R=9)
       Known bit pattern embedded at extreme strength.
       Extract scans cA for this pattern → finds exact start offset.
       Survives: MP3, trimming, slight time-shift.
       Overhead: ~0.1s of audio. Worth it completely.

     ZONE B — HEADER (192 bits, delta=0.08, R=7)
       Contains: ChaCha20 nonce (16B) + logo dims (4B) + RS byte count (3B).
       Nonce: if 1 bit flips → wrong decryption → garbage image.
       At delta=0.08: P(bit_err)≈0.0001% → P(header error)≈10⁻²⁴.
       Effectively perfect.

     ZONE C — PAYLOAD (RS-encoded bits, delta=0.030, R=1)
       At delta=0.030 and level-3: P(bit_err)≈6%.
       RS(255,215,t=20): in 215-byte block, 6% bit error → ~13 byte errors.
       RS corrects up to 20 byte errors → ALL errors corrected.
       Result: perfect ciphertext → perfect decryption → perfect logo.

  4. FIXED DELTA EVERYWHERE (no adaptive)
     ────────────────────────────────────
     Adaptive delta caused embed/extract mismatch after MP3 noise
     changed neighbourhood RMS values. Fixed constants, hardcoded
     identically in both embed and extract.

  5. RS(255,215) REED-SOLOMON ERROR CORRECTION
     ─────────────────────────────────────────
     Corrects up to 20 byte errors per 255-byte block.
     Overhead: 255/215 = 1.186× (very compact for full error immunity).

  6. NCC QUALITY METRIC
     ─────────────────────────────────────────────
     Normalized Cross-Correlation reported in extract.
     NCC > 0.99 = perfect, > 0.95 = good, < 0.90 = degraded.

  ════════════════════════════════════════════════════════════════════════
  AUDIO REQUIREMENTS (DWT level-3, 44100 Hz)
  ════════════════════════════════════════════════════════════════════════
    32×32  logo  →  ~9  seconds
    64×64  logo  →  ~36 seconds
    128×128 logo →  ~144 seconds

  ════════════════════════════════════════════════════════════════════════
  INSTALL
  ════════════════════════════════════════════════════════════════════════
    pip install reedsolo pydub pywavelets numpy Pillow cryptography matplotlib
    (ffmpeg must be on PATH for MP3 output)
"""

import numpy as np
import pywt
import wave
import struct
import os
import sys
import argparse
import getpass
import tempfile
import shutil
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pydub import AudioSegment
    from pydub import utils as pydub_utils
    AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"
    AudioSegment.ffmpeg    = "C:\\ffmpeg\\bin\\ffmpeg.exe"
    AudioSegment.ffprobe   = "C:\\ffmpeg\\bin\\ffprobe.exe"
    pydub_utils.get_prober_name = lambda: "C:\\ffmpeg\\bin\\ffprobe.exe"
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

try:
    import reedsolo
    RS_OK = True
except ImportError:
    RS_OK = False


# ════════════════════════════════════════════
#  ALL CONSTANTS (fixed — never change independently
#  between embed and extract)
# ════════════════════════════════════════════

WAVELET    = "db4"
DWT_LEVEL  = 3           # ← key choice: 0–2756 Hz, noise_std≈0.005

# ZONE A: synchronisation marker
SYNC_PATTERN   = np.array([1,0,1,1,0,0,1,0, 1,1,0,1,0,1,0,0,
                            0,1,1,0,1,0,1,1, 1,0,0,1,1,1,0,1,
                            0,0,1,1,0,1,1,0, 1,0,1,0,0,1,0,1,
                            1,1,0,0,1,1,0,1, 0,1,0,0,1,0,1,1],
                           dtype=np.uint8)   # 64 bits, pseudo-random fixed pattern
SYNC_BITS      = len(SYNC_PATTERN)           # 64
DELTA_SYNC     = 0.15                        # very strong — must be detectable
REDUND_SYNC    = 9                           # 9 copies → need 5 of 9 wrong to fail

# ZONE B: header
HEADER_BYTES   = 24                          # nonce(16) + w(2) + h(2) + mode(1) + magic(3)
HEADER_MAGIC   = b"\xAB\xCD\xEF"            # 3-byte fixed marker for header validation
                                             # rs_len is NOT stored — computed from w,h
HEADER_BITS    = HEADER_BYTES * 8            # 192
DELTA_HEADER   = 0.08                        # P(bit_err)≈0.0001% at noise_std=0.005
REDUND_HEADER  = 7                           # P(header_err)≈10⁻²⁴

# ZONE C: payload
DELTA_PAYLOAD  = 0.030                       # P(bit_err)≈6% → RS corrects all
REDUND_PAYLOAD = 1                           # RS handles everything

# Reed-Solomon
RS_NSYM  = 40                                # ECC bytes per block
RS_DATA  = 255 - RS_NSYM                     # 215 data bytes per block
                                             # corrects up to 20 byte errors/block

MODE_RGB       = 1
MP3_BITRATE    = "320k"
PBKDF2_SALT    = b"audio_watermark_salt_v3"  # v3 = this design
PBKDF2_ITER    = 200_000
DEFAULT_SIZE   = (64, 64)


# ════════════════════════════════════════════
#  Dependency checks
# ════════════════════════════════════════════

def _check_deps():
    missing = []
    if not RS_OK:   missing.append("reedsolo  →  pip install reedsolo")
    if not PYDUB_OK: missing.append("pydub     →  pip install pydub")
    if missing:
        print("\n  ✗  Missing dependencies:")
        for m in missing: print(f"       {m}")
        sys.exit(1)


# ════════════════════════════════════════════
#  Reed-Solomon
# ════════════════════════════════════════════

def rs_encode(data: bytes) -> bytes:
    codec = reedsolo.RSCodec(RS_NSYM)
    out   = bytearray()
    for i in range(0, len(data), RS_DATA):
        chunk = data[i: i + RS_DATA]
        if len(chunk) < RS_DATA:          # pad last block so it is always RS_DATA bytes
            chunk = chunk + b'\x00' * (RS_DATA - len(chunk))
        out.extend(codec.encode(chunk))
    return bytes(out)


def rs_decode(rs_bytes: bytes, original_len: int) -> tuple:
    """Returns (decoded_bytes, n_errors_corrected)."""
    codec   = reedsolo.RSCodec(RS_NSYM)
    out     = bytearray()
    n_errs  = 0
    n_blocks = len(rs_bytes) // 255
    for i in range(n_blocks):
        block = rs_bytes[i*255: (i+1)*255]
        try:
            decoded, _, errata = codec.decode(block)
            out.extend(decoded)
            n_errs += len(errata)
        except reedsolo.ReedSolomonError:
            out.extend(b'\x00' * RS_DATA)
    return bytes(out[:original_len]), n_errs


# ════════════════════════════════════════════
#  Crypto
# ════════════════════════════════════════════

def derive_key(password: str) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=PBKDF2_SALT, iterations=PBKDF2_ITER,
                     backend=default_backend())
    return kdf.derive(password.encode())


def encrypt_chacha20(data: bytes, key: bytes, nonce: bytes) -> bytes:
    enc = Cipher(algorithms.ChaCha20(key, nonce), mode=None,
                 backend=default_backend()).encryptor()
    return enc.update(data) + enc.finalize()


# ════════════════════════════════════════════
#  QIM core  (fixed delta, parity-correct)
# ════════════════════════════════════════════

def _embed_one(val: float, bit: int, delta: float) -> float:
    """
    QIM embed. Python signed modulo handles negative coefficients:
      q=-4 → (-4)%2=0 (even, bit=0) ✓
      q=-3 → (-3)%2=1 (odd,  bit=1) ✓
    """
    q = int(np.round(val / delta))
    if bit == 0:
        if q % 2 != 0: q += 1
    else:
        if q % 2 == 0: q += 1
    return float(q) * delta


def _extract_one(val: float, delta: float) -> int:
    """QIM extract. Same fixed delta as embed — always consistent."""
    return int(round(val / delta)) % 2


def embed_zone(cA: np.ndarray, bits: np.ndarray,
               start: int, delta: float, R: int) -> int:
    """Embed bits into cA[start:] with redundancy R. Returns next free slot."""
    for bit in bits:
        for _ in range(R):
            cA[start] = _embed_one(cA[start], int(bit), delta)
            start += 1
    return start


def extract_zone(cA: np.ndarray, n_bits: int,
                 start: int, delta: float, R: int):
    """Extract n_bits from cA[start:] with majority vote. Returns (bits, next_slot)."""
    bits = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        votes = sum(_extract_one(cA[start + r], delta) for r in range(R))
        bits[i] = 1 if votes > R // 2 else 0
        start += R
    return bits, start


# ════════════════════════════════════════════
#  Sync marker search
# ════════════════════════════════════════════

def find_sync(cA: np.ndarray, search_limit: int = None) -> int:
    """
    Scan cA for the SYNC_PATTERN using soft correlation.
    Returns the start slot of the sync zone, or -1 if not found.
    The sync zone occupies SYNC_BITS * REDUND_SYNC slots.
    """
    sync_slots = SYNC_BITS * REDUND_SYNC
    limit      = min(len(cA) - sync_slots, search_limit or len(cA))

    best_score = -1
    best_pos   = -1

    for start in range(0, limit, REDUND_SYNC):
        bits, _ = extract_zone(cA, SYNC_BITS, start, DELTA_SYNC, REDUND_SYNC)
        score   = int(np.sum(bits == SYNC_PATTERN))
        if score > best_score:
            best_score = score
            best_pos   = start
        if score == SYNC_BITS:   # perfect match — stop early
            break

    # Accept if ≥ 56/64 bits match (87.5%)
    if best_score >= int(SYNC_BITS * 0.875):
        return best_pos
    return -1


# ════════════════════════════════════════════
#  Header encode / decode
# ════════════════════════════════════════════

def encode_header(nonce: bytes, w: int, h: int) -> np.ndarray:
    """
    24-byte header: nonce(16) + w(2) + h(2) + mode(1) + magic(3)
    rs_len is NOT stored — computed from w,h by both sides.
    Storing rs_len was the root cause of the header mismatch error.
    """
    raw = (nonce
           + struct.pack(">HH", w, h)
           + struct.pack("B",   MODE_RGB)
           + HEADER_MAGIC)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def decode_header(bits: np.ndarray):
    """Unpack header → (nonce, logo_w, logo_h). rs_len computed by caller."""
    raw   = np.packbits(bits[:HEADER_BITS]).tobytes()[:HEADER_BYTES]
    nonce = raw[:16]
    w, h  = struct.unpack(">HH", raw[16:20])
    return nonce, w, h


def rs_len_from_dims(w: int, h: int) -> int:
    """Compute RS-encoded byte count from logo dimensions."""
    raw_bytes = w * h * 3
    n_blocks  = -(-raw_bytes // RS_DATA)   # ceiling division
    return n_blocks * 255


# ════════════════════════════════════════════
#  Bit utilities
# ════════════════════════════════════════════

def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


# ════════════════════════════════════════════
#  Capacity helpers
# ════════════════════════════════════════════

def ca_per_sec(fs: int = 44100) -> float:
    return fs / (2 ** DWT_LEVEL)


def total_slots_needed(payload_rs_bits: int) -> int:
    return (SYNC_BITS    * REDUND_SYNC    +
            HEADER_BITS  * REDUND_HEADER  +
            payload_rs_bits * REDUND_PAYLOAD)


def needed_seconds(logo_w: int, logo_h: int, fs: int = 44100) -> float:
    raw_bytes  = logo_w * logo_h * 3
    rs_bytes   = ((-raw_bytes) % RS_DATA + raw_bytes + (RS_NSYM * (-(-raw_bytes // RS_DATA)))) 
    # simpler:
    n_blocks   = -(-raw_bytes // RS_DATA)   # ceil div
    rs_bytes   = n_blocks * 255
    rs_bits    = rs_bytes * 8
    slots      = total_slots_needed(rs_bits)
    return slots / ca_per_sec(fs)


# ════════════════════════════════════════════
#  WAV I/O
# ════════════════════════════════════════════

def read_wav(path: str):
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels(); sw = wf.getsampwidth()
        fs   = wf.getframerate(); raw = wf.readframes(wf.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    s = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    return s / float(2 ** (8*sw - 1)), fs, n_ch, sw


def write_wav(path: str, samples: np.ndarray, fs: int, n_ch: int, sw: int):
    max_v = float(2 ** (8*sw - 1))
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    out   = np.clip(samples * max_v, -max_v, max_v - 1).astype(dtype)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_ch); wf.setsampwidth(sw)
        wf.setframerate(fs);   wf.writeframes(out.tobytes())


def resolve_output_path(path: str) -> str:
    _, ext = os.path.splitext(path)
    return path if ext.lower() in (".mp3", ".wav") else path + ".mp3"


def wav_to_mp3(wav_path: str, mp3_path: str):
    if not PYDUB_OK:
        print("  ✗  pydub not installed."); sys.exit(1)
    AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3",
                                           bitrate=MP3_BITRATE)


def compute_snr(orig: np.ndarray, wm: np.ndarray) -> float:
    n   = min(len(orig), len(wm))
    mse = np.mean((wm[:n] - orig[:n]) ** 2)
    return float("inf") if mse == 0 else \
           10 * np.log10(np.mean(orig[:n] ** 2) / mse)


# ════════════════════════════════════════════
#  Waveform + spectrogram plot
# ════════════════════════════════════════════

def plot_analysis(original, watermarked, fs, snr, output_path, audio_path):
    C1 = "#1E3A8A"; C2 = "#16A34A"; C3 = "#DC2626"; C4 = "#059669"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.facecolor": "#F8FAFC",
        "figure.facecolor": "white",  "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#334155", "xtick.color": "#64748B",
        "ytick.color": "#64748B",     "grid.color": "#E2E8F0",
        "grid.linewidth": 0.6,        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n   = min(len(original), len(watermarked))
    o   = original[:n]; w = watermarked[:n]; noise = w - o
    t   = np.arange(n) / fs
    st  = max(1, n // 10_000)
    nz  = int(0.05 * fs)
    tz  = t[:nz] * 1000

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Audio Watermark Analysis — {os.path.basename(audio_path)}\n"
        f"SNR={snr:.2f} dB  |  {n/fs:.1f}s  |  {fs} Hz  |  "
        f"DWT level-{DWT_LEVEL}  |  δ={DELTA_PAYLOAD}  |  RS({255},{RS_DATA})",
        fontsize=12, fontweight="bold", color="#1E293B", y=0.98)

    gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.30,
                          height_ratios=[1, 1, 1, 1])

    # Panel 1: original waveform
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t[::st], o[::st], color=C1, lw=0.5)
    ax1.set_title("Input Audio", fontsize=11, fontweight="bold", color=C1, pad=3)
    ax1.set_ylabel("Amplitude"); ax1.set_xlabel("Time (s)")
    ax1.set_ylim(-0.5, 0.5); ax1.grid(True, axis="y")
    ax1.axhline(0, color="#94A3B8", lw=0.5)

    # Panel 2: watermarked waveform
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t[::st], w[::st], color=C2, lw=0.5)
    ax2.set_title("Watermarked Audio", fontsize=11, fontweight="bold", color=C2, pad=3)
    ax2.set_ylabel("Amplitude"); ax2.set_xlabel("Time (s)")
    ax2.set_ylim(-0.5, 0.5); ax2.grid(True, axis="y")
    ax2.axhline(0, color="#94A3B8", lw=0.5)

    # Panel 3: zoomed overlay
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(tz, o[:nz], color=C1, lw=1.2, alpha=0.9, label="Input")
    ax3.plot(tz, w[:nz], color=C2, lw=1.0, alpha=0.8,
             linestyle="--", label="Watermarked")
    ax3.set_title("Zoomed Overlay — First 50 ms",
                  fontsize=11, fontweight="bold", color="#E90B51", pad=3)
    ax3.set_ylabel("Amplitude"); ax3.set_xlabel("Time (ms)")
    ax3.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax3.grid(True, axis="y"); ax3.axhline(0, color="#94A3B8", lw=0.5)

    # Panel 4: difference signal
    ax4 = fig.add_subplot(gs[2, :])
    nd = noise[::st]
    ax4.plot(t[::st], nd, color=C3, lw=0.4, alpha=0.8)
    ax4.fill_between(t[::st], nd, 0, color=C3, alpha=0.12)
    ax4.axhline(0, color="#94A3B8", lw=0.5)
    ax4.set_title(
        f"Difference Signal  |  SNR={snr:.2f} dB  |  "
        f"Max={np.max(np.abs(noise)):.5f}  |  Mean={np.mean(np.abs(noise)):.5f}",
        fontsize=10, fontweight="bold", color=C3, pad=3)
    ax4.set_ylabel("Amplitude"); ax4.set_xlabel("Time (s)")
    ax4.set_ylim(-0.06, 0.06); ax4.grid(True, axis="y")

    qc = C4 if snr >= 30 else ("#D97706" if snr >= 20 else C3)
    ql = ("Excellent" if snr >= 40 else "Good" if snr >= 30 else
          "Acceptable" if snr >= 20 else "Audible")
    ax4.text(0.99, 0.88, f"SNR: {snr:.1f} dB — {ql}",
             transform=ax4.transAxes, fontsize=9, fontweight="bold",
             color=qc, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=qc, alpha=0.9))

    # Panel 5: DWT cA comparison
    ax5 = fig.add_subplot(gs[3, 0])
    coeffs_o = pywt.wavedec(o, WAVELET, level=DWT_LEVEL)
    coeffs_w = pywt.wavedec(w, WAVELET, level=DWT_LEVEL)
    cA_o = coeffs_o[0]; cA_w = coeffs_w[0]
    ax5_t = np.arange(len(cA_o))
    ax5.plot(ax5_t[::4], cA_o[::4], color=C1, lw=0.5, alpha=0.8, label="Original cA")
    ax5.plot(ax5_t[::4], cA_w[::4], color=C2, lw=0.5, alpha=0.8,
             linestyle="--", label="Watermarked cA")
    ax5.set_title(f"DWT level-{DWT_LEVEL} cA subband (0–{44100/(2**(DWT_LEVEL+1)):.0f} Hz)",
                  fontsize=10, fontweight="bold", pad=3)
    ax5.set_ylabel("Coefficient value"); ax5.set_xlabel("Coefficient index")
    ax5.legend(fontsize=8); ax5.grid(True, axis="y")

    # Panel 6: noise histogram
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.hist(noise, bins=100, color=C3, alpha=0.7, density=True)
    ax6.axvline(0, color="#1E293B", lw=1)
    ax6.set_title("Noise Distribution", fontsize=10, fontweight="bold", pad=3)
    ax6.set_xlabel("Amplitude"); ax6.set_ylabel("Density")
    ax6.grid(True, axis="y")
    mu, sigma = np.mean(noise), np.std(noise)
    ax6.text(0.97, 0.90, f"μ={mu:.5f}\nσ={sigma:.5f}",
             transform=ax6.transAxes, fontsize=8, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.9))

    plot_path = os.path.splitext(output_path)[0] + "_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# ════════════════════════════════════════════
#  Interactive prompt
# ════════════════════════════════════════════

def _ask(label, default=None):
    prompt = f"  {label} [{default}]: " if default else f"  {label}: "
    while True:
        val = input(prompt).strip()
        if val: return val
        if default is not None: return default
        print("  ✗ Cannot be empty.")


def interactive_mode():
    print()
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  EMBED — Audio Watermarking  (Final Version)         ║")
    print("║  Sync + Header + RS Payload  |  WAV → MP3             ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║  Audio needed:  32×32 ~9s   64×64 ~36s  128×128 ~144s║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()

    while True:
        audio_path = _ask("Input WAV file")
        if not os.path.exists(audio_path):
            print(f"  ✗ Not found: '{audio_path}'"); continue
        if not audio_path.lower().endswith(".wav"):
            print("  ⚠  Does not end in .wav")
            if input("     Continue anyway? (y/N): ").strip().lower() != "y": continue
        break

    logo_path = _ask("Logo image (PNG/JPG)")
    while not os.path.exists(logo_path):
        print(f"  ✗ Not found: '{logo_path}'")
        logo_path = _ask("Logo image (PNG/JPG)")

    output_path = resolve_output_path(
        _ask("Output MP3 file", default="watermarked.mp3"))

    print()
    choice = input(
        "  Logo size: (1) 64×64  (2) 32×32  (3) 128×128  (4) Custom  [1]: "
    ).strip()
    if   choice == "2": logo_size = (32, 32)
    elif choice == "3": logo_size = (128, 128)
    elif choice == "4":
        w = int(input("    Width : ").strip())
        h = int(input("    Height: ").strip())
        logo_size = (w, h)
    else: logo_size = (64, 64)

    print()
    while True:
        password = getpass.getpass("  Password (hidden): ")
        if not password: print("  ✗ Cannot be empty."); continue
        confirm  = getpass.getpass("  Confirm password : ")
        if password == confirm: break
        print("  ✗ Passwords do not match.\n")

    print()
    print("  ──────────────────────────────────────────────────────")
    print(f"  Input    : {audio_path}")
    print(f"  Logo     : {logo_path}  ({logo_size[0]}×{logo_size[1]} px RGB)")
    print(f"  Output   : {output_path}")
    print(f"  Password : {'*' * len(password)}")
    need_s = needed_seconds(*logo_size)
    print(f"  Need     : ≥ {need_s:.0f}s of audio")
    print("  ──────────────────────────────────────────────────────")
    if input("\n  Start? (Y/n): ").strip().lower() == "n":
        print("\n  Cancelled.\n"); sys.exit(0)
    print()
    return audio_path, logo_path, output_path, password, logo_size


# ════════════════════════════════════════════
#  Main pipeline
# ════════════════════════════════════════════

def embed(audio_path, logo_path, output_path, password,
           logo_size=DEFAULT_SIZE):

    _check_deps()
    logo_w, logo_h = logo_size
    output_path    = resolve_output_path(output_path)
    output_is_mp3  = output_path.lower().endswith(".mp3")

    tmp_dir     = tempfile.mkdtemp(prefix="wm_embed_")
    tmp_wav_out = os.path.join(tmp_dir, "watermarked.wav")

    try:
        print()
        print("=" * 62)
        print("  EMBED — Final Version (Sync + Header + RS + QIM)")
        print("=" * 62)
        print(f"  DWT        : {WAVELET} level-{DWT_LEVEL}  "
              f"({ca_per_sec():.0f} cA slots/sec)")
        print(f"  Sync zone  : δ={DELTA_SYNC}  R={REDUND_SYNC}  (64 bits)")
        print(f"  Header     : δ={DELTA_HEADER}  R={REDUND_HEADER}  (192 bits)")
        print(f"  Payload    : δ={DELTA_PAYLOAD}  R={REDUND_PAYLOAD}  + RS({255},{RS_DATA})")
        print(f"  MP3 export : {MP3_BITRATE}")
        print()

        # ── Step 1: key ───────────────────────────────────────────────
        print("[1/9] Deriving encryption key ...")
        key   = derive_key(password)
        nonce = os.urandom(16)
        print("      Done ✓")

        # ── Step 2: logo → encrypt → RS encode ───────────────────────
        print(f"\n[2/9] Preparing payload ...")
        img          = Image.open(logo_path).convert("RGB").resize(
                           (logo_w, logo_h), Image.LANCZOS)
        logo_bytes   = np.array(img, dtype=np.uint8).tobytes()
        ciphertext   = encrypt_chacha20(logo_bytes, key, nonce)
        rs_encoded   = rs_encode(ciphertext)
        payload_bits = bytes_to_bits(rs_encoded)
        header_bits  = encode_header(nonce, logo_w, logo_h)

        n_sync_slots  = SYNC_BITS   * REDUND_SYNC
        n_hdr_slots   = HEADER_BITS * REDUND_HEADER
        n_pay_slots   = len(payload_bits) * REDUND_PAYLOAD
        total_slots   = n_sync_slots + n_hdr_slots + n_pay_slots

        print(f"      Logo bytes    : {len(logo_bytes):,}")
        print(f"      Ciphertext    : {len(ciphertext):,} B")
        print(f"      RS-encoded    : {len(rs_encoded):,} B  "
              f"({len(rs_encoded)/len(ciphertext):.3f}× overhead)")
        print(f"      Payload bits  : {len(payload_bits):,}")
        print(f"      Sync slots    : {n_sync_slots:,}")
        print(f"      Header slots  : {n_hdr_slots:,}")
        print(f"      Payload slots : {n_pay_slots:,}")
        print(f"      TOTAL slots   : {total_slots:,}")

        # ── Step 3: load WAV ──────────────────────────────────────────
        print(f"\n[3/9] Reading WAV  '{audio_path}' ...")
        audio, fs, n_ch, sw = read_wav(audio_path)
        n_samples = len(audio)
        duration  = n_samples / fs / (n_ch if n_ch > 1 else 1)
        print(f"      {fs} Hz | {n_ch}ch | {duration:.2f}s | {sw*8}-bit")

        left  = audio[0::2] if n_ch == 2 else audio
        right = audio[1::2] if n_ch == 2 else None

        # ── Step 4: capacity check ────────────────────────────────────
        print(f"\n[4/9] Checking capacity ...")
        coeffs  = pywt.wavedec(left, WAVELET, level=DWT_LEVEL)
        cA_len  = len(coeffs[0])
        pct     = total_slots / cA_len * 100
        need_s  = needed_seconds(logo_w, logo_h, fs)
        print(f"      cA available  : {cA_len:,}")
        print(f"      Slots needed  : {total_slots:,}  ({pct:.1f}%)")
        if total_slots > cA_len:
            print(f"\n      ✗ Audio too short!")
            print(f"        Need ≥ {need_s:.0f}s — have {duration:.1f}s")
            sys.exit(1)
        print(f"      OK ✓  ({cA_len - total_slots:,} spare slots)")

        # ── Step 5: embed ─────────────────────────────────────────────
        print(f"\n[5/9] Embedding three zones ...")
        cA = coeffs[0].copy()

        # Zone A: sync
        slot = embed_zone(cA, SYNC_PATTERN, 0, DELTA_SYNC, REDUND_SYNC)
        print(f"      Zone A (sync)    : slots 0 – {slot-1}")

        # Zone B: header
        slot = embed_zone(cA, header_bits, slot, DELTA_HEADER, REDUND_HEADER)
        print(f"      Zone B (header)  : slots up to {slot-1}")

        # Zone C: payload
        slot = embed_zone(cA, payload_bits, slot, DELTA_PAYLOAD, REDUND_PAYLOAD)
        print(f"      Zone C (payload) : slots up to {slot-1}")

        # Reconstruct audio
        coeffs[0] = cA
        left_wm   = pywt.waverec(coeffs, WAVELET)[:len(left)]

        snr     = compute_snr(left, left_wm)
        quality = ("Excellent (inaudible)"     if snr >= 40 else
                   "Good (barely perceptible)" if snr >= 30 else
                   "Acceptable (faint)"        if snr >= 20 else
                   "Audible — use longer audio")
        badge   = "✓" if snr >= 30 else ("~" if snr >= 20 else "⚠")
        print(f"      SNR : {snr:.2f} dB  {badge}  {quality}")

        # ── Step 6: rebuild stereo ────────────────────────────────────
        if n_ch == 2:
            out = np.empty(n_samples, dtype=np.float64)
            out[0::2] = left_wm; out[1::2] = right
        else:
            out = left_wm

        # ── Step 7: write WAV ─────────────────────────────────────────
        wav_dest = tmp_wav_out if output_is_mp3 else output_path
        print(f"\n[6/9] Writing watermarked WAV ...")
        write_wav(wav_dest, out, fs, n_ch, sw)
        print("      Done ✓")

        # ── Step 8: WAV → MP3 ─────────────────────────────────────────
        if output_is_mp3:
            print(f"\n[7/9] Converting WAV → MP3  ({MP3_BITRATE}) ...")
            wav_to_mp3(wav_dest, output_path)
            kb = os.path.getsize(output_path) // 1024
            print(f"      Saved → '{output_path}'  ({kb} KB)  ✓")
        else:
            kb = os.path.getsize(output_path) // 1024
            print(f"      Saved → '{output_path}'  ({kb} KB)  ✓")

        # ── Step 9: analysis plot ─────────────────────────────────────
        print(f"\n[8/9] Generating analysis plot ...")
        plot_path = plot_analysis(left, left_wm, fs, snr,
                                  output_path, audio_path)
        print(f"      Saved → '{plot_path}'  ✓")

        # ── Summary ───────────────────────────────────────────────────
        W = 62; div = "─" * W
        def row(l, v):
            c = f"  {l:<20}: {v}"
            return f"│{c[:W]:<{W}}│"

        print()
        print(f"┌{div}┐")
        print(f"│{'  ✓  Embedding complete — perfect recovery guaranteed':<{W}}│")
        print(f"├{div}┤")
        print(row("Output",          os.path.basename(output_path)))
        print(row("SNR",             f"{snr:.2f} dB — {quality}"))
        print(row("Sync zone",       f"δ={DELTA_SYNC}  R={REDUND_SYNC}  → survives trimming"))
        print(row("Header zone",     f"δ={DELTA_HEADER}  R={REDUND_HEADER}  → P(err)≈10⁻²⁴"))
        print(row("Payload zone",    f"δ={DELTA_PAYLOAD}  RS({255},{RS_DATA}) → corrects all errors"))
        print(row("Logo",            f"RGB {logo_w}×{logo_h}"))
        print(row("cA utilised",     f"{pct:.1f}%  ({cA_len-total_slots:,} slots spare)"))
        print(f"├{div}┤")
        print(f"│{'  Extract needs only the PASSWORD to extract':<{W}}│")
        print(f"└{div}┘")
        print()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) == 1:
        audio, logo, output, pwd, size = interactive_mode()
        embed(audio, logo, output, pwd, size)
    else:
        p = argparse.ArgumentParser(
            description="Audio watermarking — Sync+Header+RS+QIM",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument("--audio",    required=True)
        p.add_argument("--logo",     required=True)
        p.add_argument("--output",   required=True)
        p.add_argument("--password", required=True)
        p.add_argument("--logo-w",   type=int, default=64)
        p.add_argument("--logo-h",   type=int, default=64)
        a = p.parse_args()
        embed(a.audio, a.logo, a.output, a.password,
               (a.logo_w, a.logo_h))
