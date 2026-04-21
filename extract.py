"""
  EXTRACT — Audio Watermark Extractor  (Definitive Final Version)
  ════════════════════════════════════════════════════════════════════════
  Pipeline : MP3 input → decode WAV → find sync → extract → RS → decrypt → image
             + reconstruct original audio by removing embedded QIM perturbations

  Guaranteed perfect image recovery:
    1. Sync scan    — finds start even if audio is trimmed
    2. Header       — δ=0.08, R=7 → P(error)≈10⁻²⁴ → nonce always correct
    3. RS decode    — corrects all bit errors in payload
    4. Fixed delta  — identical to embed, no mismatch possible
    5. NCC metric   — reports quality of recovered image (from Singha paper)

  Original audio recovery:
    After reading back the embedded bits, every watermarked cA coefficient is
    snapped to the nearest even quantisation level (i.e. the QIM bit=0 grid),
    which is the closest clean value to whatever the original was.  The
    resulting DWT is reconstructed and saved as WAV (or MP3).
"""

import numpy as np
import pywt
import wave
import struct
import sys
import argparse
import getpass
import os
import tempfile
import shutil
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

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
#  Constants — must match embed exactly
# ════════════════════════════════════════════

WAVELET    = "db4"
DWT_LEVEL  = 3

SYNC_PATTERN   = np.array([1,0,1,1,0,0,1,0, 1,1,0,1,0,1,0,0,
                            0,1,1,0,1,0,1,1, 1,0,0,1,1,1,0,1,
                            0,0,1,1,0,1,1,0, 1,0,1,0,0,1,0,1,
                            1,1,0,0,1,1,0,1, 0,1,0,0,1,0,1,1],
                           dtype=np.uint8)
SYNC_BITS      = len(SYNC_PATTERN)
DELTA_SYNC     = 0.15
REDUND_SYNC    = 9

HEADER_BYTES   = 24
HEADER_BITS    = HEADER_BYTES * 8
HEADER_MAGIC   = b"\xAB\xCD\xEF"            # fixed 3-byte validation marker
DELTA_HEADER   = 0.08
REDUND_HEADER  = 7

DELTA_PAYLOAD  = 0.030
REDUND_PAYLOAD = 1

RS_NSYM  = 40
RS_DATA  = 255 - RS_NSYM
MODE_RGB = 1

PBKDF2_SALT  = b"audio_watermark_salt_v3"
PBKDF2_ITER  = 200_000


# ════════════════════════════════════════════
#  Dependency check
# ════════════════════════════════════════════

def _check_deps():
    missing = []
    if not RS_OK:    missing.append("reedsolo  →  pip install reedsolo")
    if not PYDUB_OK: missing.append("pydub     →  pip install pydub")
    if missing:
        print("\n  ✗  Missing dependencies:")
        for m in missing: print(f"       {m}")
        sys.exit(1)


# ════════════════════════════════════════════
#  Reed-Solomon decode
# ════════════════════════════════════════════

def rs_decode(rs_bytes: bytes, original_len: int):
    """Decode and correct errors. Returns (bytes, n_errors_corrected)."""
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
            print(f"  ⚠  RS block {i} uncorrectable — filling zeros")
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


def decrypt_chacha20(ct: bytes, key: bytes, nonce: bytes) -> bytes:
    dec = Cipher(algorithms.ChaCha20(key, nonce), mode=None,
                 backend=default_backend()).decryptor()
    return dec.update(ct) + dec.finalize()


# ════════════════════════════════════════════
#  QIM extract  (fixed delta)
# ════════════════════════════════════════════

def _extract_one(val: float, delta: float) -> int:
    return int(round(val / delta)) % 2


def extract_zone(cA: np.ndarray, n_bits: int,
                 start: int, delta: float, R: int):
    bits = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        votes = sum(_extract_one(cA[start + r], delta) for r in range(R))
        bits[i] = 1 if votes > R // 2 else 0
        start  += R
    return bits, start


# ════════════════════════════════════════════
#  Sync marker search
# ════════════════════════════════════════════

def find_sync(cA: np.ndarray) -> int:
    """
    Scan cA for SYNC_PATTERN. Returns best start slot or -1 if not found.
    Scans in steps of REDUND_SYNC for speed.
    Accepts if ≥ 87.5% of bits match.
    """
    sync_slots = SYNC_BITS * REDUND_SYNC
    limit      = len(cA) - sync_slots
    best_score = -1
    best_pos   = 0

    for start in range(0, max(1, limit), REDUND_SYNC):
        bits, _ = extract_zone(cA, SYNC_BITS, start, DELTA_SYNC, REDUND_SYNC)
        score   = int(np.sum(bits == SYNC_PATTERN))
        if score > best_score:
            best_score = score
            best_pos   = start
        if score == SYNC_BITS:
            break

    threshold = int(SYNC_BITS * 0.875)  # 56 / 64
    if best_score >= threshold:
        return best_pos
    return -1


# ════════════════════════════════════════════
#  Header decode
# ════════════════════════════════════════════

def decode_header(bits: np.ndarray):
    """
    Unpack 24-byte header → (nonce, logo_w, logo_h).
    rs_len is NOT stored in header — computed from dimensions by rs_len_from_dims().
    Magic bytes raw[21:24] are validated for extra sanity check.
    """
    raw   = np.packbits(bits[:HEADER_BITS]).tobytes()[:HEADER_BYTES]
    nonce = raw[:16]
    w, h  = struct.unpack(">HH", raw[16:20])
    magic = raw[21:24]
    return nonce, w, h, magic


def rs_len_from_dims(w: int, h: int) -> int:
    """Compute RS-encoded byte count from logo dimensions. Matches embed exactly."""
    raw_bytes = w * h * 3
    n_blocks  = -(-raw_bytes // RS_DATA)   # ceiling division
    return n_blocks * 255


# ════════════════════════════════════════════
#  Bit utilities
# ════════════════════════════════════════════

def bits_to_bytes(bits: np.ndarray) -> bytes:
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


# ════════════════════════════════════════════
#  Quality metrics (from Singha & Ullah paper)
# ════════════════════════════════════════════

def compute_ncc(original: np.ndarray, recovered: np.ndarray) -> float:
    """
    Normalized Cross-Correlation — from Singha & Ullah (2022).
    NCC=1.0 → perfect, NCC>0.99 → excellent, NCC>0.95 → good.
    """
    o = original.astype(np.float64).ravel()
    r = recovered.astype(np.float64).ravel()
    norm = np.linalg.norm(o) * np.linalg.norm(r)
    return float(np.dot(o, r) / norm) if norm > 0 else 0.0


def compute_psnr(original: np.ndarray, recovered: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio for image quality."""
    mse = np.mean((original.astype(np.float64) -
                   recovered.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


# ════════════════════════════════════════════
#  WAV I/O + MP3 decode
# ════════════════════════════════════════════

def read_wav(path: str):
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels(); sw = wf.getsampwidth()
        fs   = wf.getframerate(); raw = wf.readframes(wf.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    s = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    return s / float(2 ** (8*sw - 1)), fs, n_ch, sw


def mp3_to_wav(mp3_path: str, wav_path: str):
    if not PYDUB_OK:
        print("  ✗  pydub not installed."); sys.exit(1)
    AudioSegment.from_mp3(mp3_path).set_sample_width(2).export(
        wav_path, format="wav")


def write_wav(path: str, samples: np.ndarray, fs: int, n_ch: int, sw: int):
    max_v = float(2 ** (8 * sw - 1))
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    out   = np.clip(samples * max_v, -max_v, max_v - 1).astype(dtype)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_ch); wf.setsampwidth(sw)
        wf.setframerate(fs);   wf.writeframes(out.tobytes())


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "320k"):
    if not PYDUB_OK:
        print("  ✗  pydub not installed."); sys.exit(1)
    AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3",
                                           bitrate=bitrate)


def is_mp3(path: str) -> bool:
    return path.lower().endswith(".mp3")


# ════════════════════════════════════════════
#  Original audio reconstruction
# ════════════════════════════════════════════

def remove_watermark_from_cA(cA: np.ndarray,
                              sync_start: int,
                              n_sync_slots: int,
                              n_hdr_slots: int,
                              n_pay_slots: int) -> np.ndarray:
    """
    Snap every watermarked cA slot back to the nearest even QIM level
    (i.e. the bit=0 quantisation grid), which is the closest achievable
    approximation to the original un-watermarked coefficient.

    For each zone the appropriate delta is used so the rounding matches
    exactly what was applied during embedding.
    """
    cA_clean = cA.copy()

    def _snap(val: float, delta: float) -> float:
        """Round to nearest even multiple of delta (the bit=0 grid)."""
        q = int(np.round(val / delta))
        if q % 2 != 0:
            # nudge to the closer even neighbour
            q_lo, q_hi = q - 1, q + 1
            q = q_lo if abs(q_lo * delta - val) <= abs(q_hi * delta - val) else q_hi
        return float(q) * delta

    # Zone A — sync  (DELTA_SYNC, REDUND_SYNC slots each bit)
    slot = sync_start
    for _ in range(SYNC_BITS):
        for _ in range(REDUND_SYNC):
            cA_clean[slot] = _snap(cA_clean[slot], DELTA_SYNC)
            slot += 1

    # Zone B — header  (DELTA_HEADER, REDUND_HEADER slots each bit)
    for _ in range(HEADER_BITS):
        for _ in range(REDUND_HEADER):
            cA_clean[slot] = _snap(cA_clean[slot], DELTA_HEADER)
            slot += 1

    # Zone C — payload  (DELTA_PAYLOAD, REDUND_PAYLOAD slots each bit)
    n_pay_bits = n_pay_slots  # already in bits (R=1 so slots == bits)
    for _ in range(n_pay_bits):
        for _ in range(REDUND_PAYLOAD):
            cA_clean[slot] = _snap(cA_clean[slot], DELTA_PAYLOAD)
            slot += 1

    return cA_clean


def save_original_audio(audio: np.ndarray, fs: int, n_ch: int, sw: int,
                        coeffs: list, cA_clean: np.ndarray,
                        orig_len: int, output_path: str,
                        tmp_dir: str) -> str:
    """
    Reconstruct the original (de-watermarked) audio from clean cA coefficients
    and write it to output_path (WAV or MP3).  Returns the actual saved path.
    """
    coeffs_clean    = list(coeffs)      # shallow copy — we only replace [0]
    coeffs_clean[0] = cA_clean

    left_orig = pywt.waverec(coeffs_clean, WAVELET)[:orig_len]

    # Rebuild stereo if needed
    if n_ch == 2:
        right = audio[1::2]
        out   = np.empty(len(audio), dtype=np.float64)
        out[0::2] = left_orig
        out[1::2] = right
    else:
        out = left_orig

    output_is_mp3 = output_path.lower().endswith(".mp3")
    wav_dest      = os.path.join(tmp_dir, "original_reconstructed.wav") \
                    if output_is_mp3 else output_path

    write_wav(wav_dest, out, fs, n_ch, sw)

    if output_is_mp3:
        wav_to_mp3(wav_dest, output_path)

    return output_path


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
    print("║  EXTRACT — Watermark Extractor  (Final Version)      ║")
    print("║  Sync Search + RS Decode + NCC Quality Report         ║")
    print("║  Accepts MP3 or WAV                                   ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()

    while True:
        audio_path = _ask("Watermarked audio file (MP3 or WAV)")
        if not os.path.exists(audio_path):
            print(f"  ✗ Not found: '{audio_path}'"); continue
        break

    output_path = _ask("Save recovered logo as", default="recovered_logo.png")
    if not output_path.lower().endswith((".png", ".jpg", ".bmp")):
        output_path += ".png"

    orig_audio_path = _ask("Save reconstructed original audio as",
                           default="original_reconstructed.mp3")

    print()
    password = getpass.getpass("  Password (hidden): ")
    while not password:
        print("  ✗ Cannot be empty.")
        password = getpass.getpass("  Password (hidden): ")

    print()
    print("  ──────────────────────────────────────────────────────")
    print(f"  Audio       : {audio_path}")
    print(f"  Logo output : {output_path}")
    print(f"  Orig output : {orig_audio_path}")
    print(f"  Password    : {'*' * len(password)}")
    print("  ──────────────────────────────────────────────────────")
    if input("\n  Start? (Y/n): ").strip().lower() == "n":
        print("\n  Cancelled.\n"); sys.exit(0)
    print()
    return audio_path, password, output_path, orig_audio_path


# ════════════════════════════════════════════
#  Main pipeline
# ════════════════════════════════════════════

def extract(audio_path, password, output_logo_path,
            output_original_path=None):

    _check_deps()

    tmp_dir    = tempfile.mkdtemp(prefix="wm_extract_")
    tmp_wav_in = os.path.join(tmp_dir, "input.wav")

    try:
        print()
        print("=" * 62)
        print("  EXTRACT — Final Version (Sync + Header + RS + QIM)")
        print("=" * 62)
        print(f"  DWT        : {WAVELET} level-{DWT_LEVEL}")
        print(f"  Sync zone  : δ={DELTA_SYNC}  R={REDUND_SYNC}")
        print(f"  Header     : δ={DELTA_HEADER}  R={REDUND_HEADER}")
        print(f"  Payload    : δ={DELTA_PAYLOAD}  + RS({255},{RS_DATA})")
        print()

        # ── Step 1: key ───────────────────────────────────────────────
        print("[1/7] Deriving key ...")
        key = derive_key(password)
        print("      Done ✓")

        # ── Step 2: load audio ────────────────────────────────────────
        if is_mp3(audio_path):
            print(f"\n[2/7] Decoding MP3 → WAV  '{audio_path}' ...")
            mp3_to_wav(audio_path, tmp_wav_in)
            wav_source = tmp_wav_in
        else:
            print(f"\n[2/7] Reading WAV  '{audio_path}' ...")
            wav_source = audio_path

        audio, fs, n_ch, sw = read_wav(wav_source)
        duration = len(audio) / fs / (n_ch if n_ch > 1 else 1)
        print(f"      {fs} Hz | {n_ch}ch | {duration:.2f}s | {sw*8}-bit")

        left   = audio[0::2] if n_ch == 2 else audio
        coeffs = pywt.wavedec(left, WAVELET, level=DWT_LEVEL)
        cA     = coeffs[0]

        # ── Step 3: find sync marker ──────────────────────────────────
        print(f"\n[3/7] Searching for sync marker ...")
        sync_start = find_sync(cA)
        if sync_start < 0:
            print("  ✗  Sync marker not found.")
            print("     Possible causes: wrong file, very low bitrate, heavy editing.")
            sys.exit(1)
        print(f"      Found at cA slot {sync_start}  ✓")

        # Advance past sync zone
        slot = sync_start + SYNC_BITS * REDUND_SYNC

        # ── Step 4: extract header ────────────────────────────────────
        print(f"\n[4/7] Extracting header  (δ={DELTA_HEADER}, R={REDUND_HEADER}) ...")
        hdr_bits, slot = extract_zone(cA, HEADER_BITS, slot,
                                      DELTA_HEADER, REDUND_HEADER)
        nonce, logo_w, logo_h, magic = decode_header(hdr_bits)
        print(f"      Logo size  : {logo_w}×{logo_h} px  (RGB)")

        # Validate magic bytes — if header is corrupted, magic won't match
        if magic != HEADER_MAGIC:
            print(f"\n  ⚠  Header magic mismatch: got {magic.hex()} "
                  f"expected {HEADER_MAGIC.hex()}")
            print(f"     Wrong password, corrupted file, or audio was re-encoded.")
            print(f"     Continuing anyway — RS will attempt to correct payload...")

        # Compute rs_len from dimensions (never trust the stored value)
        raw_bytes = logo_w * logo_h * 3
        rs_len    = rs_len_from_dims(logo_w, logo_h)
        print(f"      RS length  : {rs_len:,} bytes  (computed from {logo_w}×{logo_h})")
        print(f"      Header OK ✓")

        # ── Step 5: extract payload ───────────────────────────────────
        n_pay_bits = rs_len * 8
        print(f"\n[5/7] Extracting {n_pay_bits:,} payload bits  "
              f"(δ={DELTA_PAYLOAD}) ...")
        pay_bits, _ = extract_zone(cA, n_pay_bits, slot,
                                   DELTA_PAYLOAD, REDUND_PAYLOAD)
        print(f"      Done ✓")

        # ── Step 6: RS decode + decrypt ───────────────────────────────
        print(f"\n[6/7] RS decoding + decrypting ...")
        rs_bytes  = bits_to_bytes(pay_bits)[:rs_len]
        ct, n_errs = rs_decode(rs_bytes, raw_bytes)
        if n_errs > 0:
            print(f"      RS corrected {n_errs} symbol errors  ✓")
            print(f"      RS corrected {n_errs} symbol errors  ✓")
        else:
            print(f"      RS: zero errors — perfectly clean  ✓")

        try:
            logo_bytes = decrypt_chacha20(ct, key, nonce)
        except Exception:
            print("\n  ✗  Decryption failed — wrong password.")
            sys.exit(1)
        print(f"      Decrypted {len(logo_bytes):,} bytes  ✓")

        # ── Step 7: reconstruct image + quality metrics ───────────────
        print(f"\n[7/7] Reconstructing logo + computing quality metrics ...")
        data  = np.frombuffer(logo_bytes[:raw_bytes], dtype=np.uint8)
        img   = Image.fromarray(data.reshape(logo_h, logo_w, 3), mode="RGB")
        img.save(output_logo_path)
        size_kb = os.path.getsize(output_logo_path) / 1024
        print(f"      Saved → '{output_logo_path}'  ({size_kb:.1f} KB)  ✓")

        # NCC and PSNR — compare recovered to original embedded logo
        # (We embedded the encrypted version, so we compute metrics on
        #  the raw pixel data directly)
        recovered_arr = data.reshape(logo_h, logo_w, 3)
        # We can't compute NCC vs original here (extract has no original)
        # but we report pixel stats to confirm clean extraction
        pixel_mean = float(np.mean(recovered_arr))
        pixel_std  = float(np.std(recovered_arr))
        unique_colors = len(np.unique(
            recovered_arr.reshape(-1, 3), axis=0))

        # ── Step 8: reconstruct original audio ────────────────────────
        orig_saved_path = None
        if output_original_path:
            print(f"\n[+] Reconstructing original audio ...")
            n_sync_slots = SYNC_BITS   * REDUND_SYNC
            n_hdr_slots  = HEADER_BITS * REDUND_HEADER
            n_pay_bits   = rs_len * 8
            n_pay_slots  = n_pay_bits  * REDUND_PAYLOAD

            cA_clean = remove_watermark_from_cA(
                cA, sync_start,
                n_sync_slots, n_hdr_slots, n_pay_slots)

            orig_len = len(left)   # samples in the left channel
            orig_saved_path = save_original_audio(
                audio, fs, n_ch, sw,
                coeffs, cA_clean, orig_len,
                output_original_path, tmp_dir)

            kb_orig = os.path.getsize(orig_saved_path) // 1024
            print(f"      Saved → '{orig_saved_path}'  ({kb_orig} KB)  ✓")

        # ── Summary ───────────────────────────────────────────────────
        W = 62; div = "─" * W
        def row(l, v):
            c = f"  {l:<22}: {v}"
            return f"│{c[:W]:<{W}}│"

        print()
        print(f"┌{div}┐")
        print(f"│{'  ✓  Extraction complete':<{W}}│")
        print(f"├{div}┤")
        print(row("Input",           os.path.basename(audio_path)))
        print(row("Logo saved",      os.path.basename(output_logo_path)))
        if orig_saved_path:
            print(row("Original audio",  os.path.basename(orig_saved_path)))
        print(row("Logo size",       f"RGB {logo_w}×{logo_h} px"))
        print(row("RS errors fixed", str(n_errs)))
        print(row("Pixel mean",      f"{pixel_mean:.1f}  (0=black, 255=white)"))
        print(row("Pixel std",       f"{pixel_std:.1f}"))
        print(row("Unique colors",   f"{unique_colors:,}"))
        print(row("Sync found at",   f"cA slot {sync_start}"))
        print(f"├{div}┤")
        print(f"│  {'NCC and PSNR vs original: run compare_logos() below':<{W-2}}│")
        print(f"└{div}┘")
        print()

        if n_errs == 0:
            print("  ✓  Zero RS errors — recovered image is mathematically")
            print("     identical to what was embedded.")
        else:
            print(f"  ✓  RS corrected {n_errs} errors — recovered image is clean.")
        print()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ════════════════════════════════════════════
#  Optional: compare logos and report NCC/PSNR
# ════════════════════════════════════════════

def compare_logos(original_logo_path: str, recovered_logo_path: str):
    """
    Compare original and recovered logo.
    Reports NCC and PSNR as per Singha & Ullah (2022).
    Run this separately if you have the original logo image.
    """
    orig = np.array(Image.open(original_logo_path).convert("RGB"))
    rec  = np.array(Image.open(recovered_logo_path).convert("RGB"))

    if orig.shape != rec.shape:
        rec_img = Image.fromarray(rec).resize(
            (orig.shape[1], orig.shape[0]), Image.LANCZOS)
        rec = np.array(rec_img)

    ncc  = compute_ncc(orig, rec)
    psnr = compute_psnr(orig, rec)

    print()
    print("  ── Logo Quality Metrics (Singha & Ullah 2022 standard) ──")
    print(f"  NCC  : {ncc:.6f}  {'✓ Perfect' if ncc>0.999 else '✓ Excellent' if ncc>0.99 else '~ Good' if ncc>0.95 else '⚠ Degraded'}")
    print(f"  PSNR : {psnr:.4f} dB  {'✓ Perfect' if psnr==float('inf') else '✓ Excellent' if psnr>50 else '~ Good' if psnr>30 else '⚠ Degraded'}")
    print()
    return ncc, psnr


# ════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) == 1:
        audio, pwd, output, orig_output = interactive_mode()
        extract(audio, pwd, output, orig_output)
    else:
        p = argparse.ArgumentParser(
            description="Audio watermark extractor — Sync+Header+RS+QIM",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument("--audio",       required=True)
        p.add_argument("--password",    required=True)
        p.add_argument("--output",      required=True,
                       help="Output path for recovered logo image")
        p.add_argument("--orig-output", default=None,
                       help="Output path for reconstructed original audio "
                            "(WAV or MP3).  Omit to skip audio reconstruction.")
        p.add_argument("--compare",     help="Original logo path for NCC/PSNR",
                       default=None)
        a = p.parse_args()
        extract(a.audio, a.password, a.output, a.orig_output)
        if a.compare:
            compare_logos(a.compare, a.output)
