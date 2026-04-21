# 🎧 AudioMark

### 🔐 Cryptographic Audio Watermarking System

<p align="center">

**Invisibly embed encrypted ownership logos inside audio files.**
**Protect your work. Prove ownership. Stay secure.**

</p>

---

## 🚀 Overview

**AudioMark** is an advanced **cryptographic audio watermarking system** that embeds an **encrypted logo** invisibly into an audio file using:

* 🎚️ **Digital Signal Processing (DWT)**
* 🔒 **Strong Cryptography (ChaCha20)**
* 🛡️ **Error Correction (Reed–Solomon)**

The watermark:

✅ Survives MP3 compression
✅ Cannot be detected by human hearing
✅ Can only be extracted using the correct password

Perfect for creators, journalists, and organizations who need **tamper-resistant ownership proof**.

---

## ❗ The Problem

When audio files are shared:

* Metadata can be **removed instantly**
* Ownership proof gets **lost**
* Audio piracy becomes **easy**

Traditional metadata-based protection is:

❌ Removable
❌ Weak
❌ Unreliable

---

## 💡 The Solution

**AudioMark embeds ownership proof directly inside the audio signal**, making it:

✔ Invisible
✔ Encrypted
✔ Tamper-resistant
✔ Persistent

---

# ⚙️ How It Works

```
Input Audio (WAV)
        │
        ▼
🔷 DWT Transform
   (Daubechies-4, Level 3)
        │
        ▼
🔒 Logo Encryption
   ChaCha20 + PBKDF2
        │
        ▼
🛡 Reed-Solomon Encoding
   RS(255,215)
        │
        ▼
📡 3-Zone QIM Embedding
   Sync | Header | Payload
        │
        ▼
🎧 Inverse DWT
        │
        ▼
Watermarked Audio (MP3)
```

---

# ✨ Features

| Feature                | Description               |
| ---------------------- | ------------------------- |
| 🔇 Inaudible Watermark | SNR ≥ **21 dB**           |
| 🎵 MP3 Resistant       | Survives compression      |
| 🔐 Password Protected  | ChaCha20 encryption       |
| 🛡 Error Correction    | Reed-Solomon RS(255,215)  |
| 📡 Sync Protection     | Survives trimming         |
| 🧪 Quality Metrics     | NCC & PSNR evaluation     |
| 🌐 Web Interface       | Drag & drop UI            |
| 📊 Visualization       | Waveform & noise analysis |

---

# 🧰 Tech Stack

| Layer            | Technology                    |
| ---------------- | ----------------------------- |
| Transform        | PyWavelets (db4 DWT Level 3)  |
| Embedding        | Quantisation Index Modulation |
| Error Correction | Reed-Solomon                  |
| Encryption       | ChaCha20                      |
| Key Derivation   | PBKDF2-HMAC-SHA256            |
| Backend          | Flask                         |
| Frontend         | HTML/CSS/JS                   |
| Audio Processing | pydub, ffmpeg                 |
| Visualization    | Matplotlib                    |

---

# 📁 Project Structure

```
audiomark/
│
├── embed.py        # Watermark embedding pipeline
├── extract.py      # Extraction & reconstruction
├── app.py          # Flask backend
└── index.html      # Web UI
```

---

# 🛠 Installation

## Prerequisites

* Python **3.8+**
* ffmpeg installed

### Install ffmpeg

**Windows**

Download from:
https://ffmpeg.org

Add to PATH.

**Linux**

```bash
sudo apt install ffmpeg
```

**Mac**

```bash
brew install ffmpeg
```

---

## Install Dependencies

```bash
pip install flask flask-cors reedsolo pydub pywavelets numpy Pillow cryptography matplotlib
```

---

## Run the Server

```bash
python app.py
```

Open:

```
http://localhost:5000
```

---

# 🖥 Usage

## 🌐 Web Interface (Recommended)

### 🔹 Embed Watermark

1. Open browser
2. Select **Embed**
3. Upload:

   * WAV Audio
   * Logo Image
4. Enter password
5. Click **Embed**

Download → Watermarked MP3

---

### 🔹 Extract Watermark

1. Select **Extract**
2. Upload watermarked audio
3. Enter password
4. Click **Extract**

Download → Recovered logo

---

# 💻 Command Line Usage

## Embed

```bash
python embed.py \
 --audio input.wav \
 --logo logo.png \
 --output watermarked.mp3 \
 --password yourpassword \
 --logo-w 64 \
 --logo-h 64
```

---

## Extract

```bash
python extract.py \
 --audio watermarked.mp3 \
 --password yourpassword \
 --output recovered_logo.png \
 --orig-output original.wav
```

---

## Compare Logo Quality

```bash
python extract.py \
 --audio watermarked.mp3 \
 --password yourpassword \
 --output recovered.png \
 --compare original_logo.png
```

---

# ⏱ Audio Length Requirements

| Logo Size | Minimum Audio |
| --------- | ------------- |
| 32 × 32   | ~9 seconds    |
| 64 × 64   | ~36 seconds   |
| 128 × 128 | ~144 seconds  |

---

# 📊 Watermark Quality

| Metric           | Value          |
| ---------------- | -------------- |
| SNR              | ≥ 21 dB        |
| RS Overhead      | 18.6%          |
| Error Correction | 20 bytes/block |

---

# 🔐 Security Model

| Attack                     | Result        |
| -------------------------- | ------------- |
| Remove metadata            | ❌ Fails       |
| MP3 re-encode              | ❌ Fails       |
| Brute-force password       | ❌ Impractical |
| Read logo without password | ❌ Impossible  |
| Trim audio                 | ❌ Fails       |

---

# 🎯 Use Cases

🎵 **Music Creators**
Protect ownership of original tracks.

🎙 **Podcast Producers**
Trace leaked content.

📰 **Journalists**
Verify authenticity in legal cases.

🏢 **Enterprises**
Track confidential audio leaks.

🔬 **Forensic Audio**
Authenticate evidence.

---

# ⭐ If You Like This Project

Give it a **star ⭐** on GitHub!
