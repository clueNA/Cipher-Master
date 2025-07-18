# 🔐 Cipher Master

A comprehensive cryptography application built with Streamlit that implements 14 different ciphers and encoding methods, featuring interactive audio generation, timing visualizations, and detailed educational content.

![Python](https://img.shields.io/badge/python-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Last Updated](https://img.shields.io/badge/last%20updated-July%202025-brightgreen.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Supported Ciphers](#-supported-ciphers)  
- [Installation](#-installation)
- [Usage](#-usage)
- [Cipher Details](#-cipher-details)
- [Audio Features](#-audio-features)
- [Educational Content](#-educational-content)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **14 Different Ciphers** - From ancient to modern cryptographic methods
- **Interactive Audio Generation** - Morse code with adjustable WPM and frequency
- **Visual Timing Diagrams** - Matplotlib-powered Morse code visualization
- **Educational Explanations** - Learn how each cipher works
- **Real-time Conversion** - Instant encryption/decryption
- **Audio Download** - Export Morse code as WAV files
- **Responsive Design** - Clean, intuitive web interface
- **Error Handling** - Graceful handling of invalid inputs

## 🔐 Supported Ciphers

### Historical Ciphers (6)
- **📡 Morse Code** - Telegraph communication with audio playback
- **🏛️ Caesar Cipher** - Classic Roman shift cipher
- **⬜ Playfair Cipher** - Victorian-era digraph substitution
- **📐 Polybius Square** - Ancient Greek coordinate cipher
- **🐷 Pigpen Cipher** - Masonic symbolic cipher
- **🔄 Atbash Cipher** - Biblical alphabet reversal

### Modern Encodings (5)
- **🔣 Hexadecimal** - Base-16 representation
- **🌐 URL Encoding** - Percent encoding for web
- **📊 Base64** - Standard data encoding
- **💻 Binary** - 8-bit binary representation
- **🔄 ROT13** - Simple letter rotation

### Advanced Ciphers (3)
- **🔑 Vigenère Cipher** - Polyalphabetic keyword cipher
- **🚂 Rail Fence Cipher** - Zigzag transposition
- **🔢 A1Z26 Cipher** - Simple number substitution

## 🚀 Installation

### Prerequisites
- Python
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/clueNA/Cipher-Master
   cd Cipher-Master
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

5. **Start converting!**

### Dependencies
```
streamlit
numpy
scipy
matplotlib
```

## 🎯 Usage

### Basic Operation

1. **Select a Cipher** - Choose from the sidebar dropdown
2. **Enter Text** - Type your message in the input area
3. **Convert** - Click the encrypt/decrypt button
4. **View Results** - See the converted output instantly

### Morse Code Special Features

- **Audio Settings** - Adjust WPM (5-40) and frequency (300-1000 Hz)
- **Audio Playback** - Listen to your Morse code
- **Download Audio** - Save as WAV file
- **Timing Diagram** - Visual representation of dots, dashes, and gaps

### Example Workflows

**Encrypt with Vigenère:**
```
Input: "HELLO WORLD"
Keyword: "CIPHER"
Output: "JINQS HMRWP"
```

**Generate Morse Audio:**
```
Input: "SOS"
Output: "... --- ..." + Audio file
```

## 📚 Cipher Details

### Self-Inverse Ciphers
Special ciphers that are their own inverse operation:

- **Atbash** - A↔Z, B↔Y, etc.
- **ROT13** - 13-position rotation

### Historical Significance

- **Polybius Square** - Used in ancient Greece (~150 BC)
- **Caesar Cipher** - Named after Julius Caesar
- **Playfair Cipher** - Used in WWI and WWII
- **Morse Code** - Revolutionized long-distance communication

### Modern Applications

- **Base64** - Email attachments, web data
- **URL Encoding** - Web form submissions
- **Hexadecimal** - Programming and debugging

## 🎵 Audio Features

### Morse Code Audio Generation

- **Standard Timing** - ITU-T recommendations
- **Adjustable Speed** - 5-40 WPM range
- **Custom Frequency** - 300-1000 Hz tones
- **Professional Quality** - 44.1kHz WAV output

### Timing Specifications

- Dot: 1 unit
- Dash: 3 units
- Symbol gap: 1 unit
- Letter gap: 3 units
- Word gap: 7 units

## 📖 Educational Content

### Interactive Learning

- **How-to Guides** - Step-by-step cipher explanations
- **Visual Examples** - Rail fence patterns, Playfair grids
- **Reference Tables** - Quick lookup for all ciphers
- **Historical Context** - Learn the origins and uses

### Self-Inverse Explanation

Understanding why some ciphers decode themselves:

- Mathematical properties
- Practical applications
- Historical examples


## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Adding New Ciphers

1. **Create cipher functions** - Follow existing patterns
2. **Add to main interface** - Update the selectbox
3. **Include documentation** - Add educational content
4. **Test thoroughly** - Ensure proper encoding/decoding

### Bug Reports

- Use GitHub Issues
- Include steps to reproduce
- Provide system information

### Feature Requests

- Describe the use case
- Explain the expected behavior
- Consider educational value

## 📊 Statistics

```
Total Ciphers: 14
├── Historical: 6
├── Modern: 5
└── Self-Inverse: 2

Lines of Code: ~1,000+
Educational Features: 15+
Audio Features: 5
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **clueNA** - Initial work and development
