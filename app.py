import streamlit as st
import numpy as np
import io
import base64
import time
import urllib.parse
import re
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Morse Code Dictionary
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', ',': '--..--', '.': '.-.-.-', '?': '..--..',
    '/': '-..-.', '-': '-....-', '(': '-.--.', ')': '-.--.-', ' ': '/'
}

# Reverse dictionary for decoding
REVERSE_MORSE_DICT = {value: key for key, value in MORSE_CODE_DICT.items()}

# Polybius Square
POLYBIUS_SQUARE = {
    'A': '11', 'B': '12', 'C': '13', 'D': '14', 'E': '15',
    'F': '21', 'G': '22', 'H': '23', 'I': '24', 'J': '24',  # I and J share 24
    'K': '25', 'L': '31', 'M': '32', 'N': '33', 'O': '34',
    'P': '35', 'Q': '41', 'R': '42', 'S': '43', 'T': '44',
    'U': '45', 'V': '51', 'W': '52', 'X': '53', 'Y': '54', 'Z': '55'
}

REVERSE_POLYBIUS = {v: k for k, v in POLYBIUS_SQUARE.items()}

# Pigpen Cipher mapping (text representation)
PIGPEN_DICT = {
    'A': '⌐', 'B': '⌐•', 'C': '¬', 'D': '├', 'E': '├•', 'F': '┤',
    'G': '└', 'H': '└•', 'I': '┘', 'J': '<', 'K': '<•', 'L': '>',
    'M': 'v', 'N': 'v•', 'O': '^', 'P': '◣', 'Q': '◣•', 'R': '◤',
    'S': '◢', 'T': '◢•', 'U': '◥', 'V': '●', 'W': '◐', 'X': '◑',
    'Y': '◒', 'Z': '◓'
}

REVERSE_PIGPEN = {v: k for k, v in PIGPEN_DICT.items()}

# ===== MORSE CODE FUNCTIONS =====

def text_to_morse(text):
    """Convert text to Morse code"""
    morse_code = ''
    for char in text.upper():
        if char in MORSE_CODE_DICT:
            morse_code += MORSE_CODE_DICT[char] + ' '
        else:
            morse_code += char + ' '
    return morse_code.strip()

def morse_to_text(morse):
    """Convert Morse code to text"""
    morse_chars = morse.split(' ')
    decoded_text = ''
    for morse_char in morse_chars:
        if morse_char in REVERSE_MORSE_DICT:
            decoded_text += REVERSE_MORSE_DICT[morse_char]
        elif morse_char == '':
            continue
        else:
            decoded_text += '?'
    return decoded_text

def generate_tone(frequency, duration, sample_rate=44100, fade_ms=5):
    """Generate a tone with specified frequency and duration"""
    frames = int(duration * sample_rate)
    fade_frames = int(fade_ms * sample_rate / 1000)
    
    t = np.linspace(0, duration, frames, False)
    tone = np.sin(2 * np.pi * frequency * t)
    
    if fade_frames > 0:
        fade_in = np.linspace(0, 1, fade_frames)
        fade_out = np.linspace(1, 0, fade_frames)
        tone[:fade_frames] *= fade_in
        tone[-fade_frames:] *= fade_out
    
    return tone

def morse_to_audio(morse_code, wpm=20, frequency=600, sample_rate=44100):
    """Convert Morse code to audio signal"""
    unit_duration = 1.2 / wpm
    
    dot_duration = unit_duration
    dash_duration = 3 * unit_duration
    symbol_gap = unit_duration
    letter_gap = 3 * unit_duration
    word_gap = 7 * unit_duration
    
    audio_segments = []
    
    i = 0
    while i < len(morse_code):
        char = morse_code[i]
        
        if char == '.':
            tone = generate_tone(frequency, dot_duration, sample_rate)
            audio_segments.append(tone)
        elif char == '-':
            tone = generate_tone(frequency, dash_duration, sample_rate)
            audio_segments.append(tone)
        elif char == ' ':
            if i + 1 < len(morse_code) and morse_code[i + 1] == ' ':
                silence = np.zeros(int(word_gap * sample_rate))
                audio_segments.append(silence)
                while i + 1 < len(morse_code) and morse_code[i + 1] == ' ':
                    i += 1
            else:
                silence = np.zeros(int(letter_gap * sample_rate))
                audio_segments.append(silence)
        elif char == '/':
            silence = np.zeros(int(word_gap * sample_rate))
            audio_segments.append(silence)
        
        if char in '.-' and i + 1 < len(morse_code) and morse_code[i + 1] in '.-':
            silence = np.zeros(int(symbol_gap * sample_rate))
            audio_segments.append(silence)
        
        i += 1
    
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val * 0.8
        return full_audio
    else:
        return np.array([])

def create_audio_download(audio_data, sample_rate=44100, filename="morse_code.wav"):
    """Create a downloadable audio file"""
    audio_int = (audio_data * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_int)
    buffer.seek(0)
    return buffer.getvalue()

def create_audio_player(audio_data, sample_rate=44100):
    """Create an HTML audio player"""
    wav_data = create_audio_download(audio_data, sample_rate)
    b64_audio = base64.b64encode(wav_data).decode()
    
    audio_html = f"""
    <audio controls autoplay style="width: 100%;">
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    
    return audio_html, wav_data

def visualize_morse_timing(morse_code, wpm=20):
    """Create a visual representation of Morse code timing"""
    unit_duration = 1.2 / wpm
    
    dot_duration = unit_duration
    dash_duration = 3 * unit_duration
    symbol_gap = unit_duration
    letter_gap = 3 * unit_duration
    word_gap = 7 * unit_duration
    
    timeline = []
    current_time = 0
    
    i = 0
    while i < len(morse_code):
        char = morse_code[i]
        
        if char == '.':
            timeline.append((current_time, current_time + dot_duration, 'dot'))
            current_time += dot_duration
            
        elif char == '-':
            timeline.append((current_time, current_time + dash_duration, 'dash'))
            current_time += dash_duration
            
        elif char == ' ':
            if i + 1 < len(morse_code) and morse_code[i + 1] == ' ':
                timeline.append((current_time, current_time + word_gap, 'word_gap'))
                current_time += word_gap
                while i + 1 < len(morse_code) and morse_code[i + 1] == ' ':
                    i += 1
            else:
                timeline.append((current_time, current_time + letter_gap, 'letter_gap'))
                current_time += letter_gap
                
        elif char == '/':
            timeline.append((current_time, current_time + word_gap, 'word_gap'))
            current_time += word_gap
        
        if char in '.-' and i + 1 < len(morse_code) and morse_code[i + 1] in '.-':
            current_time += symbol_gap
        
        i += 1
    
    return timeline, current_time

# ===== NEW CIPHER IMPLEMENTATIONS =====

def vigenere_encrypt(text, key):
    """Vigenère cipher encryption"""
    if not key:
        return "Key cannot be empty"
    
    result = ""
    key = key.upper()
    key_index = 0
    
    for char in text:
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('A')
            
            if char.isupper():
                result += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            else:
                result += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            
            key_index += 1
        else:
            result += char
    
    return result

def vigenere_decrypt(text, key):
    """Vigenère cipher decryption"""
    if not key:
        return "Key cannot be empty"
    
    result = ""
    key = key.upper()
    key_index = 0
    
    for char in text:
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('A')
            
            if char.isupper():
                result += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            else:
                result += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            
            key_index += 1
        else:
            result += char
    
    return result

def rail_fence_encrypt(text, rails):
    """Rail Fence cipher encryption"""
    if rails <= 1:
        return text
    
    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1
    
    for char in text:
        fence[rail].append(char)
        rail += direction
        
        if rail == rails - 1 or rail == 0:
            direction = -direction
    
    result = ''
    for rail_chars in fence:
        result += ''.join(rail_chars)
    
    return result

def rail_fence_decrypt(cipher, rails):
    """Rail Fence cipher decryption"""
    if rails <= 1:
        return cipher
    
    rail_lengths = [0] * rails
    rail = 0
    direction = 1
    
    for _ in range(len(cipher)):
        rail_lengths[rail] += 1
        rail += direction
        
        if rail == rails - 1 or rail == 0:
            direction = -direction
    
    rails_text = []
    start = 0
    for length in rail_lengths:
        rails_text.append(list(cipher[start:start + length]))
        start += length
    
    result = []
    rail = 0
    direction = 1
    rail_indices = [0] * rails
    
    for _ in range(len(cipher)):
        result.append(rails_text[rail][rail_indices[rail]])
        rail_indices[rail] += 1
        rail += direction
        
        if rail == rails - 1 or rail == 0:
            direction = -direction
    
    return ''.join(result)

def a1z26_encode(text):
    """A1Z26 cipher - A=1, B=2, etc."""
    result = []
    for char in text.upper():
        if char.isalpha():
            result.append(str(ord(char) - ord('A') + 1))
        elif char == ' ':
            result.append('/')
        else:
            result.append(char)
    return '-'.join(result)

def a1z26_decode(text):
    """A1Z26 cipher decoding"""
    try:
        parts = text.split('-')
        result = ''
        for part in parts:
            if part == '/':
                result += ' '
            elif part.isdigit():
                num = int(part)
                if 1 <= num <= 26:
                    result += chr(num - 1 + ord('A'))
                else:
                    result += '?'
            else:
                result += part
        return result
    except:
        return "Invalid A1Z26 format"

def playfair_prepare_key(key):
    """Prepare Playfair key matrix"""
    key = key.upper().replace('J', 'I')
    seen = set()
    key_chars = []
    
    for char in key:
        if char.isalpha() and char not in seen:
            key_chars.append(char)
            seen.add(char)
    
    for char in 'ABCDEFGHIKLMNOPQRSTUVWXYZ':
        if char not in seen:
            key_chars.append(char)
            seen.add(char)
    
    matrix = []
    for i in range(5):
        matrix.append(key_chars[i*5:(i+1)*5])
    
    return matrix

def playfair_find_position(matrix, char):
    """Find position of character in Playfair matrix"""
    for i in range(5):
        for j in range(5):
            if matrix[i][j] == char:
                return i, j
    return None, None

def playfair_encrypt(text, key):
    """Playfair cipher encryption"""
    if not key:
        return "Key cannot be empty"
    
    matrix = playfair_prepare_key(key)
    text = text.upper().replace('J', 'I').replace(' ', '')
    
    # Prepare text pairs
    pairs = []
    i = 0
    while i < len(text):
        if i == len(text) - 1:
            pairs.append(text[i] + 'X')
            i += 1
        elif text[i] == text[i + 1]:
            pairs.append(text[i] + 'X')
            i += 1
        else:
            pairs.append(text[i:i+2])
            i += 2
    
    # Encrypt pairs
    result = ''
    for pair in pairs:
        if len(pair) == 2 and pair[0].isalpha() and pair[1].isalpha():
            row1, col1 = playfair_find_position(matrix, pair[0])
            row2, col2 = playfair_find_position(matrix, pair[1])
            
            if row1 is not None and row2 is not None:
                if row1 == row2:  # Same row
                    result += matrix[row1][(col1 + 1) % 5]
                    result += matrix[row2][(col2 + 1) % 5]
                elif col1 == col2:  # Same column
                    result += matrix[(row1 + 1) % 5][col1]
                    result += matrix[(row2 + 1) % 5][col2]
                else:  # Rectangle
                    result += matrix[row1][col2]
                    result += matrix[row2][col1]
            else:
                result += pair
        else:
            result += pair
    
    return result

def playfair_decrypt(text, key):
    """Playfair cipher decryption"""
    if not key:
        return "Key cannot be empty"
    
    matrix = playfair_prepare_key(key)
    
    pairs = []
    for i in range(0, len(text), 2):
        if i + 1 < len(text):
            pairs.append(text[i:i+2])
        else:
            pairs.append(text[i] + 'X')
    
    result = ''
    for pair in pairs:
        if len(pair) == 2 and pair[0].isalpha() and pair[1].isalpha():
            row1, col1 = playfair_find_position(matrix, pair[0])
            row2, col2 = playfair_find_position(matrix, pair[1])
            
            if row1 is not None and row2 is not None:
                if row1 == row2:  # Same row
                    result += matrix[row1][(col1 - 1) % 5]
                    result += matrix[row2][(col2 - 1) % 5]
                elif col1 == col2:  # Same column
                    result += matrix[(row1 - 1) % 5][col1]
                    result += matrix[(row2 - 1) % 5][col2]
                else:  # Rectangle
                    result += matrix[row1][col2]
                    result += matrix[row2][col1]
            else:
                result += pair
        else:
            result += pair
    
    return result

def hex_encode(text):
    """Hexadecimal encoding"""
    return text.encode('utf-8').hex().upper()

def hex_decode(hex_text):
    """Hexadecimal decoding"""
    try:
        hex_text = hex_text.replace(' ', '').lower()
        return bytes.fromhex(hex_text).decode('utf-8')
    except:
        return "Invalid hexadecimal input"

def url_encode(text):
    """URL encoding"""
    return urllib.parse.quote(text, safe='')

def url_decode(text):
    """URL decoding"""
    try:
        return urllib.parse.unquote(text)
    except:
        return "Invalid URL encoding"

def polybius_encode(text):
    """Polybius Square encoding"""
    result = []
    for char in text.upper():
        if char in POLYBIUS_SQUARE:
            result.append(POLYBIUS_SQUARE[char])
        elif char == ' ':
            result.append('/')
        else:
            result.append(char)
    return ' '.join(result)

def polybius_decode(text):
    """Polybius Square decoding"""
    try:
        parts = text.split(' ')
        result = ''
        for part in parts:
            if part == '/':
                result += ' '
            elif part in REVERSE_POLYBIUS:
                result += REVERSE_POLYBIUS[part]
            else:
                result += part
        return result
    except:
        return "Invalid Polybius input"

def pigpen_encode(text):
    """Pigpen cipher encoding"""
    result = ''
    for char in text.upper():
        if char in PIGPEN_DICT:
            result += PIGPEN_DICT[char] + ' '
        elif char == ' ':
            result += '/ '
        else:
            result += char + ' '
    return result.strip()

def pigpen_decode(text):
    """Pigpen cipher decoding"""
    parts = text.split(' ')
    result = ''
    for part in parts:
        if part == '/':
            result += ' '
        elif part in REVERSE_PIGPEN:
            result += REVERSE_PIGPEN[part]
        else:
            result += part
    return result

# ===== EXISTING CIPHER FUNCTIONS =====

def caesar_cipher_encrypt(text, shift):
    """Caesar cipher encryption"""
    result = ""
    for char in text:
        if char.isalpha():
            ascii_offset = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
        else:
            result += char
    return result

def caesar_cipher_decrypt(text, shift):
    """Caesar cipher decryption"""
    return caesar_cipher_encrypt(text, -shift)

def atbash_cipher(text):
    """Atbash cipher - A=Z, B=Y, etc. (Self-inverse)"""
    result = ""
    for char in text:
        if char.isalpha():
            if char.isupper():
                result += chr(ord('Z') - (ord(char) - ord('A')))
            else:
                result += chr(ord('z') - (ord(char) - ord('a')))
        else:
            result += char
    return result

def rot13_cipher(text):
    """ROT13 cipher (Self-inverse)"""
    return caesar_cipher_encrypt(text, 13)

def base64_encode(text):
    """Base64 encoding"""
    return base64.b64encode(text.encode()).decode()

def base64_decode(text):
    """Base64 decoding"""
    try:
        return base64.b64decode(text.encode()).decode()
    except:
        return "Invalid Base64 input"

def binary_encode(text):
    """Convert text to binary"""
    return ' '.join(format(ord(char), '08b') for char in text)

def binary_decode(binary_text):
    """Convert binary to text"""
    try:
        binary_chars = binary_text.split(' ')
        return ''.join(chr(int(binary_char, 2)) for binary_char in binary_chars)
    except:
        return "Invalid binary input"

# ===== MAIN APPLICATION =====

def main():
    st.set_page_config(
        page_title="Cipher Master",
        page_icon="🔐",
        layout="wide"
    )
    
    st.title("🔐 Cipher Master")
    st.markdown("Convert text using various encryption and encoding methods")
    
    # Sidebar for cipher selection
    st.sidebar.title("Select Cipher Type")
    cipher_type = st.sidebar.selectbox(
        "Choose a cipher:",
        [
            "Morse Code", 
            "Caesar Cipher", 
            "Vigenère Cipher", 
            "Rail Fence Cipher",
            "A1Z26 Cipher",
            "Playfair Cipher",
            "Polybius Square",
            "Pigpen Cipher",
            "Atbash Cipher", 
            "ROT13", 
            "Hexadecimal",
            "URL Encoding",
            "Base64", 
            "Binary"
        ]
    )
    
    # Main content area
    col1, col2 = st.columns(2)
    
    if cipher_type == "Morse Code":
        st.header("📡 Morse Code Converter with Audio")
        
        # Audio settings
        st.sidebar.header("🔊 Audio Settings")
        wpm = st.sidebar.slider("Words Per Minute (WPM):", min_value=5, max_value=40, value=20)
        frequency = st.sidebar.slider("Tone Frequency (Hz):", min_value=300, max_value=1000, value=600)
        
        with col1:
            st.subheader("Text to Morse Code")
            text_input = st.text_area("Enter text to convert:", height=100)
            
            if st.button("Convert to Morse", key="to_morse"):
                if text_input:
                    morse_result = text_to_morse(text_input)
                    st.code(morse_result, language=None)
                    
                    st.subheader("🔊 Audio Playback")
                    with st.spinner("Generating audio..."):
                        audio_data = morse_to_audio(morse_result, wpm=wpm, frequency=frequency)
                        
                    if len(audio_data) > 0:
                        audio_html, wav_data = create_audio_player(audio_data)
                        st.components.v1.html(audio_html, height=60)
                        
                        st.download_button(
                            label="📥 Download Audio (WAV)",
                            data=wav_data,
                            file_name=f"morse_code_{int(time.time())}.wav",
                            mime="audio/wav"
                        )
                        
                        # Timing visualization (RESTORED)
                        with st.expander("📊 View Timing Diagram"):
                            timeline, total_duration = visualize_morse_timing(morse_result, wpm)
                            
                            fig, ax = plt.subplots(figsize=(12, 4))
                            
                            y_pos = 0
                            colors = {'dot': 'blue', 'dash': 'red', 'letter_gap': 'lightgray', 'word_gap': 'gray'}
                            labels_used = set()
                            
                            for start, end, element_type in timeline:
                                color = colors.get(element_type, 'black')
                                height = 0.8 if element_type in ['dot', 'dash'] else 0.3
                                label = element_type.replace('_', ' ').title() if element_type not in labels_used else ""
                                if label:
                                    labels_used.add(element_type)
                                
                                ax.barh(y_pos, end - start, left=start, height=height, 
                                       color=color, alpha=0.7, label=label)
                            
                            ax.set_xlabel('Time (seconds)')
                            ax.set_ylabel('Signal')
                            ax.set_title(f'Morse Code Timing Diagram - {wpm} WPM')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.error("Could not generate audio for the given input.")
        
        with col2:
            st.subheader("Morse Code to Text")
            morse_input = st.text_area("Enter Morse code:", height=100)
            
            if st.button("Convert to Text", key="to_text"):
                if morse_input:
                    text_result = morse_to_text(morse_input)
                    st.success(f"Decoded text: **{text_result}**")
                    
                    st.subheader("🔊 Play Morse Code")
                    if st.button("🎵 Play Audio", key="play_decoded"):
                        with st.spinner("Generating audio..."):
                            audio_data = morse_to_audio(morse_input, wpm=wpm, frequency=frequency)
                            
                        if len(audio_data) > 0:
                            audio_html, wav_data = create_audio_player(audio_data)
                            st.components.v1.html(audio_html, height=60)
        
        # Morse code reference
        with st.expander("📋 Morse Code Reference"):
            st.write("**Letters:**")
            letters = {k: v for k, v in MORSE_CODE_DICT.items() if k.isalpha()}
            st.write(letters)
            st.write("**Numbers:**")
            numbers = {k: v for k, v in MORSE_CODE_DICT.items() if k.isdigit()}
            st.write(numbers)
        
        # Audio information
        st.info("""
        🎵 **Audio Features:**
        - Adjustable speed (WPM) and frequency
        - Standard Morse timing (dot = 1 unit, dash = 3 units)
        - Proper spacing between letters and words
        - Download audio as WAV file
        - Visual timing diagram
        """)
    
    elif cipher_type == "Vigenère Cipher":
        st.header("🔑 Vigenère Cipher")
        st.info("Uses a keyword to encrypt/decrypt. Each letter of the keyword determines the shift for each letter of the message.")
        
        keyword = st.text_input("Enter keyword:", value="CIPHER")
        
        with col1:
            st.subheader("Encrypt")
            plain_text = st.text_area("Enter text to encrypt:", height=100)
            if st.button("Encrypt with Vigenère"):
                if plain_text and keyword:
                    encrypted = vigenere_encrypt(plain_text, keyword)
                    st.code(encrypted, language=None)
                    st.info(f"Using keyword: **{keyword.upper()}**")
        
        with col2:
            st.subheader("Decrypt")
            cipher_text = st.text_area("Enter text to decrypt:", height=100)
            if st.button("Decrypt with Vigenère"):
                if cipher_text and keyword:
                    decrypted = vigenere_decrypt(cipher_text, keyword)
                    st.success(f"Decrypted text: **{decrypted}**")
        
        with st.expander("📚 How Vigenère Works"):
            st.write(f"""
            **Example with keyword '{keyword.upper()}':**
            - H + C = J (H=7, C=2, 7+2=9=J)
            - E + I = M (E=4, I=8, 4+8=12=M)
            - L + P = A (L=11, P=15, 11+15=26=0=A)
            - The keyword repeats: {keyword.upper()}{keyword.upper()}{keyword.upper()}...
            """)
    
    elif cipher_type == "Rail Fence Cipher":
        st.header("🚂 Rail Fence Cipher")
        st.info("Writes the message in a zigzag pattern across multiple 'rails', then reads off each rail.")
        
        rails = st.slider("Number of rails:", min_value=2, max_value=8, value=3)
        
        with col1:
            st.subheader("Encrypt")
            plain_text = st.text_area("Enter text to encrypt:", height=100)
            if st.button("Encrypt with Rail Fence"):
                if plain_text:
                    encrypted = rail_fence_encrypt(plain_text.replace(' ', ''), rails)
                    st.code(encrypted, language=None)
                    st.info(f"Using {rails} rails")
        
        with col2:
            st.subheader("Decrypt")
            cipher_text = st.text_area("Enter text to decrypt:", height=100)
            if st.button("Decrypt with Rail Fence"):
                if cipher_text:
                    decrypted = rail_fence_decrypt(cipher_text, rails)
                    st.success(f"Decrypted text: **{decrypted}**")
        
        with st.expander("📊 Visualize Rail Fence Pattern"):
            if rails == 3:
                st.code("""
Example with 3 rails and "HELLO WORLD":
H   L   O   R   D
 E L   W O L
  L     L

Reading off: HLORD + ELWOL + LL = HLORDEL WOLL
                """)
    
    elif cipher_type == "A1Z26 Cipher":
        st.header("🔢 A1Z26 Cipher")
        st.info("Simple substitution: A=1, B=2, C=3, ..., Z=26")
        
        with col1:
            st.subheader("Encode")
            plain_text = st.text_area("Enter text to encode:", height=100)
            if st.button("Encode to A1Z26"):
                if plain_text:
                    encoded = a1z26_encode(plain_text)
                    st.code(encoded, language=None)
        
        with col2:
            st.subheader("Decode")
            cipher_text = st.text_area("Enter A1Z26 to decode (format: 8-5-12-12-15):", height=100)
            if st.button("Decode from A1Z26"):
                if cipher_text:
                    decoded = a1z26_decode(cipher_text)
                    st.success(f"Decoded text: **{decoded}**")
        
        with st.expander("📋 A1Z26 Reference"):
            st.write("A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8, I=9, J=10, K=11, L=12, M=13")
            st.write("N=14, O=15, P=16, Q=17, R=18, S=19, T=20, U=21, V=22, W=23, X=24, Y=25, Z=26")
    
    elif cipher_type == "Playfair Cipher":
        st.header("⬜ Playfair Cipher")
        st.info("Uses a 5×5 grid based on a keyword. Encrypts pairs of letters using grid rules.")
        
        playfair_key = st.text_input("Enter keyword:", value="MONARCHY")
        
        with col1:
            st.subheader("Encrypt")
            plain_text = st.text_area("Enter text to encrypt:", height=100)
            if st.button("Encrypt with Playfair"):
                if plain_text and playfair_key:
                    encrypted = playfair_encrypt(plain_text, playfair_key)
                    st.code(encrypted, language=None)
                    
                    # Show preprocessing explanation for HELLO
                    if plain_text.upper().replace(' ', '') == "HELLO":
                        st.info("**Why HELLO becomes HELXLOX:** Playfair inserts 'X' between repeated letters (LL) and adds 'X' for odd length.")
        
        with col2:
            st.subheader("Decrypt")
            cipher_text = st.text_area("Enter text to decrypt:", height=100)
            if st.button("Decrypt with Playfair"):
                if cipher_text and playfair_key:
                    decrypted = playfair_decrypt(cipher_text, playfair_key)
                    st.success(f"Decrypted text: **{decrypted}**")
        
        if playfair_key:
            with st.expander("🗂️ View Playfair Grid"):
                matrix = playfair_prepare_key(playfair_key)
                for row in matrix:
                    st.write(" ".join(row))
    
    elif cipher_type == "Polybius Square":
        st.header("📐 Polybius Square")
        st.info("Ancient Greek cipher using a 5×5 grid. Each letter becomes two numbers (row, column).")
        
        with col1:
            st.subheader("Encode")
            plain_text = st.text_area("Enter text to encode:", height=100)
            if st.button("Encode with Polybius"):
                if plain_text:
                    encoded = polybius_encode(plain_text)
                    st.code(encoded, language=None)
        
        with col2:
            st.subheader("Decode")
            cipher_text = st.text_area("Enter Polybius to decode:", height=100)
            if st.button("Decode from Polybius"):
                if cipher_text:
                    decoded = polybius_decode(cipher_text)
                    st.success(f"Decoded text: **{decoded}**")
        
        with st.expander("📋 Polybius Square Grid"):
            st.code("""
  1  2  3  4  5
1 A  B  C  D  E
2 F  G  H  I/J K
3 L  M  N  O  P
4 Q  R  S  T  U
5 V  W  X  Y  Z
            """)
    
    elif cipher_type == "Pigpen Cipher":
        st.header("🐷 Pigpen Cipher")
        st.info("Masonic cipher using special symbols. Each letter gets a unique symbol based on its grid position.")
        
        with col1:
            st.subheader("Encode")
            plain_text = st.text_area("Enter text to encode:", height=100)
            if st.button("Encode with Pigpen"):
                if plain_text:
                    encoded = pigpen_encode(plain_text)
                    st.code(encoded, language=None)
                    st.caption("Pigpen symbols (simplified text representation)")
        
        with col2:
            st.subheader("Decode")
            cipher_text = st.text_area("Enter Pigpen to decode:", height=100)
            if st.button("Decode from Pigpen"):
                if cipher_text:
                    decoded = pigpen_decode(cipher_text)
                    st.success(f"Decoded text: **{decoded}**")
        
        with st.expander("🔤 Pigpen Symbol Reference"):
            st.write("**Sample mappings (simplified symbols):**")
            st.write("A=⌐, B=⌐•, C=¬, D=├, E=├•, F=┤, G=└, H=└•, I=┘")
            st.write("J=<, K=<•, L=>, M=v, N=v•, O=^")
            st.write("**Note:** Real Pigpen uses grid-based symbols, these are text approximations.")
    
    elif cipher_type == "Hexadecimal":
        st.header("🔣 Hexadecimal Encoding")
        st.info("Converts text to base-16 representation using 0-9 and A-F.")
        
        with col1:
            st.subheader("Encode")
            plain_text = st.text_area("Enter text to encode:", height=100)
            if st.button("Encode to Hex"):
                if plain_text:
                    encoded = hex_encode(plain_text)
                    st.code(encoded, language=None)
        
        with col2:
            st.subheader("Decode")
            hex_text = st.text_area("Enter hex to decode:", height=100)
            if st.button("Decode from Hex"):
                if hex_text:
                    decoded = hex_decode(hex_text)
                    st.success(f"Decoded text: **{decoded}**")
    
    elif cipher_type == "URL Encoding":
        st.header("🌐 URL Encoding")
        st.info("Percent-encoding for safe transmission in URLs. Special characters become %XX.")
        
        with col1:
            st.subheader("Encode")
            plain_text = st.text_area("Enter text to encode:", height=100)
            if st.button("URL Encode"):
                if plain_text:
                    encoded = url_encode(plain_text)
                    st.code(encoded, language=None)
        
        with col2:
            st.subheader("Decode")
            url_text = st.text_area("Enter URL encoded text:", height=100)
            if st.button("URL Decode"):
                if url_text:
                    decoded = url_decode(url_text)
                    st.success(f"Decoded text: **{decoded}**")
    
    elif cipher_type == "Caesar Cipher":
        st.header("🏛️ Caesar Cipher")
        shift = st.slider("Select shift value:", min_value=1, max_value=25, value=3)
        
        with col1:
            st.subheader("Encrypt")
            plain_text = st.text_area("Enter text to encrypt:", height=100)
            if st.button("Encrypt", key="caesar_encrypt"):
                if plain_text:
                    encrypted = caesar_cipher_encrypt(plain_text, shift)
                    st.code(encrypted, language=None)
        
        with col2:
            st.subheader("Decrypt")
            cipher_text = st.text_area("Enter text to decrypt:", height=100)
            if st.button("Decrypt", key="caesar_decrypt"):
                if cipher_text:
                    decrypted = caesar_cipher_decrypt(cipher_text, shift)
                    st.success(f"Decrypted text: **{decrypted}**")
    
    elif cipher_type == "Atbash Cipher":
        st.header("🔄 Atbash Cipher")
        st.info("ℹ️ Atbash is **self-inverse**: A↔Z, B↔Y, C↔X, etc.")
        
        with st.expander("📋 See Atbash Mapping"):
            st.write("**Uppercase:** A↔Z, B↔Y, C↔X, D↔W, E↔V, F↔U, G↔T, H↔S, I↔R, J↔Q, K↔P, L↔O, M↔N")
            st.write("**Lowercase:** a↔z, b↔y, c↔x, d↔w, e↔v, f↔u, g↔t, h↔s, i↔r, j↔q, k↔p, l↔o, m↔n")
        
        with col1:
            st.subheader("📝 Original Text → Atbash")
            text_input = st.text_area("Enter original text:", height=100, key="atbash_original")
            if st.button("Apply Atbash", key="atbash_encode"):
                if text_input:
                    result = atbash_cipher(text_input)
                    st.code(result, language=None)
                    st.success("✅ Text converted using Atbash cipher!")
        
        with col2:
            st.subheader("🔓 Atbash Text → Original")
            atbash_input = st.text_area("Enter Atbash-encoded text:", height=100, key="atbash_encoded")
            if st.button("Decode Atbash", key="atbash_decode"):
                if atbash_input:
                    result = atbash_cipher(atbash_input)
                    st.success(f"Decoded text: **{result}**")
    
    elif cipher_type == "ROT13":
        st.header("🔄 ROT13 Cipher")
        st.info("ℹ️ ROT13 is **self-inverse**: rotates letters 13 positions.")
        
        with st.expander("📋 See ROT13 Mapping"):
            st.write("**First half → Second half:** A→N, B→O, C→P, D→Q, E→R, F→S, G→T, H→U, I→V, J→W, K→X, L→Y, M→Z")
            st.write("**Second half → First half:** N→A, O→B, P→C, Q→D, R→E, S→F, T→G, U→H, V→I, W→J, X→K, Y→L, Z→M")
        
        with col1:
            st.subheader("📝 Original Text → ROT13")
            text_input = st.text_area("Enter original text:", height=100, key="rot13_original")
            if st.button("Apply ROT13", key="rot13_encode"):
                if text_input:
                    result = rot13_cipher(text_input)
                    st.code(result, language=None)
                    st.success("✅ Text converted using ROT13!")
        
        with col2:
            st.subheader("🔓 ROT13 Text → Original")
            rot13_input = st.text_area("Enter ROT13-encoded text:", height=100, key="rot13_encoded")
            if st.button("Decode ROT13", key="rot13_decode"):
                if rot13_input:
                    result = rot13_cipher(rot13_input)
                    st.success(f"Decoded text: **{result}**")
    
    elif cipher_type == "Base64":
        st.header("📊 Base64 Encoding")
        
        with col1:
            st.subheader("Encode")
            plain_text = st.text_area("Enter text to encode:", height=100)
            if st.button("Encode to Base64"):
                if plain_text:
                    encoded = base64_encode(plain_text)
                    st.code(encoded, language=None)
        
        with col2:
            st.subheader("Decode")
            encoded_text = st.text_area("Enter Base64 to decode:", height=100)
            if st.button("Decode from Base64"):
                if encoded_text:
                    decoded = base64_decode(encoded_text)
                    st.success(f"Decoded text: **{decoded}**")
    
    elif cipher_type == "Binary":
        st.header("💻 Binary Encoding")
        
        with col1:
            st.subheader("Text to Binary")
            text_input = st.text_area("Enter text to convert to binary:", height=100)
            if st.button("Convert to Binary"):
                if text_input:
                    binary_result = binary_encode(text_input)
                    st.code(binary_result, language=None)
        
        with col2:
            st.subheader("Binary to Text")
            binary_input = st.text_area("Enter binary (space-separated 8-bit groups):", height=100)
            if st.button("Convert to Text"):
                if binary_input:
                    text_result = binary_decode(binary_input)
                    st.success(f"Decoded text: **{text_result}**")
    
    # Footer with additional information
    st.markdown("---")
    st.subheader("🧠 Understanding Self-Inverse Ciphers")
    
    st.markdown("""
    **Self-inverse ciphers** are special because they are their own opposite operation:
    
    - **ROT13**: Shifting 13 positions twice = shifting 26 positions = back to start
    - **Atbash**: Reversing the alphabet twice = back to original position
    - **XOR with same key**: XORing with the same value twice cancels out
    
    This makes them particularly interesting for quick encoding/decoding!
    """)
    
    # Cipher statistics
    st.subheader("📊 Cipher Collection Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Ciphers", "14")
    with col2:
        st.metric("Historical", "6")
    with col3:
        st.metric("Modern", "5")
    with col4:
        st.metric("Self-Inverse", "2")

if __name__ == "__main__":
    main()