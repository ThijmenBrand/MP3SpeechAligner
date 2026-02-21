import os
import subprocess
import argparse
from pathlib import Path
from pydub import AudioSegment
import unicodedata

supported_languages = {
    "german": {
        "dict": "german_mfa",
        "model": "german_mfa"
    },
    "hungarian": {
        "dict": None,
        "model": "hungarian_cv"
    }
}

hungarian_letter_mapping = {
    'á': 'aː', 'é': 'eː', 'í': 'iː', 'ó': 'oː', 'ú': 'uː', 'ő': 'øː', 'ű': 'yː',
    'a': 'ɑ', 'e': 'ɛ', 'ö': 'ø', 'ü': 'y', 'i': 'i', 'o': 'o', 'u': 'u',
    'sz': 's', 'zs': 'ʒ', 'ty': 'c', 'gy': 'ɟ', 'ny': 'ɲ', 'ly': 'j', 's': 'ʃ',
    'cs': 'tʃ', 'dzs': 'dʒ', 'dz': 'ts',
    'z': 'z', 'f': 'f', 'h': 'h', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 
    'n': 'n', 'p': 'p', 'r': 'r', 't': 't', 'v': 'v', 'b': 'b', 'd': 'd', 'g': 'ɡ'
}

def prepare_path(input_dir, output_dir) -> tuple[Path, Path, Path]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    corpus_path = output_path / "corpus"
    result_path = output_path / "aligned_results"

    if not input_path.exists():
        print(f"❌ Input directory does not exist: {input_path}")
        exit(1)

    corpus_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    return input_path, corpus_path, result_path

def prepare_mfa_corpus(input_dir, output_dir):
    # Ensure output directory exists
    input_path = Path(input_dir)

    print(f"--- Step 1: Converting files for MFA ---")
    extensions = ('.mp3', '.wav')

    print(f"🚀 Processing files from: {input_path}")

    for file in input_path.iterdir():
        if file.suffix.lower() in extensions:
            # 1. Determine the 'word' prefix (e.g., 'apple_german' -> 'apple')
            word_prefix = file.stem.split('_')[0]
            language = determine_language_from_filename(file.name)
            target_directory = output_dir / language
            target_directory.mkdir(parents=True, exist_ok=True)

            # Determine output filenames
            wav_filename =  target_directory / f"{file.stem}.wav"
            txt_filename = target_directory / f"{file.stem}.txt"

            print(f"📦 Processing: {file.name} -> {word_prefix}")

            try:
                # 2. Convert to WAV (16kHz, Mono, 16-bit) for MFA optimization
                audio = AudioSegment.from_file(file)
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                audio.export(wav_filename, format="wav")

                # 3. Create the transcription .txt file
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(word_prefix)

                print(f"✅ Successfully processed: {file.name}")
            except Exception as e:
                print(f"❌ Error processing {file.name}: {e}")

    print(f"🎉 All files processed! Output saved to: {output_dir}")

def generate_simple_dict(corpus_dir, dict_path, language="unknown") -> Path:
    """Creates a local dictionary where word = L E T T E R S"""

    dict_file = Path(dict_path / f"{language}_dict.txt")
    words = set()

    for txt_file in corpus_dir.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            word = unicodedata.normalize('NFC', f.read().strip().lower())
            if word:
                words.add(word)

    with open(dict_file, 'w', encoding='utf-8') as f:
        for word in sorted(words):
            if language == "hungarian":
                phones = []
                i = 0
                while i < len(word):
                    # 1. Check for Double Consonants (e.g., 'tt' -> 'tː')
                    char = word[i]
                    if i + 1 < len(word) and char == word[i+1]:
                        if char in hungarian_letter_mapping:
                            base_phone = hungarian_letter_mapping[char]
                            phones.append(f"{base_phone}ː")
                            i += 2
                            continue
                        
                    # 2. Standard Mapping (Trigrams -> Digrams -> Singles)
                    if word[i:i+3] in hungarian_letter_mapping:
                        phones.append(hungarian_letter_mapping[word[i:i+3]])
                        i += 3
                    elif word[i:i+2] in hungarian_letter_mapping:
                        phones.append(hungarian_letter_mapping[word[i:i+2]])
                        i += 2
                    elif word[i] in hungarian_letter_mapping:
                        phones.append(hungarian_letter_mapping[word[i]])
                        i += 1
                    else:
                        phones.append(word[i])
                        i += 1
                f.write(f"{word}\t{' '.join(phones)}\n")
            else:
                # Fallback for German or other languages
                phones = " ".join(list(word))
                f.write(f"{word}\t{phones}\n")
    
    return dict_file

def determine_language_from_filename(filename):
    # Simple heuristic to determine language based on filename
    filename_lower = filename.lower()
    for lang in supported_languages.keys():
        if lang in filename_lower:
            return lang
    
    raise ValueError(f"Unable to determine language from filename: {filename}. Please include 'german' or 'hungarian' in the filename.")

def prepare_MFA_alignment(corpus_dir, results_dir, dict_name="german_mfa", model_name="german_mfa"):
    print(f"--- Step 2: Running MFA Alignment ---")
    print(f"📂 Corpus Directory: {corpus_dir}")
    print(f"📂 Results Directory: {results_dir}")

    # check the languages in the corpus directory
    languages_in_corpus = set()
    for subdir in corpus_dir.iterdir():
        if subdir.is_dir():
            languages_in_corpus.add(subdir.name)

    print(f"🔍 Detected languages in corpus: {', '.join(languages_in_corpus)}")

    # for each language, check if the corresponding dictionary and model are available
    for lang in languages_in_corpus:
        if lang not in supported_languages:
            print(f"❌ No dictionary/model found for language: {lang}. Skipping MFA alignment for this language.")
            continue

        dict_name = supported_languages[lang]["dict"]
        model_name = supported_languages[lang]["model"]

        if (dict_name is None):
            print(f"⚠ No dictionary specified for language: {lang}. Creating a simple dictionary based on the corpus.")
            dict_path = corpus_dir / "dictionaries"
            dict_path.mkdir(parents=True, exist_ok=True)
            
            dict_file = generate_simple_dict(corpus_dir / lang, dict_path, language=lang)
            dict_name = str(dict_file)

        localized_corpus_dir = corpus_dir / lang
        localized_results_dir = results_dir / lang
        localized_results_dir.mkdir(parents=True, exist_ok=True)

        run_mfa_alignment(localized_corpus_dir, localized_results_dir, dict_name, model_name)

def run_mfa_alignment(corpus_dir, results_dir, dict_name, model_name):
    mfa_cmd = [
        "mfa", "align",
        str(corpus_dir),  # Corpus directory
        dict_name,
        model_name,
        str(results_dir),  # Output directory
        "--clean",  # Clean intermediate files
        "--overwrite",  # Verbose output for debugging
        "--beam", "100", # Wider search for difficult non-words
        "--retry_beam", "400" # Final attempt with maximum depth
    ]

    try:
        print(f"Executing: {' '.join(mfa_cmd)}")
        subprocess.run(mfa_cmd, check=True, shell=True if os.name == 'nt' else False)
        print(f"✅ MFA Alignment completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ MFA failed. Ensure you are in the correct environment.")
    except FileNotFoundError:
        print("\n❌ 'mfa' command not found. Is MFA installed and in your PATH?")

def run_pipeline(input_dir, output_dir, dict_name="german_mfa", model_name="german_mfa"):
    input_path, corpus_path, result_path = prepare_path(input_dir, output_dir)
    prepare_mfa_corpus(input_path, corpus_path)
    prepare_MFA_alignment(corpus_path, result_path, dict_name=dict_name, model_name=model_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare audio files for MFA training.")
    parser.add_argument("input_dir", help="Directory containing input audio files (MP3/WAV).")
    parser.add_argument("output_dir", help="Directory to save processed WAV and TXT files.")
    parser.add_argument("--dict_name", default="german_mfa", help="Name of the dictionary to use.")
    parser.add_argument("--model_name", default="german_mfa", help="Name of the model to use.")
    args = parser.parse_args()

    run_pipeline(args.input_dir, args.output_dir, dict_name=args.dict_name, model_name=args.model_name)