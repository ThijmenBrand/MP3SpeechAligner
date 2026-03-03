import os
import subprocess
import argparse
from pathlib import Path
from pydub import AudioSegment
import unicodedata

hungarian_letter_mapping = {
    'á': 'aː', 'é': 'eː', 'í': 'iː', 'ó': 'oː', 'ú': 'uː', 'ő': 'øː', 'ű': 'yː',
    'a': 'ɑ', 'e': 'ɛ', 'ö': 'ø', 'ü': 'y', 'i': 'i', 'o': 'o', 'u': 'u',
    'sz': 's', 'zs': 'ʒ', 'ty': 'c', 'gy': 'ɟ', 'ny': 'ɲ', 'ly': 'j', 's': 'ʃ',
    'cs': 'tʃ', 'dzs': 'dʒ', 'dz': 'ts',
    'z': 'z', 'f': 'f', 'h': 'h', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 
    'n': 'n', 'p': 'p', 'r': 'r', 't': 't', 'v': 'v', 'b': 'b', 'd': 'd', 'g': 'ɡ'
}

# Add this to your mappings
german_letter_mapping = {
    'aa': 'aː', 'ee': 'eː', 'ie': 'iː', 'oo': 'oː', 'uu': 'uː',
    'ah': 'aː', 'eh': 'eː', 'ih': 'iː', 'oh': 'oː', 'uh': 'uː',
    'ü': 'yː', 'ö': 'øː', 'ä': 'ɛː',
    'a': 'a', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɔ', 'u': 'ʊ',
    'ch': 'ç', 'sch': 'ʃ', 'sz': 's', 'ß': 's', 'ph': 'f',
    'p': 'pʰ', 't': 'tʰ', 'k': 'kʰ', 'b': 'b', 'd': 'd', 'g': 'ɡ',
    'f': 'f', 'v': 'f', 'w': 'v', 's': 'z', 'z': 'ts', 'j': 'j', 'h': 'h',
    'l': 'l', 'm': 'm', 'n': 'n', 'r': 'ʁ', 'ng': 'ŋ'
}

supported_languages = {
    "german": {
        "dict": None,
        "model": "german_mfa",
        "letter_mapping": german_letter_mapping
    },
    "hungarian": {
        "dict": None,
        "model": "hungarian_cv",
        "letter_mapping": hungarian_letter_mapping
    }
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

    letter_mapping = supported_languages.get(language, {}).get("letter_mapping", {})

    with open(dict_file, 'w', encoding='utf-8') as f:
        for word in sorted(words):
            if language == "hungarian":
                phones = []
                i = 0
                while i < len(word):
                    # if language is hungarian, we need to check for digraphs and trigraphs first
                    # 1. Check for Double Consonants (e.g., 'tt' -> 'tː')
                    if language == "hungarian":
                        char = word[i]
                        if i + 1 < len(word) and char == word[i+1]:
                            if char in letter_mapping:
                                base_phone = letter_mapping[char]
                                phones.append(f"{base_phone}ː")
                                i += 2
                                continue

                    # If language is german 
                    # In German, double consonants make the vowel short, but phone stays single
                    if language == "german":
                        if i + 1 < len(word) and word[i] == word[i+1] and word[i] not in 'aeiouäöü':
                            if word[i] in letter_mapping:
                                phones.append(letter_mapping[word[i]].replace('ʰ', '')) # Unaspirated in middle
                                i += 2
                                continue

                    # 2. Standard Mapping (Trigrams -> Digrams -> Singles)
                    if word[i:i+3] in letter_mapping:
                        phones.append(letter_mapping[word[i:i+3]])
                        i += 3
                    elif word[i:i+2] in letter_mapping:
                        phones.append(letter_mapping[word[i:i+2]])
                        i += 2
                    elif word[i] in letter_mapping:
                        phones.append(letter_mapping[word[i]])
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

    
def convert_recursive_wav_to_mp3(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for file in input_path.rglob("*.wav"):
        rel_path = file.relative_to(input_path)
        target = output_path / rel_path.with_suffix(".mp3")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"🎵 Converting: {rel_path}")
            AudioSegment.from_wav(file).export(target, format="mp3", bitrate="192k")
            print(f"✅ Converted: {rel_path} -> {target.relative_to(output_path)}")
        except Exception as e:
            print(f"❌ Error converting {rel_path}: {e}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Aligner Toolkit")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Align Command
    align_parser = subparsers.add_parser("align", help="Prepare audio and run MFA alignment")
    align_parser.add_argument("input", help="Input directory with .mp3 files")
    align_parser.add_argument("output", help="Output directory for results")
    align_parser.add_argument("--dict_name", default="german_mfa", help="Name of the dictionary to use.")
    align_parser.add_argument("--model_name", default="german_mfa", help="Name of the model to use.")

    # Convert Command
    convert_parser = subparsers.add_parser("convert", help="Recursively convert WAV to MP3")
    convert_parser.add_argument("input", help="Input directory with .wav files")
    convert_parser.add_argument("output", help="Output directory for .mp3 files")
    
    args = parser.parse_args()

    if args.command == "align":   
        input_path, corpus_path, result_path = prepare_path(args.input, args.output)
        prepare_mfa_corpus(input_path, corpus_path)
        prepare_MFA_alignment(corpus_path, result_path, dict_name=args.dict_name, model_name=args.model_name)

    elif args.command == "convert":
        convert_recursive_wav_to_mp3(args.input, args.output)

    else:
        parser.print_help()