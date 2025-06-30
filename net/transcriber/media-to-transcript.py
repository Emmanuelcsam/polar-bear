#!/usr/bin/env python3
"""
Media to Transcript Converter
-----------------------------
Converts video/audio files to text transcripts using various transcription services.
Supports: MP4, MP3, WAV, MOV, AVI, MKV, M4A, FLAC, OGG, WebM, and more.

Features:
- Multiple transcription engines (Whisper local/API, AssemblyAI)
- Video to audio extraction
- Multiple output formats (TXT, JSON, SRT)
- Interactive configuration
- Batch processing
"""

import os
import sys
import json
import subprocess
import tempfile
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import glob

# Try to import optional dependencies
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Note: Whisper not installed. Install with: pip install openai-whisper")

# Constants
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg']
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.aiff', '.aif']
ALL_SUPPORTED_FORMATS = SUPPORTED_VIDEO_FORMATS + SUPPORTED_AUDIO_FORMATS

# API Keys (these should be in environment variables in production)
#ASSEMBLY_API_KEY = os.getenv('ASSEMBLY_API_KEY', '7272147812d6494087b023038649592b')
#HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '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')


ASSEMBLY_API_KEY = '7272147812d6494087b023038649592b'
HUGGINGFACE_API_KEY = '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'


class InteractiveTranscriber:
    def __init__(self):
        self.files_to_process = []
        self.transcription_method = 'whisper'
        self.whisper_model = 'base'
        self.output_format = 'txt'
        self.output_directory = os.getcwd()
        self.language = None
        self.api_key = None
        self.transcriber = None

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Print a nice header"""
        print("=" * 60)
        print("        MEDIA TO TRANSCRIPT CONVERTER")
        print("=" * 60)
        print()

    def ask_yes_no(self, question, default='y'):
        """Ask a yes/no question"""
        while True:
            if default == 'y':
                answer = input(f"{question} [Y/n]: ").strip().lower()
                if answer == '' or answer == 'y' or answer == 'yes':
                    return True
                elif answer == 'n' or answer == 'no':
                    return False
            else:
                answer = input(f"{question} [y/N]: ").strip().lower()
                if answer == 'n' or answer == 'no' or answer == '':
                    return False
                elif answer == 'y' or answer == 'yes':
                    return True
            print("Please answer 'y' or 'n'")

    def ask_choice(self, question, options, default=None):
        """Ask user to choose from a list of options"""
        print(f"\n{question}")
        for i, option in enumerate(options, 1):
            if default and option == default:
                print(f"  {i}. {option} (default)")
            else:
                print(f"  {i}. {option}")

        while True:
            if default:
                choice = input(f"\nEnter your choice (1-{len(options)}) [default: {options.index(default)+1}]: ").strip()
                if choice == '':
                    return default
            else:
                choice = input(f"\nEnter your choice (1-{len(options)}): ").strip()

            try:
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return options[index]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

    def select_files(self):
        """Interactive file selection"""
        self.clear_screen()
        self.print_header()
        print("FILE SELECTION")
        print("-" * 40)

        print("\nTips for entering file paths:")
        print("  • You can paste paths directly (with or without quotes)")
        print("  • Tab completion works in most terminals")
        print("  • For paths with spaces, just paste as-is")
        print("  • Example: /home/user/My Videos/lecture.mp4")

        while True:
            print("\nHow would you like to select files?")
            print("  1. Enter file paths manually")
            print("  2. Select all files in a directory")
            print("  3. Use wildcards (e.g., *.mp4)")
            print("  4. Continue with selected files")

            if self.files_to_process:
                print(f"\nCurrently selected: {len(self.files_to_process)} file(s)")
                if len(self.files_to_process) <= 5:
                    for f in self.files_to_process:
                        print(f"  - {os.path.basename(f)}")
                else:
                    for f in self.files_to_process[:3]:
                        print(f"  - {os.path.basename(f)}")
                    print(f"  ... and {len(self.files_to_process) - 3} more")

            choice = input("\nYour choice (1-4): ")


            if choice == '1':
                # Manual entry
                print("\nEnter file paths (press Enter with empty line to finish):")
                print("Tip: For paths with spaces, just paste the path as-is")
                print("Example: /home/jarvis/Downloads/ETC/Audio/South Africa's Struggle for Human Rights.mp3")
                print()
                while True:
                    file_path = input("File path: ").strip()
                    if not file_path:
                        break

                    # Handle quotes and escaped spaces
                    # Remove surrounding quotes if present
                    if (file_path.startswith('"') and file_path.endswith('"')) or \
                       (file_path.startswith("'") and file_path.endswith("'")):
                        file_path = file_path[1:-1]

                    # Replace escaped spaces with regular spaces
                    file_path = file_path.replace('\\ ', ' ')

                    # Expand user path
                    file_path = os.path.expanduser(file_path)

                    if os.path.exists(file_path):
                        ext = Path(file_path).suffix.lower()
                        if ext in ALL_SUPPORTED_FORMATS:
                            if file_path not in self.files_to_process:
                                self.files_to_process.append(file_path)
                                print(f"  ✓ Added: {os.path.basename(file_path)}")
                            else:
                                print(f"  ! Already added: {os.path.basename(file_path)}")
                        else:
                            print(f"  ✗ Unsupported format: {ext}")
                            print(f"    Supported: {', '.join(ALL_SUPPORTED_FORMATS)}")
                    else:
                        print(f"  ✗ File not found: {file_path}")
                        # Try to help with common issues
                        if '\\' in file_path and not os.path.exists(file_path):
                            # Try without backslashes
                            test_path = file_path.replace('\\', '')
                            if os.path.exists(test_path):
                                print(f"    Tip: Try entering the path without backslashes")
                                print(f"    Working path: {test_path}")
                        elif file_path.startswith('~'):
                            expanded = os.path.expanduser(file_path)
                            if os.path.exists(expanded) and expanded != file_path:
                                print(f"    Expanded path: {expanded}")

                        # Suggest using tab completion
                        if not os.path.exists(os.path.dirname(file_path)) and os.path.dirname(file_path):
                            print(f"    Directory not found: {os.path.dirname(file_path)}")
                        else:
                            print(f"    Tip: Use Tab key for auto-completion in your terminal")

            elif choice == '2':
                # Directory selection
                dir_path = input("\nEnter directory path: ").strip()

                # Handle quotes
                if (dir_path.startswith('"') and dir_path.endswith('"')) or \
                   (dir_path.startswith("'") and dir_path.endswith("'")):
                    dir_path = dir_path[1:-1]

                # Replace escaped spaces
                dir_path = dir_path.replace('\\ ', ' ')
                dir_path = os.path.expanduser(dir_path)

                if os.path.isdir(dir_path):
                    found_files = []
                    for ext in ALL_SUPPORTED_FORMATS:
                        found_files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
                        found_files.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))

                    if found_files:
                        print(f"\nFound {len(found_files)} supported file(s)")
                        if self.ask_yes_no("Add all these files?"):
                            for f in found_files:
                                if f not in self.files_to_process:
                                    self.files_to_process.append(f)
                            print(f"  ✓ Added {len(found_files)} file(s)")
                    else:
                        print("  ✗ No supported media files found in directory")
                else:
                    print(f"  ✗ Directory not found: {dir_path}")

            elif choice == '3':
                # Wildcard selection
                pattern = input("\nEnter file pattern (e.g., ~/Videos/*.mp4): ").strip()

                # Handle quotes
                if (pattern.startswith('"') and pattern.endswith('"')) or \
                   (pattern.startswith("'") and pattern.endswith("'")):
                    pattern = pattern[1:-1]

                # Don't replace escaped spaces in patterns - glob needs them
                pattern = os.path.expanduser(pattern)

                found_files = glob.glob(pattern)
                valid_files = []

                for f in found_files:
                    if os.path.isfile(f):
                        ext = Path(f).suffix.lower()
                        if ext in ALL_SUPPORTED_FORMATS:
                            valid_files.append(f)

                if valid_files:
                    print(f"\nFound {len(valid_files)} matching file(s)")
                    if len(valid_files) <= 10:
                        for f in valid_files:
                            print(f"  - {os.path.basename(f)}")
                    else:
                        for f in valid_files[:5]:
                            print(f"  - {os.path.basename(f)}")
                        print(f"  ... and {len(valid_files) - 5} more")

                    if self.ask_yes_no("Add these files?"):
                        for f in valid_files:
                            if f not in self.files_to_process:
                                self.files_to_process.append(f)
                        print(f"  ✓ Added {len(valid_files)} file(s)")
                else:
                    print("  ✗ No matching files found")

            elif choice == '4':
                if self.files_to_process:
                    return True
                else:
                    print("\n  ✗ No files selected yet!")

            else:
                print("Invalid choice. Please try again.")

    def configure_transcription(self):
        """Configure transcription settings"""
        self.clear_screen()
        self.print_header()
        print("TRANSCRIPTION SETTINGS")
        print("-" * 40)

        # Choose transcription method
        methods = []
        if WHISPER_AVAILABLE:
            methods.append("whisper")
        methods.extend(["whisper-api", "assembly"])

        if len(methods) > 1:
            print("\nAvailable transcription methods:")
            if "whisper" in methods:
                print("  - whisper: Local processing (no internet required)")
            print("  - whisper-api: Cloud-based Whisper (requires internet)")
            print("  - assembly: AssemblyAI service (requires internet)")

            method_display = {
                "whisper": "Whisper (Local)",
                "whisper-api": "Whisper (Cloud API)",
                "assembly": "AssemblyAI"
            }

            display_methods = [method_display[m] for m in methods]
            choice = self.ask_choice("Select transcription method:", display_methods)

            # Map back to method key
            for key, value in method_display.items():
                if value == choice:
                    self.transcription_method = key
                    break
        else:
            self.transcription_method = methods[0]
            print(f"\nUsing {self.transcription_method} for transcription")

        # Configure method-specific settings
        if self.transcription_method == 'whisper':
            models = ['tiny', 'base', 'small', 'medium', 'large']
            print("\nWhisper model sizes:")
            print("  - tiny: Fastest, least accurate (39M)")
            print("  - base: Fast, good accuracy (74M)")
            print("  - small: Balanced (244M)")
            print("  - medium: Slower, better accuracy (769M)")
            print("  - large: Slowest, best accuracy (1550M)")

            self.whisper_model = self.ask_choice("Select Whisper model:", models, default='base')

        elif self.transcription_method == 'assembly':
            print("\nAssemblyAI API Key")

            # Create a temporary transcriber instance for validation
            temp_transcriber = MediaTranscriber()

            if ASSEMBLY_API_KEY and ASSEMBLY_API_KEY != '7272147812d6494087b023038649592b':
                print(f"Found existing API key: {ASSEMBLY_API_KEY[:10]}...")
                print("Validating API key...")

                is_valid, message = temp_transcriber.validate_assembly_api_key(ASSEMBLY_API_KEY)
                if is_valid:
                    print(f"  ✓ {message}")
                    if self.ask_yes_no("Use this validated API key?"):
                        self.api_key = ASSEMBLY_API_KEY
                    else:
                        self.api_key = self._get_and_validate_api_key(temp_transcriber)
                else:
                    print(f"  ✗ {message}")
                    self.api_key = self._get_and_validate_api_key(temp_transcriber)
            else:
                self.api_key = self._get_and_validate_api_key(temp_transcriber)

    def _get_and_validate_api_key(self, transcriber):
        """Helper method to get and validate AssemblyAI API key"""
        while True:
            api_key = input("Enter your AssemblyAI API key: ").strip()

            if not api_key:
                if self.ask_yes_no("Skip API key validation?", default='n'):
                    return ASSEMBLY_API_KEY  # Fallback to default
                continue

            print("Validating API key...")
            is_valid, message = transcriber.validate_assembly_api_key(api_key)

            if is_valid:
                print(f"  ✓ {message}")
                return api_key
            else:
                print(f"  ✗ {message}")
                if not self.ask_yes_no("Try a different API key?"):
                    print("Using provided key anyway (validation may have failed due to network issues)")
                    return api_key

        # Language setting
        if self.ask_yes_no("\nSpecify transcription language?", default='n'):
            self.language = input("Enter language code (e.g., en, es, fr, de, ja): ").strip().lower()

        # Output format
        formats = ['txt', 'json', 'srt']
        print("\nOutput formats:")
        print("  - txt: Plain text transcript")
        print("  - json: Structured data with timestamps")
        print("  - srt: Subtitle file with timings")

        self.output_format = self.ask_choice("Select output format:", formats, default='txt')

        # Output directory
        print(f"\nCurrent output directory: {self.output_directory}")
        if self.ask_yes_no("Change output directory?", default='n'):
            while True:
                new_dir = input("Enter output directory path: ").strip()

                # Handle quotes
                if (new_dir.startswith('"') and new_dir.endswith('"')) or \
                   (new_dir.startswith("'") and new_dir.endswith("'")):
                    new_dir = new_dir[1:-1]

                # Replace escaped spaces
                new_dir = new_dir.replace('\\ ', ' ')
                new_dir = os.path.expanduser(new_dir)

                if os.path.exists(new_dir) and os.path.isdir(new_dir):
                    self.output_directory = new_dir
                    print(f"  ✓ Output directory set to: {self.output_directory}")
                    break
                else:
                    if self.ask_yes_no(f"Directory doesn't exist. Create it?"):
                        try:
                            os.makedirs(new_dir, exist_ok=True)
                            self.output_directory = new_dir
                            print(f"  ✓ Created directory: {self.output_directory}")
                            break
                        except Exception as e:
                            print(f"  ✗ Failed to create directory: {e}")
                    else:
                        if self.ask_yes_no("Keep current directory?"):
                            break

    def show_summary(self):
        """Show configuration summary"""
        self.clear_screen()
        self.print_header()
        print("CONFIGURATION SUMMARY")
        print("-" * 40)

        print(f"\nFiles to process: {len(self.files_to_process)}")
        if len(self.files_to_process) <= 10:
            for f in self.files_to_process:
                print(f"  - {os.path.basename(f)}")
        else:
            for f in self.files_to_process[:5]:
                print(f"  - {os.path.basename(f)}")
            print(f"  ... and {len(self.files_to_process) - 5} more")

        print(f"\nTranscription method: {self.transcription_method}")
        if self.transcription_method == 'whisper':
            print(f"Whisper model: {self.whisper_model}")

        if self.language:
            print(f"Language: {self.language}")

        print(f"Output format: {self.output_format}")
        print(f"Output directory: {self.output_directory}")

        print("\n" + "-" * 40)
        return self.ask_yes_no("\nProceed with transcription?")

    def process_files(self):
        """Process all selected files"""
        # Initialize transcriber (it's defined in this same file)
        self.transcriber = MediaTranscriber(
            transcription_method=self.transcription_method,
            whisper_model=self.whisper_model,
            output_format=self.output_format,
            api_key=self.api_key
        )

        print("\nStarting transcription...")
        print("=" * 60)

        successful = 0
        failed = 0

        for i, file_path in enumerate(self.files_to_process, 1):
            print(f"\n[{i}/{len(self.files_to_process)}] Processing: {os.path.basename(file_path)}")
            print("-" * 40)

            try:
                # Transcribe
                result = self.transcriber.transcribe(file_path, language=self.language)

                # Save transcript
                output_name = f"{Path(file_path).stem}_transcript"
                ext_map = {'txt': '.txt', 'json': '.json', 'srt': '.srt'}
                output_path = os.path.join(
                    self.output_directory,
                    output_name + ext_map[self.output_format]
                )

                saved_path = self.transcriber.save_transcript(result, file_path, output_path)
                print(f"  ✓ Saved: {os.path.basename(saved_path)}")
                successful += 1

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                failed += 1

        print("\n" + "=" * 60)
        print("TRANSCRIPTION COMPLETE")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print("=" * 60)

    def run(self):
        """Run the interactive transcriber"""
        try:
            # Welcome message
            self.clear_screen()
            self.print_header()
            print("Welcome! This tool converts media files to text transcripts.")
            print("\nSupported formats:")
            print(f"  Video: {', '.join(SUPPORTED_VIDEO_FORMATS)}")
            print(f"  Audio: {', '.join(SUPPORTED_AUDIO_FORMATS)}")

            input("\nPress Enter to continue...")

            # Step 1: Select files
            if not self.select_files():
                print("\nNo files selected. Exiting...")
                return

            # Step 2: Configure transcription
            self.configure_transcription()

            # Step 3: Show summary and confirm
            if not self.show_summary():
                print("\nTranscription cancelled.")
                return

            # Step 4: Process files
            self.process_files()

            print("\nDone! Press Enter to exit...")
            input()

        except KeyboardInterrupt:
            print("\n\nTranscription cancelled by user.")
        except Exception as e:
            print(f"\n\nError: {str(e)}")
            input("\nPress Enter to exit...")


# MediaTranscriber class (same as before, but in a separate module-like section)
class MediaTranscriber:
    def __init__(self, transcription_method='whisper', whisper_model='base',
                 output_format='txt', api_key=None):
        self.transcription_method = transcription_method
        self.whisper_model = whisper_model
        self.output_format = output_format
        self.api_key = api_key or ASSEMBLY_API_KEY
        self.whisper_model_instance = None

    def validate_assembly_api_key(self, api_key: str = None) -> tuple[bool, str]:
        """Validate AssemblyAI API key by calling the account endpoint"""
        test_key = api_key or self.api_key

        if not test_key:
            return False, "No API key provided"

        try:
            url = "https://api.assemblyai.com/v2/account"
            headers = {"Authorization": test_key}

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                account_info = response.json()
                return True, f"Valid API key for account: {account_info.get('email', 'Unknown')}"
            elif response.status_code == 401:
                return False, "Invalid API key - Authentication failed"
            elif response.status_code == 403:
                return False, "API key lacks necessary permissions"
            else:
                return False, f"API validation failed with status {response.status_code}"

        except requests.exceptions.Timeout:
            return False, "API validation timed out - check your internet connection"
        except requests.exceptions.RequestException as e:
            return False, f"Network error during API validation: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error during API validation: {str(e)}"

    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is installed"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def extract_audio_from_video(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video file using ffmpeg"""
        if not self.check_ffmpeg():
            raise RuntimeError("ffmpeg not found. Please install ffmpeg to process video files.")

        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                audio_path
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                print(f"FFmpeg error: {process.stderr}")
                return False
            return True
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False

    def convert_audio_format(self, input_path: str, output_path: str) -> bool:
        """Convert audio to WAV format for compatibility"""
        if not self.check_ffmpeg():
            raise RuntimeError("ffmpeg not found. Please install ffmpeg to convert audio files.")

        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                output_path
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)
            return process.returncode == 0
        except Exception as e:
            print(f"Error converting audio: {e}")
            return False

    def transcribe_with_whisper_local(self, audio_path: str, language: str = None) -> dict:
        """Transcribe using local Whisper model"""
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper not installed. Install with: pip install openai-whisper")

        # Load model if not already loaded
        if self.whisper_model_instance is None:
            print(f"  Loading Whisper model '{self.whisper_model}'...")
            self.whisper_model_instance = whisper.load_model(self.whisper_model)

        print("  Transcribing with Whisper...")
        result = self.whisper_model_instance.transcribe(
            audio_path,
            language=language,
            task='transcribe',
            verbose=False
        )

        return {
            'text': result['text'],
            'segments': result.get('segments', []),
            'language': result.get('language', language)
        }

    def transcribe_with_whisper_api(self, audio_path: str) -> dict:
        """Transcribe using Hugging Face Whisper API"""
        print("  Transcribing with Hugging Face Whisper API...")

        api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

        with open(audio_path, 'rb') as f:
            audio_data = f.read()

        response = requests.post(api_url, headers=headers, data=audio_data)

        if response.status_code == 200:
            result = response.json()
            return {
                'text': result.get('text', ''),
                'segments': [],
                'language': None
            }
        else:
            raise RuntimeError(f"Hugging Face API error: {response.status_code} - {response.text}")

    def transcribe_with_assembly(self, audio_path: str) -> dict:
        """Transcribe using AssemblyAI API"""
        # Validate API key first
        is_valid, message = self.validate_assembly_api_key()
        if not is_valid:
            raise RuntimeError(f"AssemblyAI API key validation failed: {message}")

        print("  Uploading audio to AssemblyAI...")

        # Upload file
        headers = {'authorization': self.api_key}
        upload_url = 'https://api.assemblyai.com/v2/upload'

        with open(audio_path, 'rb') as f:
            response = requests.post(upload_url, headers=headers, data=f)

        if response.status_code != 200:
            raise RuntimeError(f"Upload failed: {response.status_code} - {response.text}")

        audio_url = response.json()['upload_url']

        # Request transcription
        print("  Requesting transcription...")
        transcript_url = 'https://api.assemblyai.com/v2/transcript'
        transcript_request = {
            'audio_url': audio_url,
            'auto_timestamps': True,
            'format_text': True
        }

        response = requests.post(transcript_url, json=transcript_request, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"Transcription request failed: {response.status_code}")

        transcript_id = response.json()['id']

        # Poll for completion
        print("  Processing transcription...")
        polling_url = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'

        while True:
            response = requests.get(polling_url, headers=headers)
            result = response.json()

            if result['status'] == 'completed':
                # Convert to our format
                segments = []
                if result.get('words'):
                    for word in result['words']:
                        segments.append({
                            'start': word['start'] / 1000,  # Convert to seconds
                            'end': word['end'] / 1000,
                            'text': word['text']
                        })

                return {
                    'text': result['text'],
                    'segments': segments,
                    'language': result.get('language_code')
                }
            elif result['status'] == 'error':
                raise RuntimeError(f"Transcription failed: {result.get('error')}")

            time.sleep(3)

    def transcribe(self, media_path: str, language: str = None) -> dict:
        """Main transcription method that handles any media file"""
        file_ext = Path(media_path).suffix.lower()

        # Check if file exists
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"File not found: {media_path}")

        # Prepare audio file
        temp_audio = None
        audio_path = media_path

        try:
            # Extract audio from video if needed
            if file_ext in SUPPORTED_VIDEO_FORMATS:
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_audio.close()
                print(f"  Extracting audio from video...")
                if not self.extract_audio_from_video(media_path, temp_audio.name):
                    raise RuntimeError("Failed to extract audio from video")
                audio_path = temp_audio.name

            # Convert audio format if needed
            elif file_ext in SUPPORTED_AUDIO_FORMATS and file_ext != '.wav':
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_audio.close()
                print(f"  Converting audio format...")
                if not self.convert_audio_format(media_path, temp_audio.name):
                    raise RuntimeError("Failed to convert audio format")
                audio_path = temp_audio.name

            # Perform transcription
            if self.transcription_method == 'whisper':
                result = self.transcribe_with_whisper_local(audio_path, language)
            elif self.transcription_method == 'whisper-api':
                result = self.transcribe_with_whisper_api(audio_path)
            elif self.transcription_method == 'assembly':
                result = self.transcribe_with_assembly(audio_path)
            else:
                raise ValueError(f"Unknown transcription method: {self.transcription_method}")

            return result

        finally:
            # Clean up temporary files
            if temp_audio and os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)

    def format_output(self, result: dict, media_path: str) -> str:
        """Format transcription result based on output format"""
        if self.output_format == 'txt':
            return result['text']

        elif self.output_format == 'json':
            output_data = {
                'source_file': os.path.basename(media_path),
                'timestamp': datetime.now().isoformat(),
                'language': result.get('language'),
                'text': result['text'],
                'segments': result.get('segments', [])
            }
            return json.dumps(output_data, indent=2, ensure_ascii=False)

        elif self.output_format == 'srt':
            # Generate SRT format
            srt_content = []
            segments = result.get('segments', [])

            if not segments:
                # If no segments, create one segment for the whole text
                segments = [{
                    'start': 0,
                    'end': 10,
                    'text': result['text']
                }]

            for i, segment in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(segment.get('start', 0))
                end_time = self._seconds_to_srt_time(segment.get('end', 0))
                text = segment.get('text', '').strip()

                if text:
                    srt_content.append(f"{i}")
                    srt_content.append(f"{start_time} --> {end_time}")
                    srt_content.append(text)
                    srt_content.append("")

            return '\n'.join(srt_content)

        else:
            raise ValueError(f"Unknown output format: {self.output_format}")

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = int(td.total_seconds() % 60)
        milliseconds = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def save_transcript(self, result: dict, media_path: str, output_path: str = None) -> str:
        """Save transcript to file"""
        formatted_output = self.format_output(result, media_path)

        if output_path is None:
            # Generate output path
            base_name = Path(media_path).stem
            ext_map = {'txt': '.txt', 'json': '.json', 'srt': '.srt'}
            output_path = f"{base_name}_transcript{ext_map[self.output_format]}"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_output)

        return output_path


def main():
    """Main function"""
    # Check if user accidentally provided command line arguments
    if len(sys.argv) > 1:
        print("Note: This script uses an interactive interface.")
        print("Just run: python media_to_transcript.py")
        print("\nStarting interactive mode anyway...\n")
        time.sleep(2)

    app = InteractiveTranscriber()
    app.run()


if __name__ == "__main__":
    main()
