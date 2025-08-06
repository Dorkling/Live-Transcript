import sys
import subprocess
import os
# --- MODIFICATION START: Redirect stdout/stderr when frozen ---
import io # Needed for redirection target in some cases

# Redirect stdout and stderr to prevent crashes in windowed mode when libraries print
if getattr(sys, 'frozen', False): # Checks if running as a PyInstaller bundle
    print("Running frozen - Redirecting stdout/stderr...") # Log redirection attempt (won't be visible in windowed mode)
    try:
        # Option 1: Redirect to os.devnull (discard output)
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # Option 2: Redirect to a log file (for debugging) - Uncomment if needed
        # log_file_path = os.path.join(os.path.dirname(sys.executable), 'app_output.log')
        # print(f"Redirecting output to: {log_file_path}") # This print will go to the original stdout before redirection
        # sys.stdout = open(log_file_path, 'a')
        # sys.stderr = sys.stdout # Redirect stderr to the same file
        # print("--- stdout/stderr redirected ---") # This print will go to the file

    except Exception as e:
        # If redirection fails, we can't do much, but try to avoid crashing here
        print(f"Failed to redirect stdout/stderr: {e}") # This might fail too if print is broken
        pass # Continue execution

# --- MODIFICATION END ---

import importlib.util
import queue # For inter-thread data buffering (FIFO) - Used for GUI updates and Vosk audio
import json # Data structure for Vosk results
import threading # For concurrent execution tasks
import time # For timekeeping and delays
from datetime import datetime # For timestamping events
import tkinter as tk # UI toolkit base
from tkinter import ttk, scrolledtext, messagebox, font, filedialog # UI widgets and utilities, added filedialog
import configparser # For managing persistent configuration parameters
import zipfile # For extracting downloaded model archives
import requests # For downloading the model file

# --- Theme Colors ---
LIGHT_THEME = {
    "bg": "#F0F0F0",        # Default window background
    "fg": "#000000",        # Default text color
    "entry_bg": "#FFFFFF",
    "entry_fg": "#000000",
    "text_bg": "#FFFFFF",
    "text_fg": "#000000",
    "button_fg": "#000000", # Black text on buttons
    "status_bg": "#F0F0F0",
    "status_fg": "#000000",
    "frame_bg": "#F0F0F0",
    "frame_fg": "#000000", # LabelFrame title color
    "disabled_fg": "#A0A0A0",
}

# Consistent DARK_THEME (Dark backgrounds, Light Text where possible, Black text on problematic widgets for contrast)
DARK_THEME = {
    "bg": "#2E2E2E",       # Dark background for main window/frames
    "fg": "#EAEAEA",       # Light text for labels etc. on dark bg
    "entry_bg": "#3C3C3C", # Try dark entry bg
    "entry_fg": "#000000", # Black text in entry/combobox for contrast
    "text_bg": "#1E1E1E",  # Very dark background for transcript area
    "text_fg": "#EAEAEA",  # Light text in transcript area
    "button_fg": "#000000", # Black text on buttons for contrast
    "status_bg": "#2E2E2E", # Dark background for status bar
    "status_fg": "#EAEAEA", # Light text on status bar
    "frame_bg": "#2E2E2E",  # Dark frame background
    "frame_fg": "#EAEAEA",  # Light text for LabelFrame titles
    "disabled_fg": "#777777", # Gray for disabled button text
}


# --- System State & Dependency Management ---
# Define required external library packages (dependencies)
# Added sounddevice back and soundfile
DEPENDENCIES = ["SpeechRecognition", "google-cloud-speech", "vosk", "numpy", "requests", "pyaudio", "openai-whisper", "sounddevice", "soundfile"]
# Define the local directory for storing these dependencies (akin to a local environment)
LIBS_DIR = "libs"
# Define standard location for automatically downloaded Vosk models
MODEL_BASE_DIR = "vosk_models" # Subdirectory for models relative to the script/exe

# --- Vosk Model Definitions ---
# Dictionary containing info for downloadable models
# Keys: User-friendly identifier (used in config and GUI)
# Values: Dict with 'url', 'extracted_dir_name', and 'description' including rough resource estimates
MODEL_INFO = {
    "small": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "extracted_dir_name": "vosk-model-small-en-us-0.15",
        "description": "Vosk Small US English (~45MB) - Fast, Lower Accuracy (CPU: Low, RAM: <0.5GB)"
    },
    "large": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "extracted_dir_name": "vosk-model-en-us-0.22",
        "description": "Vosk Large US English (~1.8GB) - Slower, Better Accuracy (CPU: High, RAM: 2-4GB+)"
    },
    "gigaspeech": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip",
        "extracted_dir_name": "vosk-model-en-us-0.42-gigaspeech",
        "description": "Vosk Gigaspeech US English (~2.6GB) - Slowest, Best Accuracy (CPU: Very High, RAM: 4-6GB+)"
    },
    # Add other models here if desired
}
DEFAULT_VOSK_MODEL_TYPE = "small" # Default Vosk model type if config is missing

# --- Whisper Model Definitions ---
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large"] # Standard Whisper sizes
DEFAULT_WHISPER_SIZE = "base" # Default Whisper size

def check_pip_availability():
    """Checks if pip is available for the current Python interpreter."""
    # This function is primarily for running the script directly with Python,
    # less critical when bundled with PyInstaller but kept for completeness.
    print("Verifying package manager (pip) availability...")
    try:
        # Use 'pip --version' as a simple check
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True, text=True)
        print(f"  [OK] Found pip: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        # This means sys.executable (python) was not found - very unlikely but possible
        print(f"  [System Error] Python executable not found at '{sys.executable}'. Cannot proceed.")
        return False
    except subprocess.CalledProcessError as e:
        # This means 'python -m pip' failed, likely because pip module isn't installed
        print(f"  [System Error] 'pip' module not found for Python at '{sys.executable}'.")
        print("      pip is required to manage dependencies.")
        print("\n      --- ACTION REQUIRED ---")
        print("      1. Try running: python -m ensurepip --upgrade")
        print("      2. If that fails, consider reinstalling or upgrading your Python distribution from python.org,")
        print("         ensuring the option to install pip is selected.")
        print("      -----------------------\n")
        return False
    except Exception as e:
        # Catch any other unexpected errors during the check
        print(f"  [System Error] Unexpected error while checking for pip: {e}")
        return False

def install_dependencies():
    """Verifies presence of required libraries, installs locally if absent, checks for updates if present."""
    # NOTE: This function is typically NOT called when running the bundled PyInstaller application,
    # as PyInstaller includes the dependencies found during the build process.
    # It's primarily for running the .py script directly.
    print("Initializing system: Verifying external library states...")

    # --- Check for pip first ---
    if not check_pip_availability():
        return False # Halt if pip is not usable

    # --- Proceed with dependency installation/update ---
    libs_path = os.path.abspath(LIBS_DIR) # Absolute path to local library store
    os.makedirs(libs_path, exist_ok=True) # Ensure the directory exists

    # Prepend the local library path to Python's module search path
    if libs_path not in sys.path:
        sys.path.insert(0, libs_path)
        print(f"System Path Modification: Prioritizing local library path '{libs_path}'.")

    all_requirements_met = True # Assume system state is initially valid
    # Base command for invoking the pip package manager, targeting the local directory
    pip_command_base = [
        sys.executable, "-m", "pip", "install",
        "--target", libs_path, # Installation locus
        "--no-user", # Isolate from user-specific site-packages
        "--disable-pip-version-check", # Suppress verbose version checks
        "--no-cache-dir" # Avoid using pip cache (can be removed for potential speedup)
    ]

    for package in DEPENDENCIES:
        # Special handling for pyaudio on some systems might be needed, but try pip first
        install_needed = False # Reset flag for each package
        try:
            # Attempt to locate the module specification using Python's import mechanics
            # Need to map package names to actual module names for checking
            module_name = package
            if package == 'google-cloud-speech': module_name = 'google.cloud.speech'
            elif package == 'SpeechRecognition': module_name = 'speech_recognition'
            elif package == 'openai-whisper': module_name = 'whisper'
            elif package == 'pyaudio': module_name = '_portaudio' # PyAudio check is tricky, check internal? Or just try install. Let's just try install.
            elif package == 'sounddevice': module_name = 'sounddevice' # Added sounddevice check
            elif package == 'soundfile': module_name = 'soundfile' # Added soundfile check

            spec = None
            # Skip find_spec for pyaudio and soundfile (libsndfile dependency makes it tricky)
            if package not in ['pyaudio', 'soundfile']:
                spec = importlib.util.find_spec(module_name)
                if spec is None: raise ImportError

            # --- Package Found (or skipping check) - Check for Updates ---
            print(f"  [State OK] Found {package} (or skipping check). Probing for potential updates...")
            try:
                # Construct pip command to attempt an upgrade
                command = pip_command_base + ["--upgrade", package]
                # Execute pip, capturing output to avoid excessive console noise
                result = subprocess.run(command, check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    # Log deviation from expected zero return code (potential transient error)
                    print(f"      [System Warning] pip upgrade probe for '{package}' non-zero exit code ({result.returncode}):")
                    print(f"      PIP Stderr:\n{result.stderr}")
                    # System continues, assuming existing package is functional
            except Exception as e:
                 # Log failure during the update probe itself
                 print(f"      [System Warning] Update probe failed for '{package}': {e}")
                 # System continues, assuming existing package is functional

        except ImportError:
            # --- Package Not Found - Installation Required ---
            print(f"  [State Error] '{package}' not found. Initiating local installation into '{LIBS_DIR}'...")
            install_needed = True
            try:
                # Construct pip command for initial installation
                command = pip_command_base + [package]
                print(f"      Executing: {' '.join(command)}")
                # Execute pip, demanding success (check=True raises error on failure)
                subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"      Installation successful: '{package}' -> '{LIBS_DIR}'.")
                # Force Python to recognize the newly installed package
                importlib.invalidate_caches()
                # Re-check import after installation
                if package not in ['pyaudio', 'soundfile']: # Cannot easily check these imports this way
                    if importlib.util.find_spec(module_name) is None:
                        print(f"      [System Error] Installation reported success, but still cannot import '{module_name}'.")
                        all_requirements_met = False
            except subprocess.CalledProcessError as e:
                # Pip command execution failed
                print(f"      [System Error] Failed to install '{package}'. Exit Code: {e.returncode}")
                print(f"      PIP Stderr:\n{e.stderr}")
                print("\n      --- ACTION REQUIRED ---")
                print("      Installation failed. Possible reasons:")
                print("        - Network connection issue (check internet access).")
                print(f"       - Package requires system libraries or build tools (e.g., C++ compiler, PortAudio for PyAudio/sounddevice, libsndfile for soundfile, ffmpeg for Whisper).")
                print("        - Insufficient permissions to write to the 'libs' directory.")
                print("      Check the error message above from pip for specific details.")
                print(f"      Try installing manually: {' '.join(pip_command_base + [package])}")
                print("      -----------------------\n")
                all_requirements_met = False
            except FileNotFoundError:
                # Should have been caught by check_pip_availability, but handle defensively
                print(f"      [System Error] Failed to install '{package}'. 'pip' command unavailable.")
                all_requirements_met = False
            except Exception as e:
                # Catch-all for other unexpected installation failures
                print(f"      [System Error] Unexpected failure during installation of '{package}': {e}")
                all_requirements_met = False

    if all_requirements_met:
        print("Dependency state verification complete.")
    else:
        print("Dependency state verification finished with errors.")

    return all_requirements_met # Report overall success/failure status

# --- Execute Dependency Check at Script Initialization ---
# This now includes the pip check internally

# --- THIS BLOCK IS COMMENTED OUT FOR PYINSTALLER ---
# The check/install logic is generally not needed/desired in the bundled app,
# as PyInstaller includes dependencies found during the build.
# If running the .py script directly, uncomment this block.
# if not install_dependencies():
#     print("\nSystem HALT: Critical dependency or pip error prevents execution.")
#     # Attempt to display a GUI error if Tkinter is available
#     try:
#         tk_spec = importlib.util.find_spec("tkinter")
#         if tk_spec:
#             # Only import if available to avoid further errors
#             import tkinter as tk
#             from tkinter import messagebox
#             root = tk.Tk()
#             root.withdraw() # Suppress the empty root window
#             messagebox.showerror("Initialization Error", "Failed to initialize required libraries or pip.\nConsult console log for details and required actions.")
#             root.destroy()
#         else:
#              print("(Underlying Tkinter subsystem unavailable for GUI error message)")
#     except Exception as e:
#         print(f"(Exception during Tkinter error reporting: {e})")
#     sys.exit(1) # Terminate script execution
# --- END OF BLOCK COMMENTED OUT FOR PYINSTALLER ---


# --- Import Verified Dependencies ---
# These imports rely on the successful execution of install_dependencies() OR
# on the libraries being bundled correctly by PyInstaller.
try:
    import speech_recognition as sr # Use SpeechRecognition for audio input and API wrapping
    import sounddevice as sd # Audio signal acquisition interface - Re-added for Vosk
    import vosk # Still needed for offline engine
    import numpy # Often needed by audio libraries
    import requests # For HTTP requests (downloading model)
    import whisper # Needed for recognize_whisper
    import soundfile as sf # Explicitly import soundfile (though not directly used here, ensures it's found)
    # google-cloud-speech is imported implicitly by SpeechRecognition when needed
    # pyaudio is imported implicitly by SpeechRecognition
except ImportError as e:
     # This indicates a failure despite the earlier check, a critical state
     print(f"\n[FATAL SYSTEM ERROR] Failed to import essential library: {e}")
     print("If running bundled app: This might indicate a PyInstaller bundling issue or missing system dependency (like PortAudio for PyAudio/sounddevice, libsndfile for soundfile, or ffmpeg for Whisper).")
     print("If running .py script: Ensure dependencies were installed correctly (check console).")
     # Attempt GUI error message
     try:
        tk_spec = importlib.util.find_spec("tkinter")
        if tk_spec:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Import Error", f"Failed to import required library: {e}.\nConsult console log.")
            root.destroy()
        else:
            print("(Tkinter subsystem unavailable for GUI error message)")
     except Exception as e_gui:
        print(f"(Exception during Tkinter error reporting: {e_gui})")
     sys.exit(1)


# --- Configuration Parameter Handling ---
CONFIG_FILE = "config.ini" # File for persistent parameter storage
config = configparser.ConfigParser() # Instantiate configuration parser

# Define default parameter values (system defaults)
# Added Engine section, Whisper section
defaults = {
    'Paths': {'custom_model_path': ''}, # User can optionally specify an external Vosk model path here
    'Models': {'preferred_vosk_model_type': DEFAULT_VOSK_MODEL_TYPE, 'model_directory': MODEL_BASE_DIR},
    'Whisper': {'model_size': DEFAULT_WHISPER_SIZE}, # Added Whisper config
    'Engine': {'type': 'vosk', 'google_cloud_credentials_json': ''}, # Added engine config
    # MODIFICATION: Changed input_device_name to audio_source_name for clarity
    'Audio': {'audio_source_name': '', 'log_file': 'live_transcription.log'},
    'Settings': {'enable_logging': 'True', 'theme': 'light'} # Added theme default
}

def load_config():
    """Loads parameters from CONFIG_FILE or establishes default state."""
    global config
    # Determine base path (directory containing the script or executable)
    base_path = get_base_path()
    config_path = os.path.join(base_path, CONFIG_FILE)
    print(f"Using configuration file path: {config_path}")


    if not os.path.exists(config_path):
        print(f"Configuration state file {CONFIG_FILE} not found; establishing default state.")
        config.read_dict(defaults)
        try:
            with open(config_path, 'w') as configfile:
                config.write(configfile)
            print(f"Default configuration saved to {config_path}.")
        except IOError as e:
            print(f"Error persisting default state to {config_path}: {e}")
            # Operate using in-memory defaults if file persistence fails
            config.read_dict(defaults)
            return False # Indicate config needs setup
    else:
        print(f"Loading operational parameters from {config_path}")
        config.read(config_path)
        # Ensure configuration integrity by merging defaults for missing parameters
        needs_update = False
        for section, options in defaults.items():
            if not config.has_section(section):
                config.add_section(section)
                needs_update = True
            for option, value in options.items():
                # Only add default if option is truly missing
                if not config.has_option(section, option):
                    config.set(section, option, value)
                    needs_update = True
        # MODIFICATION: Rename old config option if present
        if config.has_option('Audio', 'input_device_name') and not config.has_option('Audio', 'audio_source_name'):
            print("Migrating old config option 'input_device_name' to 'audio_source_name'.")
            config.set('Audio', 'audio_source_name', config.get('Audio', 'input_device_name'))
            config.remove_option('Audio', 'input_device_name')
            needs_update = True

        if needs_update:
            try:
                with open(config_path, 'w') as configfile:
                    config.write(configfile)
                print("Updated parameter file with default values for missing options.")
            except IOError as e:
                print(f"Error updating parameter file {config_path}: {e}")

    # --- Config Validation ---
    preferred_vosk_type = config.get('Models', 'preferred_vosk_model_type', fallback=DEFAULT_VOSK_MODEL_TYPE)
    custom_path = config.get('Paths', 'custom_model_path', fallback='')
    engine_type = config.get('Engine', 'type', fallback='vosk')
    google_creds = config.get('Engine', 'google_cloud_credentials_json', fallback='')
    whisper_size = config.get('Whisper', 'model_size', fallback=DEFAULT_WHISPER_SIZE)

    # Validate Vosk model settings if Vosk engine is selected
    if engine_type == 'vosk':
        if preferred_vosk_type == 'custom' and not custom_path:
             print("\n[Configuration Warning] Vosk engine selected with type 'custom' but no path is set.")
             print("Please use the Settings panel to specify a valid 'Custom Model Path'.\n")
        elif preferred_vosk_type not in MODEL_INFO and preferred_vosk_type != 'custom':
            print(f"\n[Configuration Warning] Invalid 'preferred_vosk_model_type' ('{preferred_vosk_type}') in config.ini.")
            print(f"Falling back to default '{DEFAULT_VOSK_MODEL_TYPE}'. Consider setting via Settings panel.\n")
            config.set('Models', 'preferred_vosk_model_type', DEFAULT_VOSK_MODEL_TYPE) # Correct invalid value

    # Validate Google Cloud settings if selected
    elif engine_type == 'google_cloud':
        if not google_creds:
            print("\n[Configuration Warning] Google Cloud engine selected but no credentials path is set.")
            print("Please use the Settings panel to specify the path to your Google Cloud JSON key file.\n")

    # Validate Whisper settings if selected
    elif engine_type == 'whisper':
         if whisper_size not in WHISPER_MODEL_SIZES:
              print(f"\n[Configuration Warning] Invalid Whisper 'model_size' ('{whisper_size}') in config.ini.")
              print(f"Falling back to default '{DEFAULT_WHISPER_SIZE}'. Consider setting via Settings panel.\n")
              config.set('Whisper', 'model_size', DEFAULT_WHISPER_SIZE) # Correct invalid value


    return True # Indicate configuration loaded

def save_config():
    """Saves the current configuration state to CONFIG_FILE."""
    global config
    # Determine base path (directory containing the script or executable)
    base_path = get_base_path()
    config_path = os.path.join(base_path, CONFIG_FILE)

    try:
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        print(f"Configuration saved to {config_path}")
        return True
    except IOError as e:
        print(f"Error saving configuration to {config_path}: {e}")
        # Ensure messagebox has a parent if called from settings window context
        # Use root as fallback parent
        parent_window = root
        if 'app' in globals() and app and app.settings_window and app.settings_window.winfo_exists():
             parent_window = app.settings_window
        messagebox.showerror("Config Error", f"Could not save settings to {CONFIG_FILE}:\n{e}", parent=parent_window)
        return False


# --- Vosk Model Download and Management ---
def get_base_path():
    """Gets the base path for the application (script dir or executable dir)."""
    if getattr(sys, 'frozen', False):
        # If running as a bundled app (PyInstaller)
        return os.path.dirname(sys.executable)
    else:
        # If running as a script
        return os.path.dirname(os.path.abspath(__file__))

def download_and_extract_model(model_key, settings_window=None):
    """Downloads and extracts the specified Vosk model zip file."""
    if model_key not in MODEL_INFO:
        err_msg = f"Unknown model key '{model_key}' specified for download."
        print(f"[ERROR] {err_msg}")
        if settings_window: messagebox.showerror("Download Error", err_msg, parent=settings_window)
        return False

    model_details = MODEL_INFO[model_key]
    url = model_details['url']
    expected_model_dir_name = model_details['extracted_dir_name']
    description = model_details['description']

    base_app_path = get_base_path()
    target_base_dir = os.path.join(base_app_path, config.get('Models', 'model_directory', fallback=MODEL_BASE_DIR))
    model_zip_path = os.path.join(target_base_dir, f"vosk_model_{model_key}.zip") # Unique zip name per model
    final_model_path = os.path.join(target_base_dir, expected_model_dir_name)

    print(f"Attempting to download model [{description}]...")
    print(f"URL: {url}")
    print(f"Target Directory: {target_base_dir}")
    # Update GUI status via queue if called from worker, or directly if called from settings
    status_msg = f"Downloading model: {description}... Please wait."
    if threading.current_thread() is not threading.main_thread():
        # This function should primarily be called from the GUI thread (Settings) now
        # gui_queue.put(("status", status_msg)) # Avoid queue if possible
        if app: app.status_var.set(status_msg); app.root.update_idletasks()
    elif app: # Update status bar directly if called from GUI thread (Settings)
         app.status_var.set(status_msg)
         app.root.update_idletasks()


    try:
        os.makedirs(target_base_dir, exist_ok=True) # Ensure target directory exists

        # Download using requests with streaming for large files
        with requests.get(url, stream=True, timeout=120) as r: # Increased timeout further
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0
            chunk_size = 8192 # Process in 8KB chunks
            last_update_time = time.time()

            print(f"Downloading to {model_zip_path} ({total_size / (1024*1024):.1f} MB)...")
            with open(model_zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # Update progress roughly every second or so to avoid spamming console/GUI queue
                        current_time = time.time()
                        if current_time - last_update_time > 0.5 or downloaded_size == total_size:
                            progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                            progress_mb = downloaded_size / (1024*1024)
                            total_mb = total_size / (1024*1024)
                            print(f"  Progress: {progress_mb:.1f} / {total_mb:.1f} MB ({progress:.0f}%)", end='\r')
                            # Send progress to GUI status bar
                            status_update = f"Downloading model: {progress:.0f}% ({progress_mb:.1f}MB)"
                            if threading.current_thread() is not threading.main_thread():
                                # gui_queue.put(("status", status_update))
                                if app: app.status_var.set(status_update); app.root.update_idletasks()
                            elif app: # Update status bar directly
                                 app.status_var.set(status_update)
                                 app.root.update_idletasks() # Force GUI update
                            last_update_time = current_time


        print("\nDownload complete. Extracting archive...")
        status_msg = "Extracting downloaded model..."
        if threading.current_thread() is not threading.main_thread(): gui_queue.put(("status", status_msg))
        elif app: app.status_var.set(status_msg); app.root.update_idletasks()


        # Extract the zip file
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_base_dir)
        print(f"Extraction complete. Model placed in: {target_base_dir}")

        # Verify expected directory exists after extraction
        if not os.path.exists(final_model_path):
             err_msg = f"Expected model directory '{final_model_path}' not found after extraction!"
             print(f"[ERROR] {err_msg}")
             # Try to list contents to help debug
             try:
                 extracted_items = os.listdir(target_base_dir)
                 print(f"      Contents of '{target_base_dir}': {extracted_items}")
             except Exception: pass
             if settings_window: messagebox.showerror("Extraction Error", err_msg, parent=settings_window)
             return False # Indicate failure

    except requests.exceptions.RequestException as e:
        err_msg = f"Model download failed: {e}"
        print(f"\n[ERROR] {err_msg}")
        if threading.current_thread() is not threading.main_thread(): gui_queue.put(("error", err_msg))
        elif app: messagebox.showerror("Download Error", err_msg, parent=settings_window); app.status_var.set("ERROR: Model download failed.")
        return False
    except zipfile.BadZipFile:
        err_msg = f"Model extraction failed: Downloaded file '{model_zip_path}' is not a valid zip archive."
        print(f"\n[ERROR] {err_msg}")
        if threading.current_thread() is not threading.main_thread(): gui_queue.put(("error", "Model extraction failed: Invalid zip file."))
        elif app: messagebox.showerror("Extraction Error", err_msg, parent=settings_window); app.status_var.set("ERROR: Model extraction failed.")
        return False
    except Exception as e:
        err_msg = f"Model setup failed: {e}"
        print(f"\n[ERROR] An unexpected error occurred during model download/extraction: {e}")
        if threading.current_thread() is not threading.main_thread(): gui_queue.put(("error", err_msg))
        elif app: messagebox.showerror("Setup Error", err_msg, parent=settings_window); app.status_var.set("ERROR: Model setup failed.")
        return False
    finally:
        # Clean up the downloaded zip file regardless of success/failure
        if os.path.exists(model_zip_path):
            try:
                os.remove(model_zip_path)
                print(f"Cleaned up temporary file: {model_zip_path}")
            except OSError as e:
                print(f"[Warning] Could not remove temporary zip file '{model_zip_path}': {e}")

    # Report success via GUI
    status_msg = f"Model '{model_key}' ready in {target_base_dir}."
    if threading.current_thread() is not threading.main_thread(): gui_queue.put(("status", status_msg))
    elif app: app.status_var.set(status_msg)

    return True # Indicate success

# --- Global State Variables ---
# Re-add Vosk specific audio queue
vosk_audio_queue = queue.Queue() # Separate queue for Vosk audio data
gui_queue = queue.Queue()   # FIFO buffer for message passing to the UI thread
stop_event = threading.Event() # Used to signal termination for background listener AND Vosk worker
# Handles for different audio mechanisms
sd_stream = None # sounddevice stream handle (for Vosk)
vosk_worker_thread = None # Vosk processing thread handle
sr_recognizer = None # Global recognizer instance from SpeechRecognition (for Whisper/Google)
sr_microphone = None # Global microphone instance from SpeechRecognition
sr_audio_source = None # Global audio source instance from SpeechRecognition
sr_background_listener_stop_func = None # Function to stop SR background listening

selected_device_index = None # Index identifier for the chosen audio input device
is_running = False # Boolean flag indicating active transcription state
current_vosk_model_path = None # Store the loaded Vosk model path
google_creds_json = None # Store loaded Google credentials content
current_whisper_size = None # Store selected Whisper model size
stop_handler_thread = None # Thread to handle stopping audio sources

# --- Audio Subsystem Functions ---
# MODIFIED list_audio_devices function
def list_audio_devices():
    """Lists available audio input and loopback devices using sounddevice."""
    print("Listing available audio sources (Input & Loopback via sounddevice)...")
    devices = {}
    try:
        sd_devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_input_idx = sd.default.device[0] # Default input device index
        default_output_idx = sd.default.device[1] # Default output device index

        print(f"Default Input Device Index: {default_input_idx}")
        print(f"Default Output Device Index: {default_output_idx}")

        for i, device in enumerate(sd_devices):
            device_name = device['name']
            hostapi_idx = device['hostapi']
            hostapi_name = hostapis[hostapi_idx]['name']

            # Mark default devices
            default_marker = ""
            if i == default_input_idx and i == default_output_idx:
                default_marker = " (Default Input & Output)"
            elif i == default_input_idx:
                default_marker = " (Default Input)"
            elif i == default_output_idx:
                default_marker = " (Default Output)"


            # --- Criteria for including a device as a potential source ---
            include_device = False
            source_type = ""

            # 1. Standard Input Devices
            if device['max_input_channels'] > 0:
                include_device = True
                source_type = "Input"

            # 2. Loopback Devices (Windows WASAPI specific identification)
            #    WASAPI loopback devices typically have input channels = 0
            #    but are associated with an output device.
            #    Their names often contain "(loopback)".
            if sys.platform == 'win32' and hostapi_name == 'Windows WASAPI':
                 # Check if it's an output device that might have a loopback pair
                 # Note: sounddevice doesn't directly expose a "is_loopback" flag.
                 # We rely on naming convention or listing output devices as potential sources.
                 # A common loopback name is based on the corresponding output device.
                 if device['max_output_channels'] > 0 and device['max_input_channels'] == 0:
                     # Tentatively include output devices under WASAPI as potential loopback sources
                     # The user might need to manually enable "Stereo Mix" or similar in Windows Sound settings
                     # if a dedicated loopback device isn't listed.
                     # Or, a more direct loopback device might be listed separately (often with 0 output channels)
                     include_device = True
                     source_type = "Output/Loopback?"

                 # Also check for devices explicitly named loopback by the driver
                 if 'loopback' in device_name.lower() and device['max_input_channels'] > 0:
                     include_device = True # Explicit loopback device found
                     source_type = "Loopback"


            # 3. Default Output Device (as potential loopback source if not already included)
            #    Sometimes the default output device itself can be recorded from using loopback.
            if i == default_output_idx and not include_device and device['max_output_channels'] > 0:
                 # Let's add the default output explicitly, user might want to try capturing it
                 include_device = True
                 source_type = "Default Output (Try Loopback?)"


            if include_device:
                 list_entry_name = f"{i}: {device_name}{default_marker} [{source_type} - {hostapi_name}]"
                 devices[list_entry_name] = i # Store display name -> index
                 print(f"  Found: {list_entry_name} (In:{device['max_input_channels']}, Out:{device['max_output_channels']})")


        if not devices:
            print(" No suitable audio sources found by sounddevice.")
            # Fallback to SpeechRecognition list if sounddevice fails? Less ideal.
            try:
                 print(" Falling back to SpeechRecognition/PyAudio device list...")
                 mic_names = sr.Microphone.list_microphone_names()
                 devices = {f"{i}: {name} (PyAudio Fallback)": i for i, name in enumerate(mic_names)}
            except Exception as e_sr:
                 print(f" Error listing devices via SpeechRecognition fallback: {e_sr}")

    except Exception as e:
        print(f" Error listing devices via sounddevice: {e}")
        # Attempt fallback to SpeechRecognition listing
        try:
            print(" Error during sounddevice query. Falling back to SpeechRecognition/PyAudio device list...")
            mic_names = sr.Microphone.list_microphone_names()
            devices = {f"{i}: {name} (PyAudio Fallback)": i for i, name in enumerate(mic_names)}
        except Exception as e_sr:
            print(f" Error listing devices via SpeechRecognition fallback: {e_sr}")


    print(f"--- Final Device List ({len(devices)} sources) ---")
    # for name in devices.keys(): print(f" - {name}") # Optionally print final list
    print("-----------------------------------------")

    # We now return the dictionary generated primarily by sounddevice
    return devices
# END OF MODIFIED list_audio_devices function


# --- Vosk Specific Audio Callback & Worker ---
def vosk_audio_callback(indata, frames, time_info, status):
    """Callback executed by sounddevice upon receiving a new audio data frame (for Vosk)."""
    if status: # Report any non-nominal status flags
        gui_queue.put(("status", f"Audio System Status (Vosk): {status}"))
    # Enqueue the raw audio data (numpy array converted to bytes) if system is active
    if is_running and not stop_event.is_set():
        # Check data type - needs to be bytes
        try:
            vosk_audio_queue.put(bytes(indata))
        except Exception as e:
             print(f"Error converting/queuing Vosk audio data: {e}") # Should not happen with correct dtype
             gui_queue.put(("error", f"Vosk audio data error: {e}"))
             # Optionally stop here?
             # stop_event.set()


def vosk_transcription_worker():
    """Dedicated thread for Vosk audio data processing and speech-to-text conversion."""
    global current_vosk_model_path, selected_device_index # Access globals
    log_file = None # File handle for logging output
    last_log_time = time.time() # Timestamp for periodic log flushing
    # Retrieve parameters from configuration
    enable_logging = config.getboolean('Settings', 'enable_logging', fallback=True)
    log_file_path_str = config.get('Audio', 'log_file', fallback='live_transcription.log')

    try:
        # --- Determine Model Path (MUST be set and valid) ---
        # This logic should already be validated before starting the thread, but double-check
        if not current_vosk_model_path or not os.path.exists(current_vosk_model_path):
             raise FileNotFoundError(f"Vosk worker started without a valid model path: {current_vosk_model_path}")

        # --- Initialize Logging Subsystem ---
        base_app_path = get_base_path()
        if not os.path.isabs(log_file_path_str):
             log_file_path = os.path.join(base_app_path, log_file_path_str)
        else:
             log_file_path = log_file_path_str

        if enable_logging:
            try:
                log_dir = os.path.dirname(log_file_path)
                if log_dir: os.makedirs(log_dir, exist_ok=True)
                log_file = open(log_file_path, "a", encoding="utf-8")
                gui_queue.put(("status", f"Logging enabled: {log_file_path}"))
            except Exception as e:
                gui_queue.put(("status", f"Log file error: {e}. Logging disabled."))
                enable_logging = False

        # --- Initialize Vosk Speech Recognition Engine ---
        gui_queue.put(("status", f"Initializing Vosk engine (Model: {os.path.basename(current_vosk_model_path)})..."))
        model = vosk.Model(current_vosk_model_path) # Load the acoustic/language model

        # Query selected audio device for its native sample rate (using sounddevice index)
        print(f"Attempting to use sounddevice index for Vosk: {selected_device_index}")
        device_info = sd.query_devices(selected_device_index) # Get info for the selected index
        # Determine if it's an input or loopback - Vosk needs sample rate
        samplerate = int(device_info['default_samplerate']) # Use device's sample rate (Hz)
        if samplerate == 0: # Sample rate might be 0 for some loopback devices, try default output rate?
             print(f"Warning: Selected device {selected_device_index} has 0Hz sample rate. Trying default output device rate.")
             default_output_idx = sd.default.device[1]
             default_output_info = sd.query_devices(default_output_idx)
             samplerate = int(default_output_info['default_samplerate'])
             if samplerate == 0: # Still zero? Fallback to a common rate
                  print("Warning: Default output also 0Hz. Falling back to 48000 Hz for Vosk.")
                  samplerate = 48000

        # Instantiate the recognizer with the model and sample rate
        recognizer = vosk.KaldiRecognizer(model, samplerate)
        recognizer.SetWords(True) # Request word-level timing information (optional)
        # recognizer.SetPartialWords(True) # Vosk partial results less straightforward to integrate here

        gui_queue.put(("status", f"Listening via '{device_info['name']}' (Sample Rate: {samplerate} Hz)")) # Inform UI

        # --- Main Processing Loop ---
        while not stop_event.is_set(): # Continue until termination signal
            try:
                # Dequeue audio data; block for max 0.5s if queue is empty
                data = vosk_audio_queue.get(timeout=0.5)

                # Feed audio data chunk to the Vosk recognizer
                if recognizer.AcceptWaveform(data):
                    result_json = recognizer.Result() # Get final recognition result
                    result_dict = json.loads(result_json) # Parse JSON result
                    if result_dict.get("text"): # Check if text was recognized
                        text = result_dict["text"]
                        timestamp = datetime.now().strftime("%H:%M:%S") # Generate timestamp
                        # Send complete transcript segment to the GUI queue
                        gui_queue.put(("transcript", f"[{timestamp}] {text}"))
                        gui_queue.put(("status", "Listening...")) # Reset status
                        # Log to file if enabled
                        if enable_logging and log_file:
                            log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
                            # Periodically flush buffer to disk (e.g., every 5 seconds)
                            current_time = time.time()
                            if current_time - last_log_time > 5.0:
                                log_file.flush()
                                last_log_time = current_time
                # else: # Handle partial results if enabled
                #     partial_json = recognizer.PartialResult()
                #     partial_dict = json.loads(partial_json)
                #     if partial_dict.get("partial"):
                #          gui_queue.put(("partial", partial_dict["partial"]))

            except queue.Empty:
                # Queue was empty within the timeout; signifies silence or processing catch-up
                continue # Loop again to wait for more data
            except Exception as e:
                 # Log processing errors to the GUI status
                 gui_queue.put(("status", f"Vosk Worker Error: {e}"))
                 time.sleep(1) # Avoid busy-looping on persistent errors

    # --- Error Handling & Cleanup ---
    except sd.PortAudioError as pae:
        # Specific error if sounddevice fails to open stream (e.g., wrong index, format mismatch)
        print(f"Sounddevice/PortAudio Error in Vosk worker: {pae}")
        error_detail = f"Audio Device Error (Vosk): {pae}\nCheck device index '{selected_device_index}' or try another source/engine."
        # Add WASAPI specific hint for loopback
        if sys.platform == 'win32' and 'Windows WASAPI' in str(pae) and 'Invalid number of channels' in str(pae):
             error_detail += "\n\nHint: If using loopback, ensure 'Stereo Mix' (or similar) is enabled and set as default recording device in Windows Sound settings, OR try selecting the specific output device directly."
        gui_queue.put(("error", error_detail))
    except FileNotFoundError as e:
        gui_queue.put(("error", f"Vosk/Model Error: {e}"))
    except RuntimeError as e:
        gui_queue.put(("error", f"Vosk Setup Error: {e}"))
    except Exception as e:
        gui_queue.put(("error", f"Critical Vosk Worker Error: {e}"))
    finally:
        # This block executes regardless of exceptions, ensuring cleanup
        gui_queue.put(("status", "Vosk worker shutting down..."))
        if enable_logging and log_file:
            try:
                log_file.flush(); log_file.close()
                gui_queue.put(("status", "Log file closed."))
            except Exception as e:
                 gui_queue.put(("status", f"Error closing log file: {e}"))
        # Ensure the main loop knows to stop if the worker terminates unexpectedly
        stop_event.set()


# --- SpeechRecognition Engine Callback ---
# MODIFIED process_audio_callback function (from previous step)
def process_audio_callback(r, audio):
    """Callback executed by SpeechRecognition's listen_in_background (for Whisper/Google)."""

    global google_creds_json, current_whisper_size # Access necessary globals

    if stop_event.is_set(): # Check if we should stop processing
        return

    engine = config.get('Engine', 'type', fallback='vosk') # Check engine again in case it changed? Risky. Assume it's Whisper/Google here.
    timestamp = datetime.now().strftime("%H:%M:%S") # Timestamp for log entry

    try:
        text = ""
        gui_queue.put(("status", f"Processing with {engine}...")) # Indicate processing start

        if engine == 'google_cloud':
            if not google_creds_json:
                raise RuntimeError("Google Cloud engine selected but no valid credentials loaded.")
            text = r.recognize_google_cloud(audio, credentials_json=google_creds_json)

        elif engine == 'whisper':
            if not current_whisper_size:
                 raise RuntimeError("Whisper engine selected but no model size configured.")
            # print(f"Using Whisper model: {current_whisper_size}") # Debug: Keep this print commented out or remove
            # May need language="en" depending on audio / desired outcome
            text = r.recognize_whisper(audio, model=current_whisper_size, language="english")

        else:
             # This callback should only be active for SR engines
             # MODIFICATION: Removed print statement
             # print(f"Warning: process_audio_callback called for unexpected engine: {engine}")
             gui_queue.put(("status", f"Warning: Invalid engine '{engine}' in callback.")) # Report via GUI
             return

        if text: # Only queue if text is recognized
             gui_queue.put(("transcript", f"[{timestamp}] {text}"))
             gui_queue.put(("status", "Listening...")) # Reset status after successful transcript

    except sr.WaitTimeoutError: pass # Should not happen with listen_in_background
    except sr.UnknownValueError:
        gui_queue.put(("status", "Listening... (No speech detected)"))
        pass
    except sr.RequestError as e:
        # MODIFICATION: Removed print statement
        # print(f"API request failed for engine '{engine}': {e}")
        gui_queue.put(("error", f"API Error ({engine}): {e}"))
    except Exception as e:
        # MODIFICATION: Removed print statement
        # print(f"Unexpected error during recognition with {engine}: {e}")
        # Make the error message more specific for the GUI
        gui_queue.put(("error", f"Recognition Error ({engine}): {e}"))
        # Also print to console IF possible, but don't crash if not
        try:
            # Use a different way to print that might work even if stdout is broken
            with open(os.devnull, "w") as fnull: # Try opening devnull to see if basic IO works
                print(f"*** Recognition Error ({engine}): {e} ***", file=fnull) # Try writing somewhere valid
            # Or attempt to write to original stderr if it exists
            if sys.__stderr__:
                 print(f"*** Recognition Error ({engine}): {e} ***", file=sys.__stderr__)
        except Exception:
            pass # Ignore if printing fails

# END OF MODIFIED process_audio_callback function


# --- Stop Handler Function (runs in separate thread) ---
# MODIFIED _perform_stop function (from previous step)
def _perform_stop():
    """Handles the actual stopping of audio streams/listeners in a background thread."""
    # MODIFICATION START: Add global is_running access to set it False reliably
    global is_running, sr_background_listener_stop_func, sd_stream, vosk_worker_thread
    global sr_recognizer, sr_microphone, sr_audio_source # Clear SR globals too

    print("Stop handler thread started.")
    engine_stopped = False

    try: # Wrap main stopping logic in try...finally
        # --- Stop SR Listener ---
        if sr_background_listener_stop_func:
            try:
                print("Calling SR background listener stop function...")
                sr_background_listener_stop_func(wait_for_stop=False) # Non-blocking stop
                print("SR Background listener stop function called.")
                engine_stopped = True
            except Exception as e:
                 print(f"Error calling SR background listener stop function: {e}")
                 gui_queue.put(("error", f"Error stopping listener: {e}")) # Report error to GUI
        # Clear SR globals whether stop worked or not
        sr_background_listener_stop_func = None
        sr_recognizer = None
        sr_microphone = None
        sr_audio_source = None

        # --- Stop Vosk Stream and Worker ---
        # Stop stream first to prevent callback putting more data in queue
        if sd_stream: # Check if stream object exists
            print("Attempting to stop sounddevice stream...")
            try:
                if sd_stream.active: # Check if active before stopping
                     print("  Calling sd_stream.stop()...")
                     sd_stream.stop()
                     print("  Calling sd_stream.close()...")
                     sd_stream.close()
                print("Sounddevice stream stopped/closed.")
                engine_stopped = True
            except Exception as e:
                 # MODIFICATION: Improved error reporting for stream stop/close
                 print(f"Error stopping/closing sounddevice stream: {e}")
                 gui_queue.put(("error", f"Error stopping audio stream: {e}")) # Report error
        sd_stream = None # Clear handle

        # Wait briefly for worker thread (it checks stop_event and queue timeout)
        # We still signal it via stop_event, but don't join from GUI thread
        if vosk_worker_thread and vosk_worker_thread.is_alive():
             print("Signaled Vosk worker thread to stop (no join)...")
             # Let the daemon thread exit on its own or when app closes
        vosk_worker_thread = None # Clear the thread handle

    except Exception as e_outer:
        # Catch unexpected errors in the main try block of _perform_stop
        print(f"Unexpected error during stop sequence: {e_outer}")
        gui_queue.put(("error", f"Unexpected stop error: {e_outer}"))
    finally:
        # --- Signal GUI Update ---
        # MODIFICATION: Moved GUI update and is_running = False into finally block
        print("Stop handler thread finished execution (inside finally block).")
        # Send message to re-enable controls and set final status
        # This will now run even if errors occurred during stopping
        gui_queue.put(("enable_controls", True))
        is_running = False # Reliably set is_running to False here
        print("Sent enable_controls message and set is_running=False.")
# END OF MODIFIED _perform_stop function


# --- Graphical User Interface (Tkinter) Application Class ---
class TranscriberApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Live Transcriber (Online/Offline)") # Updated title
        # Set initial and minimum size for the main window
        self.root.geometry("750x500") # MODIFICATION: Increased width slightly for longer device names
        self.root.minsize(750, 400)   # MODIFICATION: Increased min width
        # Register callback for window close event (X button)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.settings_window = None # Placeholder for the settings Toplevel window
        self.style = ttk.Style() # Store style object

        # --- Top Frame for Controls ---
        self.controls_frame = ttk.Frame(root_window, padding="10", style="Controls.TFrame") # Assign style
        self.controls_frame.pack(side=tk.TOP, fill=tk.X) # Pack at top, fill horizontally

        # MODIFICATION: Changed label text
        self.controls_label = ttk.Label(self.controls_frame, text="Audio Source:", style="Controls.TLabel") # Store ref
        self.controls_label.pack(side=tk.LEFT, padx=(0, 5)) # Assign style

        self.devices = list_audio_devices() # MODIFICATION: Use updated device listing function
        self.device_var = tk.StringVar() # Tkinter variable bound to combobox
        # Create dropdown menu for device selection
        # MODIFICATION: Increased combobox width for longer names
        self.device_combobox = ttk.Combobox(self.controls_frame, textvariable=self.device_var,
                                            values=list(self.devices.keys()), state="readonly", width=50)
        if not self.devices: # Handle case where no input devices are found
             self.device_var.set("No audio sources detected")
             self.device_combobox.config(state=tk.DISABLED)
        else:
            # Attempt to load last used device from configuration
            # MODIFICATION: Use updated config key 'audio_source_name'
            saved_device_name = config.get('Audio', 'audio_source_name', fallback='')
            # Use a more robust check: Exact match first, then partial
            exact_match = [k for k in self.devices.keys() if saved_device_name == k]
            if exact_match:
                 self.device_var.set(exact_match[0])
            else:
                # Check if saved name is PART of any current device name (less reliable)
                partial_matches = [k for k in self.devices.keys() if saved_device_name and saved_device_name in k] # Simple substring match
                if partial_matches:
                    self.device_var.set(partial_matches[0])
                else:
                    # Try to select default input device if available
                    default_input_keys = [k for k in self.devices.keys() if '(Default Input)' in k]
                    if default_input_keys:
                         self.device_var.set(default_input_keys[0])
                    else:
                         # Fallback to first device in the list
                         self.device_var.set(list(self.devices.keys())[0])

            # Save the initially selected device name back to config (in case it was a fallback)
            config.set('Audio', 'audio_source_name', self.device_var.get())


        self.device_combobox.pack(side=tk.LEFT, padx=5)
        # MODIFICATION: Add callback for device change to save selection
        self.device_combobox.bind('<<ComboboxSelected>>', self.on_device_change)


        # Start/Stop control buttons
        self.start_button = ttk.Button(self.controls_frame, text="Start Acquisition", command=self.start_transcription, style="Controls.TButton") # Assign style
        if not self.devices: self.start_button.config(state=tk.DISABLED) # Disable if no devices
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(self.controls_frame, text="Stop Acquisition", command=self.stop_transcription, state=tk.DISABLED, style="Controls.TButton") # Assign style
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Settings Button - Added
        self.settings_button = ttk.Button(self.controls_frame, text="Settings", command=self.open_settings_window, style="Controls.TButton") # Assign style
        self.settings_button.pack(side=tk.LEFT, padx=15) # Add some padding


        # --- Transcript Display Area ---
        # Attempt to use a more common modern font based on OS
        try: transcript_font = font.Font(family="Segoe UI", size=10) # Windows preference
        except:
             try: transcript_font = font.Font(family="San Francisco", size=11) # macOS preference
             except: transcript_font = font.Font(family="Helvetica", size=11) # Generic fallback

        # Create scrollable text widget for displaying results
        self.transcript_area = scrolledtext.ScrolledText(root_window, wrap=tk.WORD, state=tk.DISABLED, font=transcript_font, relief=tk.SOLID, borderwidth=1)
        # Configure colors in apply_theme
        self.transcript_area.pack(padx=10, pady=(0, 5), expand=True, fill=tk.BOTH) # Fill available space

        # --- Status Bar (Bottom Section) ---
        self.status_var = tk.StringVar() # Tkinter variable for status text
        self.status_bar = ttk.Label(root_window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5, style="Status.TLabel") # Assign style
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X) # Pack at bottom, fill horizontally
        # MODIFICATION: Updated initial status message
        self.status_var.set("System Idle. Select audio source and Start Acquisition.")

        # Apply initial theme based on config
        self.apply_theme(config.get('Settings', 'theme', fallback='light'))

        # Initiate periodic polling of the GUI message queue
        self.check_gui_queue()

    # --- MODIFICATION START: Add callback for device selection change ---
    def on_device_change(self, event=None):
        """Saves the selected device name to config when changed."""
        selected_name = self.device_var.get()
        if selected_name and selected_name != "No audio sources detected":
            print(f"Audio source changed to: {selected_name}")
            config.set('Audio', 'audio_source_name', selected_name)
            # No need to call save_config() here unless persistence between sessions
            # without closing is strictly required. load_config() handles persistence.
            # If immediate persistence is desired:
            # save_config()
    # --- MODIFICATION END ---


    def apply_theme(self, mode):
        """Applies the selected color theme (light/dark) to widgets."""
        theme = DARK_THEME if mode == 'dark' else LIGHT_THEME
        fg_col = theme['fg']
        bg_col = theme['bg']
        entry_fg = theme['entry_fg']
        entry_bg = theme['entry_bg']
        text_fg = theme['text_fg']
        text_bg = theme['text_bg']
        btn_fg = theme['button_fg']
        # btn_bg = theme['button_bg'] # Button BG is unreliable with ttk themes
        frame_fg = theme['frame_fg']
        status_fg = theme['status_fg']
        status_bg = theme['status_bg']

        # Configure root window background
        self.root.config(bg=bg_col)

        # Configure ttk styles (more reliable for ttk widgets)
        self.style.configure("TFrame", background=bg_col)
        self.style.configure("Controls.TFrame", background=bg_col) # Specific style for controls frame
        self.style.configure("TLabel", background=bg_col, foreground=fg_col)
        self.style.configure("Status.TLabel", background=status_bg, foreground=status_fg) # Specific style for status bar
        # Configure Button: Only set foreground reliably
        self.style.configure("TButton", foreground=btn_fg)
        # Map allows configuring state-specific colors (e.g., disabled)
        self.style.map("TButton",
                       foreground=[('disabled', theme['disabled_fg']), ('active', btn_fg)])
                       # background=[...] # Avoid setting background for TButton
        # Configure Combobox and Entry
        self.style.configure("TCombobox", foreground=entry_fg, fieldbackground=entry_bg) # Set text and field bg
        self.style.map('TCombobox',
                       fieldbackground=[('readonly', entry_bg)], # Ensure field bg applies
                       foreground=[('readonly', entry_fg)],      # Ensure text color applies
                       selectbackground=[('readonly', status_bg)], # Selection highlight bg
                       selectforeground=[('readonly', status_fg)]) # Selection highlight text
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=entry_fg)
        # Configure Checkbutton and Radiobutton (background might be inherited)
        self.style.configure("TCheckbutton", background=bg_col, foreground=fg_col)
        self.style.configure("TRadiobutton", background=bg_col, foreground=fg_col)
        # Configure LabelFrame
        self.style.configure("TLabelFrame", background=bg_col, bordercolor=fg_col) # Border color might not work on all themes
        self.style.configure("TLabelFrame.Label", background=bg_col, foreground=frame_fg)


        # Configure specific widgets (especially non-ttk ones like ScrolledText)
        self.controls_frame.config(style="Controls.TFrame") # Apply style
        # Re-apply styles to ensure they take effect after initial creation
        for widget in self.controls_frame.winfo_children():
             if isinstance(widget, ttk.Label): widget.config(style="Controls.TLabel")
             elif isinstance(widget, ttk.Button): widget.config(style="Controls.TButton")
             elif isinstance(widget, ttk.Combobox): widget.config(style="TCombobox")
             # Add other ttk widget types if needed

        self.transcript_area.config(background=text_bg, foreground=text_fg,
                                    insertbackground=fg_col) # Cursor color
        self.status_bar.config(style="Status.TLabel") # Apply style

        # Apply theme to settings window if it exists
        if self.settings_window and self.settings_window.winfo_exists():
            self.apply_theme_to_window(self.settings_window, mode) # Reuse helper


    def apply_theme_to_window(self, window, mode):
        """Applies theme colors specifically to a given window and its ttk children."""
        theme = DARK_THEME if mode == 'dark' else LIGHT_THEME
        bg_col = theme['bg']
        # fg_col = theme['fg'] # Not needed directly here if styles are used

        window.config(bg=bg_col)

        # Apply styles to all ttk widgets within the window
        # No need to reconfigure styles here, just apply them to widgets
        for widget in window.winfo_children():
            self.update_widget_style(widget) # Use recursive helper

    def update_widget_style(self, widget):
         """Recursively apply styles to widgets based on current theme."""
         # Determine current theme mode
         current_theme_mode = config.get('Settings', 'theme', fallback='light')
         theme = DARK_THEME if current_theme_mode == 'dark' else LIGHT_THEME
         bg_col = theme['bg']
         fg_col = theme['fg']

         widget_class = widget.winfo_class()
         # Map Tkinter class names to ttk style names (approximate)
         style_map = {
             'Frame': 'TFrame', 'LabelFrame': 'TLabelFrame', 'Label': 'TLabel',
             'Button': 'TButton', 'Entry': 'TEntry', 'Checkbutton': 'TCheckbutton',
             'Radiobutton': 'TRadiobutton', 'Combobox': 'TCombobox'
         }
         # Add specific styles if defined
         # Check existence of controls_frame before accessing children
         controls_frame_children = []
         # Check if self has controls_frame AND if it's not None (safer)
         if hasattr(self, 'controls_frame') and self.controls_frame:
             # Check if widget is a child of controls_frame
             if widget in self.controls_frame.winfo_children():
                 if isinstance(widget, ttk.Label): style_name = "Controls.TLabel"
                 elif isinstance(widget, ttk.Button): style_name = "Controls.TButton"
                 else: style_name = style_map.get(widget_class)
             elif hasattr(self, 'status_bar') and widget is self.status_bar: # Check status bar specifically
                 style_name = "Status.TLabel"
             else:
                 style_name = style_map.get(widget_class)
         elif hasattr(self, 'status_bar') and widget is self.status_bar: # Check status bar if controls_frame doesn't exist yet
              style_name = "Status.TLabel"
         else:
             style_name = style_map.get(widget_class)


         if style_name:
             try:
                 widget.config(style=style_name)
                 # Explicitly set background for frames/labelframes for better consistency
                 if widget_class in ['Frame', 'LabelFrame']:
                      widget.config(style=style_name) # Re-apply style might be needed
             except tk.TclError: # Widget might not be a ttk widget despite class name
                 try:
                     # Fallback for standard tk widgets if needed
                     widget.config(bg=bg_col, fg=fg_col)
                 except tk.TclError:
                     pass # Widget doesn't support bg/fg
         else:
             # Apply theme to non-ttk widgets if possible
             try:
                 # Special handling for ScrolledText background/foreground
                 if isinstance(widget, scrolledtext.ScrolledText):
                      widget.config(bg=theme['text_bg'], fg=theme['text_fg'], insertbackground=fg_col)
                 else:
                      widget.config(bg=bg_col, fg=fg_col)
             except tk.TclError:
                 pass # Widget doesn't support bg/fg


         # Recurse into child widgets
         for child in widget.winfo_children():
             self.update_widget_style(child)


    def check_gui_queue(self):
        """Periodically checks the GUI message queue for updates from worker thread."""
        try:
            # Process all messages currently in the queue without blocking
            while True:
                message_type, message_content = gui_queue.get_nowait() # Non-blocking fetch

                # Update GUI based on message type
                if message_type == "transcript":
                    self.update_transcript(message_content)
                    # Update status bar if not showing partial result
                    if is_running and not self.status_var.get().startswith("Partial"): # Check is_running
                         self.status_var.set("Processing audio stream...")
                elif message_type == "partial":
                    # Partial results not directly supported by SR background listener callback easily
                    # Could potentially use streaming API for online services for this
                    pass # Ignore partial for now
                elif message_type == "status":
                    self.status_var.set(f"System Status: {message_content}")
                elif message_type == "error":
                     # Display critical errors via popup and status bar
                     messagebox.showerror("System Error", message_content)
                     self.status_var.set(f"CRITICAL ERROR: {message_content}")
                     # Attempt graceful shutdown on critical error
                     if is_running:
                         self.stop_transcription()
                         # (Existing elif for "error" should be just above here)
                elif message_type == "enable_controls":
                    # Message from _perform_stop thread after stopping completes
                    enable = message_content # Should be True
                    print("Received enable_controls message from stop handler.")
                    # Ensure is_running is actually False now (set in _perform_stop's finally block)
                    if not is_running:
                        self.start_button.config(state=tk.NORMAL if self.devices and enable else tk.DISABLED)
                        self.stop_button.config(state=tk.DISABLED) # Stop always disabled after stop finishes
                        self.device_combobox.config(state="readonly" if self.devices and enable else tk.DISABLED) # Use readonly
                        self.settings_button.config(state=tk.NORMAL if enable else tk.DISABLED)
                        if enable:
                            self.status_var.set("System Idle.")
                            self.update_transcript("--- Transcription Terminated ---")
                    else:
                        # This case should ideally not happen if _perform_stop works correctly
                        print("WARN: Received enable_controls but is_running is still True!")
                        # Force GUI update anyway, but log the inconsistency
                        self.start_button.config(state=tk.NORMAL if self.devices and enable else tk.DISABLED)
                        self.stop_button.config(state=tk.DISABLED)
                        self.device_combobox.config(state="readonly" if self.devices and enable else tk.DISABLED) # Use readonly
                        self.settings_button.config(state=tk.NORMAL if enable else tk.DISABLED)
                        self.status_var.set("System Idle (State Inconsistency?)")
                        # (Existing final else: block should be just below here)
                else:
                     # Log unexpected message types for debugging
                     print(f"Unknown GUI message type received: {message_type} - {message_content}")

        except queue.Empty:
            pass # No messages in queue, normal state
        except Exception as e:
             # Catch potential errors during GUI update itself
             print(f"Error processing GUI message queue: {e}")
        finally:
            # Reschedule this check function to run again after 100ms
            self.root.after(100, self.check_gui_queue)

    def update_transcript(self, text):
        """Appends a line of text to the transcript display area."""
        try:
            self.transcript_area.config(state=tk.NORMAL) # Enable editing
            self.transcript_area.insert(tk.END, text + "\n") # Append text and newline
            self.transcript_area.see(tk.END) # Ensure the new text is visible (auto-scroll)
            self.transcript_area.config(state=tk.DISABLED) # Disable editing
        except Exception as e:
             print(f"Error updating transcript display widget: {e}")
             self.status_var.set("Error: Failed to update transcript display.")


    def start_transcription(self):
        """Initiates the audio acquisition and transcription process."""
        global sr_recognizer, sr_microphone, sr_audio_source, sr_background_listener_stop_func
        global sd_stream, vosk_worker_thread # Vosk specific handles
        global is_running, current_vosk_model_path, google_creds_json, selected_device_index, current_whisper_size

        if is_running:
            messagebox.showwarning("System Busy", "Transcription process is already active.")
            return

        # --- Validate Device Selection ---
        selected_device_display_name = self.device_var.get()
        if not selected_device_display_name or selected_device_display_name == "No audio sources detected":
            messagebox.showerror("Configuration Error", "A valid audio source must be selected.")
            return
        try:
            # MODIFICATION: Get index from the self.devices dictionary using the display name
            selected_device_index = self.devices.get(selected_device_display_name)
            if selected_device_index is None:
                # This shouldn't happen if the combobox is populated correctly
                raise ValueError(f"Selected device '{selected_device_display_name}' not found in internal list.")
            print(f"Selected audio source: '{selected_device_display_name}' (Index: {selected_device_index})")

        except Exception as e: # Catch potential errors during index lookup
             messagebox.showerror("Device Error", f"Could not get device index for '{selected_device_display_name}':\n{e}")
             return

        # --- Load Engine Configuration ---
        engine_type = config.get('Engine', 'type', fallback='vosk')
        base_app_path = get_base_path()
        google_creds_json = None # Reset
        current_vosk_model_path = None # Reset
        current_whisper_size = None # Reset

        try:
            # --- Validate Engine Specific Config ---
            if engine_type == 'vosk':
                custom_model_path_str = config.get('Paths', 'custom_model_path', fallback='')
                preferred_type = config.get('Models', 'preferred_vosk_model_type', fallback=DEFAULT_VOSK_MODEL_TYPE)
                if preferred_type == 'custom' and custom_model_path_str:
                    if not os.path.isabs(custom_model_path_str): custom_model_path = os.path.join(base_app_path, custom_model_path_str)
                    else: custom_model_path = custom_model_path_str
                    if not os.path.exists(custom_model_path): raise FileNotFoundError(f"Custom Vosk model path specified in Settings ('{custom_model_path}') not found.")
                    current_vosk_model_path = custom_model_path
                elif preferred_type in MODEL_INFO:
                    model_details = MODEL_INFO[preferred_type]
                    model_dir_abs = os.path.join(base_app_path, config.get('Models', 'model_directory', fallback=MODEL_BASE_DIR))
                    expected_path = os.path.join(model_dir_abs, model_details['extracted_dir_name'])
                    if not os.path.exists(expected_path): raise FileNotFoundError(f"Preferred Vosk model '{preferred_type}' not found. Please download it via Settings.")
                    current_vosk_model_path = expected_path
                else: raise RuntimeError(f"Invalid Vosk model configuration (Type: '{preferred_type}'). Check Settings.")
                print(f"Using Vosk model: {current_vosk_model_path}")
                gui_queue.put(("status", f"Using Model: {os.path.basename(current_vosk_model_path)}"))

            elif engine_type == 'google_cloud':
                creds_path_str = config.get('Engine', 'google_cloud_credentials_json', fallback='')
                if not creds_path_str: raise FileNotFoundError("Google Cloud engine selected, but no credentials JSON path specified in Settings.")
                if not os.path.isabs(creds_path_str): creds_path = os.path.join(base_app_path, creds_path_str)
                else: creds_path = creds_path_str
                if not os.path.exists(creds_path): raise FileNotFoundError(f"Google Cloud credentials file not found at '{creds_path}'. Check Settings.")
                try:
                    with open(creds_path, "r") as f: google_creds_json = f.read()
                    print(f"Using Google Cloud credentials from: {creds_path}")
                    gui_queue.put(("status", "Using Engine: Google Cloud (Online)"))
                except Exception as e: raise IOError(f"Error reading Google Cloud credentials file '{creds_path}': {e}")

            elif engine_type == 'whisper':
                 current_whisper_size = config.get('Whisper', 'model_size', fallback=DEFAULT_WHISPER_SIZE)
                 if current_whisper_size not in WHISPER_MODEL_SIZES: raise ValueError(f"Invalid Whisper model size '{current_whisper_size}' configured. Choose from {WHISPER_MODEL_SIZES}")
                 print(f"Using Whisper model size: {current_whisper_size}")
                 gui_queue.put(("status", f"Using Engine: Whisper {current_whisper_size} (Offline)"))
            else:
                raise ValueError(f"Unsupported engine type configured: '{engine_type}'")

        except (FileNotFoundError, ValueError, RuntimeError, IOError) as e:
             messagebox.showerror("Configuration Error", f"Cannot start transcription:\n{e}")
             self.status_var.set(f"ERROR: {e}")
             return # Stop before starting audio

        # --- Reset state variables and queues before starting ---
        stop_event.clear() # Clear termination signal
        # Clear appropriate queue based on engine
        if engine_type == 'vosk':
            while not vosk_audio_queue.empty():
                try: vosk_audio_queue.get_nowait()
                except queue.Empty: break
        while not gui_queue.empty(): # Always clear GUI queue
            try: gui_queue.get_nowait()
            except queue.Empty: break

        self.status_var.set("Initiating audio stream...") # Update UI status

        # --- Start Audio Input and Processing based on Engine ---
        try:
            if engine_type == 'vosk':
                # --- Start Vosk Audio Input (sounddevice) ---
                # selected_device_index should be valid now from the updated list_audio_devices
                device_info = sd.query_devices(selected_device_index)
                samplerate = int(device_info['default_samplerate'])
                # Handle potentially 0 sample rate for loopback devices
                if samplerate == 0:
                     print(f"Warning: Selected device {selected_device_index} reports 0Hz sample rate. Trying default output rate.")
                     default_output_idx = sd.default.device[1]
                     default_output_info = sd.query_devices(default_output_idx)
                     samplerate = int(default_output_info['default_samplerate'])
                     if samplerate == 0:
                          print("Warning: Default output also 0Hz. Falling back to 48000 Hz for Vosk.")
                          samplerate = 48000

                print(f"Attempting to open sounddevice InputStream with device={selected_device_index}, samplerate={samplerate}")
                sd_stream = sd.InputStream(
                    samplerate=samplerate, device=selected_device_index,
                    channels=1, dtype='int16', # Use 1 channel for mono transcription input
                    callback=vosk_audio_callback,
                    blocksize=8000 # Adjust blocksize if needed
                )
                sd_stream.start()
                self.status_var.set("Audio stream active (Vosk). Initializing engine...")
                # Start Vosk worker thread
                vosk_worker_thread = threading.Thread(target=vosk_transcription_worker, name="VoskWorker")
                vosk_worker_thread.daemon = True
                vosk_worker_thread.start()

            else: # Whisper or Google Cloud using SpeechRecognition/PyAudio
                # --- Initialize SR Recognizer and Microphone ---
                # NOTE: SpeechRecognition/PyAudio might have more trouble accessing
                # WASAPI loopback devices compared to sounddevice. This might fail.
                print(f"Attempting to use SpeechRecognition/PyAudio with device_index={selected_device_index}")
                sr_recognizer = sr.Recognizer()
                # Adjust energy threshold dynamically? Maybe higher for loopback?
                sr_recognizer.energy_threshold = 400 # Start with default, adjust if needed
                sr_recognizer.pause_threshold = 0.8 # Time of silence before phrase ends
                # sr_recognizer.dynamic_energy_threshold = True # Let SR adjust energy

                try:
                    # Need to handle potential errors opening the device here
                    sr_microphone = sr.Microphone(device_index=selected_device_index)
                except Exception as e_mic:
                    # Catch errors specifically from Microphone() constructor
                    print(f"Error initializing sr.Microphone with index {selected_device_index}: {e_mic}")
                    # Try to provide a more helpful error message
                    err_msg = f"Failed to open audio source with SpeechRecognition/PyAudio (Index: {selected_device_index}):\n{e_mic}\n\n"
                    if "Invalid input device" in str(e_mic) or "Error opening stream" in str(e_mic) :
                         err_msg += "This might happen with loopback devices.\nTry using the Vosk engine (which uses sounddevice) or check if 'Stereo Mix' is enabled in Windows sound settings."
                    else:
                         err_msg += "Ensure the selected device is available and drivers are installed."
                    messagebox.showerror("Audio Source Error", err_msg)
                    self.status_var.set("ERROR: Failed to open audio source.")
                    return # Stop the start process


                self.status_var.set("Adjusting for ambient noise... Please wait.")
                self.root.update_idletasks()
                try:
                    with sr_microphone as source:
                        print("Adjusting for ambient noise...")
                        sr_recognizer.adjust_for_ambient_noise(source, duration=1.0)
                        print(f"Set minimum energy threshold to {sr_recognizer.energy_threshold:.2f}")
                    self.status_var.set("Ambient noise adjustment complete.")
                    sr_audio_source = sr_microphone # Store for background listener
                except Exception as e_adjust:
                    print(f"Error during ambient noise adjustment: {e_adjust}")
                    messagebox.showerror("Audio Source Error", f"Failed during ambient noise adjustment:\n{e_adjust}\n\nIs the selected audio source producing sound?")
                    self.status_var.set("ERROR: Failed noise adjustment.")
                    return # Stop the start process


                # --- Start SR Background Listening ---
                gui_queue.put(("status", "Starting background listener..."))
                sr_background_listener_stop_func = sr_recognizer.listen_in_background(
                    sr_audio_source,
                    process_audio_callback,
                    phrase_time_limit=10 # Max duration of a phrase before processing
                )
                print("Background listener started.")

            # --- Update GUI State ---
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.device_combobox.config(state=tk.DISABLED) # Disable during run
            self.settings_button.config(state=tk.DISABLED)
            is_running = True # Set is_running True HERE
            self.update_transcript(f"--- Transcription Initiated (Engine: {engine_type}) ---")
            self.status_var.set("Listening...")

        except sd.PortAudioError as pae:
             # Specific handling for sounddevice stream errors
             print(f"Sounddevice/PortAudio Error during startup: {pae}")
             error_detail = f"Audio Device Error: {pae}\nCheck device index '{selected_device_index}' or try another source/engine."
             if sys.platform == 'win32' and 'Windows WASAPI' in str(pae) and 'Invalid number of channels' in str(pae):
                  error_detail += "\n\nHint: If using loopback, ensure 'Stereo Mix' (or similar) is enabled and set as default recording device in Windows Sound settings, OR try selecting the specific output device directly."
             messagebox.showerror("Audio Init Error", error_detail)
             self.status_var.set("ERROR: Failed to start audio stream.")
             is_running = False # Ensure state reflects failure
             self.start_button.config(state=tk.NORMAL if self.devices else tk.DISABLED)
             self.stop_button.config(state=tk.DISABLED)
             self.device_combobox.config(state="readonly" if self.devices else tk.DISABLED)
             self.settings_button.config(state=tk.NORMAL)

        except Exception as e:
            # Handle other errors during stream or thread initialization
            print(f"Generic error during startup: {e}")
            messagebox.showerror("Initialization Error", f"Failed to start transcription process:\n{e}")
            # Attempt cleanup
            if sd_stream and sd_stream.active: sd_stream.stop(); sd_stream.close(); sd_stream = None
            if vosk_worker_thread and vosk_worker_thread.is_alive(): stop_event.set(); # vosk_worker_thread.join(0.5) # Don't join here
            if sr_background_listener_stop_func: sr_background_listener_stop_func(wait_for_stop=False); sr_background_listener_stop_func = None
            is_running = False # Ensure state reflects failure
            # Reset GUI control states
            self.start_button.config(state=tk.NORMAL if self.devices else tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
            self.device_combobox.config(state="readonly" if self.devices else tk.DISABLED)
            self.settings_button.config(state=tk.NORMAL)
            self.status_var.set("System Error during startup. Verify configuration and device.")


    # MODIFIED stop_transcription function (from previous step)
    def stop_transcription(self):
        """Signals termination and starts the background stop handler."""
        global is_running, stop_handler_thread # Added stop_handler_thread global
        if not is_running: # Ignore if already stopped
            print("Stop ignored: Already stopped (is_running is False).")
            return

        # Prevent multiple stop attempts while handler is running
        if self.stop_button['state'] == tk.DISABLED and stop_handler_thread and stop_handler_thread.is_alive():
            print("Stop ignored: Stop process already running.")
            return
            # MODIFICATION START: Removed redundant check and empty return
            # ---- Start of code block to remove ----
        # Or if stop button is disabled for other reasons (already stopped)
        # if self.stop_button['state'] == tk.DISABLED:
        #    print("Stop ignored: Stop button is already disabled.") # Added print for clarity
        #    return
            # ---- End of code block to remove ----
            # MODIFICATION END

        self.status_var.set("Stopping listener/stream...")
        # Disable Stop button immediately to prevent multiple clicks
        self.stop_button.config(state=tk.DISABLED)
        # Keep Start/Settings disabled until stop is confirmed via GUI queue
        self.start_button.config(state=tk.DISABLED)
        self.settings_button.config(state=tk.DISABLED)
        self.device_combobox.config(state=tk.DISABLED)

        stop_event.set() # Signal threads/callbacks to stop

        # Start the background thread to handle actual stopping
        print("Starting stop handler thread...")
        stop_handler_thread = threading.Thread(target=_perform_stop, name="StopHandler")
        stop_handler_thread.daemon = True
        stop_handler_thread.start()

        # Note: is_running will be set to False and GUI re-enabled
        # via a message ("enable_controls") from the _perform_stop thread's finally block
    # END OF MODIFIED stop_transcription function


    def on_closing(self):
        """Callback function executed when the main window close button is pressed."""
        # Also ensure settings window is closed if open
        if self.settings_window and self.settings_window.winfo_exists():
             self.settings_window.destroy()

        # Save config on close
        save_config()

        if is_running: # Check if transcription is active
            print("Close requested while running. Stopping transcription...")
            self.stop_transcription() # Initiate graceful shutdown
            # Schedule window destruction shortly after to allow cleanup
            self.root.after(500, self.root.destroy)
        else:
            # If not running, destroy the window immediately
            self.root.destroy()

    # --- Settings Window Logic ---
    def open_settings_window(self):
        """Creates and displays the Toplevel settings window."""
        # Prevent opening multiple settings windows
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift() # Bring existing window to front
            return

        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Settings")
        # Remove fixed size and minsize to allow auto-sizing based on content
        # self.settings_window.geometry("600x450") # REMOVED
        self.settings_window.minsize(600, 550)   # Set minimum size only - Increased height
        self.settings_window.resizable(True, True) # Allow resizing
        self.settings_window.transient(self.root) # Keep on top of main window
        self.settings_window.grab_set() # Make modal (disable main window interaction)

        # --- Variables to hold settings ---
        self.engine_type_var = tk.StringVar(value=config.get('Engine', 'type', fallback='vosk'))
        self.google_creds_var = tk.StringVar(value=config.get('Engine', 'google_cloud_credentials_json', fallback=''))
        self.preferred_vosk_model_type_var = tk.StringVar(value=config.get('Models', 'preferred_vosk_model_type', fallback=DEFAULT_VOSK_MODEL_TYPE))
        self.custom_model_path_var = tk.StringVar(value=config.get('Paths', 'custom_model_path', fallback=''))
        self.whisper_model_size_var = tk.StringVar(value=config.get('Whisper', 'model_size', fallback=DEFAULT_WHISPER_SIZE)) # Whisper size var
        self.log_file_path_var = tk.StringVar(value=config.get('Audio', 'log_file', fallback='live_transcription.log'))
        self.enable_logging_var = tk.BooleanVar(value=config.getboolean('Settings', 'enable_logging', fallback=True))
        self.theme_var = tk.StringVar(value=config.get('Settings', 'theme', fallback='light')) # Theme variable

        # Create frames for organization
        # Use pack with expand=True, fill='both' for frames inside settings to help with resizing
        engine_frame = ttk.LabelFrame(self.settings_window, text="Transcription Engine", padding="10")
        engine_frame.pack(pady=5, padx=10, fill="x", expand=False)

        vosk_model_frame = ttk.LabelFrame(self.settings_window, text="Vosk Model Selection (if Vosk engine selected)", padding="10")
        vosk_model_frame.pack(pady=5, padx=10, fill="x", expand=False) # Don't expand model frame vertically

        whisper_frame = ttk.LabelFrame(self.settings_window, text="Whisper Settings (if Whisper engine selected)", padding="10") # Whisper frame
        whisper_frame.pack(pady=5, padx=10, fill="x", expand=False)

        google_frame = ttk.LabelFrame(self.settings_window, text="Google Cloud Settings (if Google engine selected)", padding="10")
        google_frame.pack(pady=5, padx=10, fill="x", expand=False)

        log_frame = ttk.LabelFrame(self.settings_window, text="Logging", padding="10")
        log_frame.pack(pady=5, padx=10, fill="x", expand=False) # Don't expand log frame vertically

        theme_frame = ttk.LabelFrame(self.settings_window, text="Appearance", padding="10") # Theme frame
        theme_frame.pack(pady=5, padx=10, fill="x", expand=False) # Don't expand theme frame vertically

        button_frame = ttk.Frame(self.settings_window, padding="10")
        button_frame.pack(pady=10, side=tk.BOTTOM) # Pack buttons at bottom

        # --- Engine Frame Widgets ---
        ttk.Label(engine_frame, text="Engine:").pack(side=tk.LEFT, padx=5)
        vosk_rb = ttk.Radiobutton(engine_frame, text="Vosk (Offline)", variable=self.engine_type_var, value="vosk", command=self.toggle_engine_settings)
        vosk_rb.pack(side=tk.LEFT, padx=5)
        whisper_rb = ttk.Radiobutton(engine_frame, text="Whisper (Offline)", variable=self.engine_type_var, value="whisper", command=self.toggle_engine_settings) # Added Whisper RB
        whisper_rb.pack(side=tk.LEFT, padx=5)
        gcp_rb = ttk.Radiobutton(engine_frame, text="Google Cloud (Online)", variable=self.engine_type_var, value="google_cloud", command=self.toggle_engine_settings)
        gcp_rb.pack(side=tk.LEFT, padx=5)

        # --- Vosk Model Frame Widgets ---
        ttk.Label(vosk_model_frame, text="Preferred Model:").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,5)) # Span 3 columns

        row_num = 1
        for key, info in MODEL_INFO.items():
            rb = ttk.Radiobutton(vosk_model_frame, text=info['description'], variable=self.preferred_vosk_model_type_var, value=key)
            rb.grid(row=row_num, column=0, columnspan=3, sticky="w", padx=5) # Span 3 columns
            row_num += 1

        self.vosk_download_button = ttk.Button(vosk_model_frame, text="Download/Set Selected", command=self.download_and_set_preferred) # Store ref
        self.vosk_download_button.grid(row=row_num, column=0, columnspan=3, pady=(5,10))
        row_num += 1

        self.vosk_custom_rb = ttk.Radiobutton(vosk_model_frame, text="Use Custom Model Path:", variable=self.preferred_vosk_model_type_var, value="custom") # Store ref
        self.vosk_custom_rb.grid(row=row_num, column=0, sticky="w", padx=5, pady=5)

        self.vosk_custom_model_entry = ttk.Entry(vosk_model_frame, textvariable=self.custom_model_path_var, width=45) # Store ref
        self.vosk_custom_model_entry.grid(row=row_num, column=1, padx=(0,5), pady=5, sticky="ew") # Start entry in col 1
        self.vosk_custom_model_browse = ttk.Button(vosk_model_frame, text="Browse...", command=self.browse_custom_model_path) # Store ref
        self.vosk_custom_model_browse.grid(row=row_num, column=2, padx=5, pady=5)
        vosk_model_frame.columnconfigure(1, weight=1) # Make entry expand

        # --- Whisper Frame Widgets ---
        ttk.Label(whisper_frame, text="Model Size:").grid(row=0, column=0, sticky="w", pady=5, padx=5)
        col_num = 1
        self.whisper_rb_list = [] # Store radiobuttons to enable/disable
        for size in WHISPER_MODEL_SIZES:
             rb = ttk.Radiobutton(whisper_frame, text=size.capitalize(), variable=self.whisper_model_size_var, value=size)
             rb.grid(row=0, column=col_num, sticky="w", padx=5)
             self.whisper_rb_list.append(rb)
             col_num += 1
        ttk.Label(whisper_frame, text="(Model downloaded automatically on first use per size)").grid(row=1, column=0, columnspan=col_num, sticky="w", padx=5, pady=(5,0))


        # --- Google Cloud Frame Widgets ---
        ttk.Label(google_frame, text="Credentials JSON File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.google_creds_entry = ttk.Entry(google_frame, textvariable=self.google_creds_var, width=50) # Store ref
        self.google_creds_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.google_creds_browse = ttk.Button(google_frame, text="Browse...", command=self.browse_google_creds) # Store ref
        self.google_creds_browse.grid(row=0, column=2, padx=5, pady=5)
        google_frame.columnconfigure(1, weight=1)

        # --- Log Frame Widgets ---
        ttk.Label(log_frame, text="Log File Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        log_entry = ttk.Entry(log_frame, textvariable=self.log_file_path_var, width=50)
        log_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        log_browse = ttk.Button(log_frame, text="Browse...", command=self.browse_log_path)
        log_browse.grid(row=0, column=2, padx=5, pady=5)

        log_check = ttk.Checkbutton(log_frame, text="Enable Transcription Logging", variable=self.enable_logging_var)
        log_check.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        log_frame.columnconfigure(1, weight=1) # Make entry expand

        # --- Theme Frame Widgets --- Added
        ttk.Label(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=5)
        light_rb = ttk.Radiobutton(theme_frame, text="Light", variable=self.theme_var, value="light")
        light_rb.pack(side=tk.LEFT, padx=5)
        dark_rb = ttk.Radiobutton(theme_frame, text="Dark", variable=self.theme_var, value="dark")
        dark_rb.pack(side=tk.LEFT, padx=5)


        # --- Button Frame Widgets ---
        save_button = ttk.Button(button_frame, text="Save Settings", command=self.save_settings_and_close)
        save_button.pack(side=tk.LEFT, padx=10)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.settings_window.destroy)
        cancel_button.pack(side=tk.LEFT, padx=10)

        # Apply theme to the settings window itself
        self.apply_theme_to_window(self.settings_window, self.theme_var.get())
        # Initial toggle based on loaded engine type
        self.toggle_engine_settings()

        # Center the settings window initially (optional)
        self.settings_window.update_idletasks() # Update geometry info
        # Don't force position if allowing auto-size
        # x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (self.settings_window.winfo_width() // 2)
        # y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (self.settings_window.winfo_height() // 2)
        # self.settings_window.geometry(f"+{x}+{y}")

        # Make window appear after setup
        self.settings_window.deiconify()

    def toggle_engine_settings(self):
        """Enable/disable engine-specific settings based on selection."""
        engine = self.engine_type_var.get()

        # Define widget groups (ensure widgets exist before adding)
        vosk_widgets = []
        if hasattr(self, 'vosk_download_button'): vosk_widgets.append(self.vosk_download_button)
        if hasattr(self, 'vosk_custom_rb'): vosk_widgets.append(self.vosk_custom_rb)
        if hasattr(self, 'vosk_custom_model_entry'): vosk_widgets.append(self.vosk_custom_model_entry)
        if hasattr(self, 'vosk_custom_model_browse'): vosk_widgets.append(self.vosk_custom_model_browse)
        # Find Vosk model radio buttons and label
        if self.settings_window and self.settings_window.winfo_exists():
             for widget in self.settings_window.winfo_children():
                  if isinstance(widget, ttk.LabelFrame) and "Vosk Model Selection" in widget.cget("text"):
                       for child in widget.winfo_children():
                            if isinstance(child, ttk.Radiobutton) and child is not getattr(self, 'vosk_custom_rb', None):
                                 vosk_widgets.append(child)
                            elif isinstance(child, ttk.Label) and "Preferred Model" in child.cget("text"):
                                 vosk_widgets.append(child)

        whisper_widgets = []
        if hasattr(self, 'whisper_rb_list'): whisper_widgets.extend(self.whisper_rb_list)
        # Find Whisper labels
        if self.settings_window and self.settings_window.winfo_exists():
             for widget in self.settings_window.winfo_children():
                  if isinstance(widget, ttk.LabelFrame) and "Whisper Settings" in widget.cget("text"):
                       for child in widget.winfo_children():
                            if isinstance(child, ttk.Label):
                                 whisper_widgets.append(child)


        google_widgets = []
        if hasattr(self, 'google_creds_entry'): google_widgets.append(self.google_creds_entry)
        if hasattr(self, 'google_creds_browse'): google_widgets.append(self.google_creds_browse)
        # Find Google Cloud label
        if self.settings_window and self.settings_window.winfo_exists():
             for widget in self.settings_window.winfo_children():
                  if isinstance(widget, ttk.LabelFrame) and "Google Cloud Settings" in widget.cget("text"):
                       for child in widget.winfo_children():
                            if isinstance(child, ttk.Label):
                                 google_widgets.append(child)

        # Enable/Disable based on selected engine
        for w in vosk_widgets: w.config(state=tk.NORMAL if engine == 'vosk' else tk.DISABLED)
        for w in whisper_widgets: w.config(state=tk.NORMAL if engine == 'whisper' else tk.DISABLED)
        for w in google_widgets: w.config(state=tk.NORMAL if engine == 'google_cloud' else tk.DISABLED)


    def apply_theme_to_window(self, window, mode):
        """Applies theme colors specifically to a given window and its ttk children."""
        theme = DARK_THEME if mode == 'dark' else LIGHT_THEME
        bg_col = theme['bg']
        # fg_col = theme['fg'] # Not needed directly here if styles are used

        window.config(bg=bg_col)

        # Apply styles to all ttk widgets within the window
        # No need to reconfigure styles here, just apply them to widgets
        for widget in window.winfo_children():
            self.update_widget_style(widget) # Use recursive helper

    def update_widget_style(self, widget):
         """Recursively apply styles to widgets based on current theme."""
         # Determine current theme mode
         current_theme_mode = config.get('Settings', 'theme', fallback='light')
         theme = DARK_THEME if current_theme_mode == 'dark' else LIGHT_THEME
         bg_col = theme['bg']
         fg_col = theme['fg']

         widget_class = widget.winfo_class()
         # Map Tkinter class names to ttk style names (approximate)
         style_map = {
             'Frame': 'TFrame', 'LabelFrame': 'TLabelFrame', 'Label': 'TLabel',
             'Button': 'TButton', 'Entry': 'TEntry', 'Checkbutton': 'TCheckbutton',
             'Radiobutton': 'TRadiobutton', 'Combobox': 'TCombobox'
         }
         # Add specific styles if defined
         # Check existence of controls_frame before accessing children
         controls_frame_children = []
         # Check if self has controls_frame AND if it's not None (safer)
         if hasattr(self, 'controls_frame') and self.controls_frame:
             # Check if widget is a child of controls_frame
             if widget in self.controls_frame.winfo_children():
                 if isinstance(widget, ttk.Label): style_name = "Controls.TLabel"
                 elif isinstance(widget, ttk.Button): style_name = "Controls.TButton"
                 else: style_name = style_map.get(widget_class)
             elif hasattr(self, 'status_bar') and widget is self.status_bar: # Check status bar specifically
                 style_name = "Status.TLabel"
             else:
                 style_name = style_map.get(widget_class)
         elif hasattr(self, 'status_bar') and widget is self.status_bar: # Check status bar if controls_frame doesn't exist yet
              style_name = "Status.TLabel"
         else:
             style_name = style_map.get(widget_class)


         if style_name:
             try:
                 widget.config(style=style_name)
                 # Explicitly set background for frames/labelframes for better consistency
                 if widget_class in ['Frame', 'LabelFrame']:
                      widget.config(style=style_name) # Re-apply style might be needed
             except tk.TclError: # Widget might not be a ttk widget despite class name
                 try:
                     # Fallback for standard tk widgets if needed
                     widget.config(bg=bg_col, fg=fg_col)
                 except tk.TclError:
                     pass # Widget doesn't support bg/fg
         else:
             # Apply theme to non-ttk widgets if possible
             try:
                 # Special handling for ScrolledText background/foreground
                 if isinstance(widget, scrolledtext.ScrolledText):
                      widget.config(bg=theme['text_bg'], fg=theme['text_fg'], insertbackground=fg_col)
                 else:
                      widget.config(bg=bg_col, fg=fg_col)
             except tk.TclError:
                 pass # Widget doesn't support bg/fg


         # Recurse into child widgets
         for child in widget.winfo_children():
             self.update_widget_style(child)


    def download_and_set_preferred(self):
        """Handles downloading the selected Vosk model and updating config."""
        selected_type = self.preferred_vosk_model_type_var.get() # Use correct var
        if selected_type == 'custom' or selected_type not in MODEL_INFO:
             messagebox.showwarning("Select Model", f"Please select a standard Vosk model ({', '.join(MODEL_INFO.keys())}) before downloading.", parent=self.settings_window)
             return

        model_details = MODEL_INFO[selected_type]
        expected_dir_name = model_details['extracted_dir_name']
        base_app_path = get_base_path()
        model_dir_abs = os.path.join(base_app_path, config.get('Models', 'model_directory', fallback=MODEL_BASE_DIR))
        expected_path = os.path.join(model_dir_abs, expected_dir_name)

        # Check if it already exists
        if os.path.exists(expected_path):
             if messagebox.askyesno("Model Exists", f"The '{selected_type}' model seems to already exist.\nDo you want to set it as preferred anyway (without re-downloading)?", parent=self.settings_window):
                  # Set as preferred without downloading
                  config.set('Models', 'preferred_vosk_model_type', selected_type)
                  config.set('Paths', 'custom_model_path', '') # Clear custom path when selecting standard
                  config.set('Engine', 'type', 'vosk') # Ensure engine is Vosk
                  self.engine_type_var.set('vosk') # Update GUI variable
                  if save_config():
                       messagebox.showinfo("Settings Updated", f"'{selected_type}' Vosk model set as preferred.", parent=self.settings_window)
                  self.toggle_engine_settings() # Update GUI state
             return # Exit function whether user clicked yes or no

        # If model doesn't exist, proceed with download
        # Disable settings window during download? Maybe just show progress.
        if download_and_extract_model(selected_type, settings_window=self.settings_window):
            # Download successful, now update config
            config.set('Models', 'preferred_vosk_model_type', selected_type)
            config.set('Paths', 'custom_model_path', '') # Clear custom path
            config.set('Engine', 'type', 'vosk') # Ensure engine is Vosk
            self.engine_type_var.set('vosk') # Update GUI variable
            if save_config():
                 messagebox.showinfo("Download Complete", f"'{selected_type}' model downloaded and set as preferred.", parent=self.settings_window)
            self.toggle_engine_settings() # Update GUI state
            # Optionally close settings window after successful download + save?
            # self.settings_window.destroy()
        else:
            # Error message shown by download_and_extract_model
            pass # Keep settings window open

    def browse_custom_model_path(self):
        """Opens a directory selection dialog for the model path."""
        directory = filedialog.askdirectory(title="Select Custom Vosk Model Folder", parent=self.settings_window) # Set parent
        if directory: # Only update if a directory was selected
            self.custom_model_path_var.set(directory)
            # Automatically select the 'custom' radiobutton when a path is chosen
            self.preferred_vosk_model_type_var.set("custom")
            # Also set engine type to Vosk if selecting a custom Vosk model path
            self.engine_type_var.set("vosk")
            self.toggle_engine_settings()


    def browse_log_path(self):
        """Opens a file save dialog for the log file path."""
        # Suggest current filename and directory
        initial_dir = os.path.dirname(self.log_file_path_var.get())
        initial_file = os.path.basename(self.log_file_path_var.get())
        # Use get_base_path() to default initialdir if current path is invalid
        if not initial_dir or not os.path.isdir(initial_dir):
             initial_dir = get_base_path()

        filepath = filedialog.asksaveasfilename(
            title="Select Log File Path",
            initialdir=initial_dir,
            initialfile=initial_file if initial_file else "live_transcription.log",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            parent=self.settings_window # Set parent
        )
        if filepath: # Only update if a path was selected/entered
            self.log_file_path_var.set(filepath)

    def browse_google_creds(self):
        """Opens a file open dialog for the Google Cloud credentials JSON file."""
        filepath = filedialog.askopenfilename(
            title="Select Google Cloud Credentials JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.settings_window
        )
        if filepath:
            self.google_creds_var.set(filepath)
            # Automatically select Google Cloud engine when creds are chosen
            self.engine_type_var.set("google_cloud")
            self.toggle_engine_settings()


    def save_settings_and_close(self):
        """Saves the settings from the dialog to config object and file, then closes."""
        global config
        try:
            # Get values from GUI variables
            engine_type = self.engine_type_var.get()
            google_creds_path = self.google_creds_var.get()
            preferred_vosk_type = self.preferred_vosk_model_type_var.get()
            custom_model_path = self.custom_model_path_var.get()
            whisper_model_size = self.whisper_model_size_var.get() # Get Whisper size
            log_file_path = self.log_file_path_var.get()
            enable_logging = self.enable_logging_var.get()
            selected_theme = self.theme_var.get() # Get selected theme
            # Get current audio source selection to save it (though it's also saved on change)
            # audio_source_name = self.device_var.get() # From main window

            # --- Validation ---
            if not log_file_path:
                 messagebox.showwarning("Validation Error", "Log file path cannot be empty.", parent=self.settings_window)
                 return
            if engine_type == 'vosk':
                if preferred_vosk_type == 'custom' and not custom_model_path:
                     messagebox.showwarning("Validation Error", "Please specify a path if 'Use Custom Model Path' is selected for Vosk, or choose a standard model.", parent=self.settings_window)
                     return
            elif engine_type == 'google_cloud':
                 if not google_creds_path:
                      messagebox.showwarning("Validation Error", "Please specify the path to your Google Cloud credentials JSON file.", parent=self.settings_window)
                      return
                 # Check existence only if path is not empty
                 if google_creds_path and not os.path.exists(google_creds_path):
                      messagebox.showwarning("Validation Error", f"Google Cloud credentials file not found at:\n{google_creds_path}", parent=self.settings_window)
                      return
            elif engine_type == 'whisper':
                 if whisper_model_size not in WHISPER_MODEL_SIZES:
                      messagebox.showwarning("Validation Error", f"Invalid Whisper model size selected: {whisper_model_size}", parent=self.settings_window)
                      return
            else: # Should not happen
                 messagebox.showerror("Validation Error", f"Invalid engine type selected: {engine_type}", parent=self.settings_window)
                 return

            # --- Update Config Object ---
            # Ensure sections exist before setting
            for section in ['Paths', 'Audio', 'Settings', 'Models', 'Engine', 'Whisper']: # Added Whisper section
                 if not config.has_section(section): config.add_section(section)

            config.set('Engine', 'type', engine_type)
            config.set('Engine', 'google_cloud_credentials_json', google_creds_path if engine_type == 'google_cloud' else '')

            config.set('Models', 'preferred_vosk_model_type', preferred_vosk_type if engine_type == 'vosk' else config.get('Models', 'preferred_vosk_model_type', fallback=DEFAULT_VOSK_MODEL_TYPE)) # Only save if relevant
            config.set('Paths', 'custom_model_path', custom_model_path if engine_type == 'vosk' and preferred_vosk_type == 'custom' else '') # Store custom path only if selected AND engine is Vosk

            config.set('Whisper', 'model_size', whisper_model_size if engine_type == 'whisper' else config.get('Whisper', 'model_size', fallback=DEFAULT_WHISPER_SIZE)) # Save Whisper size

            config.set('Audio', 'log_file', log_file_path)
            # Config value for audio source is updated by on_device_change callback now
            # config.set('Audio', 'audio_source_name', audio_source_name)

            config.set('Settings', 'enable_logging', str(enable_logging))
            config.set('Settings', 'theme', selected_theme) # Save theme setting

            # --- Save Config File ---
            if save_config():
                self.status_var.set("System Status: Settings saved successfully.")
                # Apply the theme immediately to the main window
                self.apply_theme(selected_theme)
                self.settings_window.destroy() # Close window on successful save
            else:
                # Error message shown by save_config
                self.status_var.set("System Status: Failed to save settings.")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings:\n{e}", parent=self.settings_window)
            self.status_var.set("System Status: Error saving settings.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Load configuration parameters at startup
    config_ok = load_config()
    # Configuration issues are now treated as warnings; proceed to dependency check

    # --- Dependency Check Block ---
    # This block is intended for running the .py script directly.
    # It should be COMMENTED OUT before using PyInstaller to bundle the application.
    # if not install_dependencies():
    #     print("\nSystem HALT: Critical dependency or pip error prevents execution.")
    #     try:
    #         tk_spec = importlib.util.find_spec("tkinter")
    #         if tk_spec:
    #             import tkinter as tk
    #             from tkinter import messagebox
    #             root = tk.Tk()
    #             root.withdraw()
    #             messagebox.showerror("Initialization Error", "Failed to initialize required libraries or pip.\nConsult console log for details and required actions.")
    #             root.destroy()
    #         else:
    #              print("(Underlying Tkinter subsystem unavailable for GUI error message)")
    #     except Exception as e:
    #         print(f"(Exception during Tkinter error reporting: {e})")
    #     sys.exit(1)
    # --- End of Dependency Check Block ---


    # Initialize the Tkinter GUI framework
    root = tk.Tk()
    # Define app globally so save_config and download function can use it as fallback parent
    app = TranscriberApp(root) # Instantiate the main application class
    # Display configuration warning in status bar if needed
    if not config_ok:
         app.status_var.set("Warning: Default config created. Check Settings.")
    else:
        # Check initial model configuration status after loading config
        engine_type = config.get('Engine', 'type', fallback='vosk')
        if engine_type == 'vosk':
            preferred_type = config.get('Models', 'preferred_vosk_model_type', fallback=DEFAULT_VOSK_MODEL_TYPE)
            custom_path = config.get('Paths', 'custom_model_path', fallback='')
            if preferred_type == 'custom' and not custom_path:
                app.status_var.set("Warning: Vosk 'Custom Model' selected but no path set. Use Settings.")
            elif preferred_type == 'custom':
                 # Resolve relative path for check
                 if not os.path.isabs(custom_path): custom_path = os.path.join(get_base_path(), custom_path)
                 if not os.path.exists(custom_path):
                     app.status_var.set(f"Warning: Vosk custom model path not found. Check Settings.")
            elif preferred_type in MODEL_INFO:
                 model_details = MODEL_INFO[preferred_type]
                 model_dir_abs = os.path.join(get_base_path(), config.get('Models', 'model_directory', fallback=MODEL_BASE_DIR))
                 expected_path = os.path.join(model_dir_abs, model_details['extracted_dir_name'])
                 if not os.path.exists(expected_path):
                      app.status_var.set(f"Warning: Preferred Vosk model '{preferred_type}' not downloaded. Use Settings.")
        elif engine_type == 'google_cloud':
             google_creds = config.get('Engine', 'google_cloud_credentials_json', fallback='')
             if not google_creds:
                  app.status_var.set("Warning: Google Cloud selected but no credentials set. Use Settings.")
             else:
                  # Resolve relative path for check
                  if not os.path.isabs(google_creds): google_creds = os.path.join(get_base_path(), google_creds)
                  if not os.path.exists(google_creds):
                       app.status_var.set("Warning: Google Cloud credentials file not found. Check Settings.")
        elif engine_type == 'whisper':
             whisper_size = config.get('Whisper', 'model_size', fallback=DEFAULT_WHISPER_SIZE)
             # Could add a check here if Whisper model file exists in cache, but SR handles download
             # MODIFICATION: Updated status message slightly
             app.status_var.set(f"Whisper engine selected ({whisper_size}). Model downloads on first use.")

        # Check if selected audio source still exists
        selected_source = config.get('Audio', 'audio_source_name', fallback='')
        if selected_source and selected_source not in app.devices.keys():
             app.status_var.set(f"Warning: Saved audio source '{selected_source}' not found. Select a valid source.")
        elif not selected_source and app.devices:
             app.status_var.set(f"Audio source set to default: {app.device_var.get()}.")


    # Start the Tkinter event loop (makes the window interactive)
    root.mainloop()