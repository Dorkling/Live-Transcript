Live Transcriber (Online/Offline)

A versatile real-time transcription application for Windows that captures audio from any source (microphone or system audio/loopback) and converts it to text using your choice of powerful transcription engines.

<p align="center">
<img src="https://drive.google.com/uc?id=1NFu7S5s9HIItL1fwX1HiK1EBEXE1aBUk" alt="Main Application Window" width="70%"/>
</p>

Features

    Multiple Transcription Engines: Choose the best engine for your needs.

        Vosk (Offline): Fully offline transcription. Ideal for privacy and environments without internet.

        Whisper (Offline): High-accuracy offline transcription using OpenAI's Whisper models.

        Google Cloud Speech-to-Text (Online): Best-in-class accuracy using Google's powerful online API.

    Real-Time Transcription: See the transcribed text appear live in the application window.

    System Audio & Microphone Support: Transcribe not only your voice from a microphone but also any audio playing on your computer (e.g., from a video, conference call, or game) using loopback.

    Highly Configurable:

        Select from various model sizes for Vosk and Whisper to balance speed, accuracy, and resource usage.

        Download Vosk models directly from within the app.

        Easily configure your Google Cloud credentials.

    Transcription Logging: Save all transcribed text to a log file with timestamps for later review.

    Light & Dark Themes: Switch between themes for your viewing comfort.

    Standalone Executable: No need to install Python or dependencies. Just download the release and run.

<p align="center">
<img src="https://drive.google.com/uc?id=1X3OKLpo8tq1kcPS9pnM6qQHlpnyf22J0" alt="Settings Window" width="70%"/>
</p>
How to Use

    Select Your Engine:

        Click the Settings button.

        Choose your desired engine: Vosk, Whisper, or Google Cloud.

        Configure the engine-specific settings (see below).

        Click Save Settings.

    Select an Audio Source:

        From the "Audio Source" dropdown menu, select the device you want to transcribe. This can be a physical microphone or a loopback device like "Stereo Mix" or "Broadcast Stream Mix" to capture system audio.

    Start and Stop:

        Click Start Acquisition. The application will begin listening. If you are using Whisper, it may take a moment to download the necessary model on the first use for a specific model size.

        Speak or play audio. The transcribed text will appear in the main window.

        Click Stop Acquisition to end the session.

Engine Details

Vosk (Offline)

    Setup: In the Settings window, select a Vosk model from the list and click Download/Set Selected. The application will download and extract the model for you. Alternatively, you can point to a local model folder using the "Use Custom Model Path" option.

    Use Case: Excellent for fully offline, private, and fast transcription where top-tier accuracy is not the primary concern.

Whisper (Offline)

    Setup: Simply select the "Whisper" engine and choose a model size. The first time you start an acquisition with a specific model size, the application will automatically download it. This may take some time depending on your internet speed and the model size.

    Use Case: State-of-the-art accuracy for offline transcription. The larger the model, the better the accuracy, but the higher the CPU/RAM requirements.

Google Cloud (Online)

    Setup:

        You must have a Google Cloud Platform account with the Speech-to-Text API enabled.

        Create a service account and download its credentials as a JSON key file.

        In the Settings window, select the "Google Cloud" engine.

        Click Browse... and select the JSON key file you downloaded.

    Use Case: Unmatched accuracy for critical applications where an internet connection is available.

    License

This project is currently unlicensed. Feel free to use and modify the code.
