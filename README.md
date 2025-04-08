# YouTube AI Clip Generator (Korean Subtitles & Social Post)

This tool automates the process of taking YouTube videos (with English audio), finding their most interesting parts, and turning them into short clips with Korean subtitles, complete with a suggested social media post in Korean!

Think of it as your personal AI assistant for creating engaging video highlights. It uses smart AI technology (from OpenAI) to handle tasks like listening to the audio, figuring out the best segment, translating, adding subtitles, and even drafting promotional text.

## Quick Start Guide

This guide assumes you have:
*   Downloaded the project files (including `video.py`).
*   Installed **Python 3.12.x** itself.
*   Installed **FFmpeg** system-wide and ensured it's accessible from your terminal (in your PATH). (See Step 1 below if unsure).
*   Obtained your OpenAI API Secret Key.
*   Downloaded the required Korean font file (e.g., `NanumGothicBold.ttf`).

**Steps:**

1.  **Install FFmpeg (System Dependency):**
    *   This is essential for video/audio processing and *must* be installed separately from the Python packages.
    *   Open your **system's main terminal or command prompt**.
    *   **On macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```
    *   **On Linux (Debian/Ubuntu):**
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```
    *   **On Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), unzip, and **add its `bin` folder to your system's PATH environment variable**. (Search online for "how to add to PATH windows").
    *   **Verify:** Close and reopen your terminal and type `ffmpeg -version`. If it runs without errors, it's likely set up correctly.

2.  **Prepare Project Folder:**
    *   Navigate to where you downloaded the project files (the folder containing `video.py`).
    *   Place the Korean font file (e.g., `NanumGothicBold.ttf`) inside this project folder.

3.  **Open Project in IDE/Terminal:**
    *   Open your IDE (like VS Code) and use "File -> Open Folder" to open the project folder.
    *   Open the integrated terminal within your IDE. Ensure it's pointing to your project folder directory.

4.  **Create Virtual Environment:**
    *   In the project terminal, run (using your Python 3.12):
        ```bash
        python3.12 -m venv app
        ```

5.  **Activate Virtual Environment:**
    *   Activate *in the same terminal*:
    *   **macOS / Linux:** `source app/bin/activate`
    *   **Windows (Cmd):** `app\Scripts\activate.bat`
    *   **Windows (PowerShell):** `.\app\Scripts\Activate.ps1`
    *   Confirm you see the `(app)` prefix in your prompt.

6.  **Install Required Libraries (Python Packages + yt-dlp):**
    *   With the virtual environment active, run this single `pip install` command in the terminal. It includes `yt-dlp` itself as a Python package (which provides the command-line tool when the venv is active) and specifies the exact versions needed for `moviepy` and `Pillow`:
        ```bash
        pip install yt-dlp openai "moviepy==1.0.3" "Pillow<10.0.0" pydub python-dotenv
        ```
    *   Wait for the installation to complete.

7.  **Create `.env` File (API Key):**
    *   In the project folder, create the file `.env`.
    *   Add your OpenAI API key:
        ```env
        OPENAI_API_KEY="sk-YOUR_REAL_OPENAI_API_KEY_HERE"
        ```
    *   Save the file.

8.  **Create `urls.txt` File (Video Links):**
    *   In the project folder, create the file `urls.txt`.
    *   Add your YouTube URLs, one per line. Use `#` for comments.
        ```txt
        https://www.youtube.com/watch?v=VIDEO_ID_1
        https://www.youtube.com/watch?v=VIDEO_ID_2
        ```
    *   Save the file.

9.  **Check Font File:**
    *   Ensure the Korean font file (e.g., `NanumGothicBold.ttf`) is in the main project folder.

10. **Run the Script:**
    *   Make sure the virtual environment is still active (`(app)`).
    *   In the terminal, run:
        ```bash
        python video.py
        ```
    *   The script will now execute, using the `yt-dlp` installed within the virtual environment.

11. **Check Output:**
    *   Monitor the logs in the terminal.
    *   Find the resulting `.mp4` and `.txt` files in the `output` folder.


## Features

*   **Reads URLs from File:** Easily process multiple videos by listing their YouTube URLs in a simple text file (`urls.txt`).
*   **Smart Download (1080p/720p):** Tries to download videos in Full HD (1080p) or HD (720p) resolution, avoiding unnecessarily large 4K files.
*   **AI Transcription:** Automatically converts the English speech in the audio into text.
*   **AI "Best Part" Finder:** Analyzes the English text to identify the most engaging or informative segment (roughly 60-90 seconds).
*   **Targeted Video Download:** Downloads *only* the video segment identified as the best part.
*   **Contextual Korean Translation:** Translates *only* the English text for the best segment into natural-sounding Korean for subtitles.
*   **Automatic Subtitling:** Adds the generated Korean subtitles onto the video clip.
*   **AI Social Post Generation:** Creates a ready-to-use social media post (in Korean, with hashtags) about the clip.
*   **Sequential Processing:** Handles multiple videos one after another.
*   **Error Handling:** Tries to continue processing the next video even if one fails.
*   **Automated Cleanup:** Removes temporary files after processing each video.

## How It Works (Simplified)

Here's a step-by-step look at what the assistant does for each video URL listed in your `urls.txt` file:

1.  **Listen First:** It downloads *only the audio* from the YouTube video.
2.  **Transcribe:** It uses AI (Whisper) to listen to the audio and write down everything said in English.
3.  **Analyze:** It uses AI (GPT-4o) to read the *English* text and decide, "Which 60-90 second part is the most interesting, impactful, or 'viral'?"
4.  **Download Video Segment:** Now that it knows the best part, it goes back and downloads *only that specific video segment* (aiming for 1080p or 720p resolution).
5.  **Translate (Smartly):** It takes the English text *only* for that best segment and uses AI (GPT-4o) to translate it into Korean, making sure it sounds natural.
6.  **Add Subtitles:** It adds the Korean text as subtitles onto the downloaded video segment.
7.  **Write Post:** It uses AI (GPT-4o) again to write a short promotional text in Korean (like for Twitter or Instagram) about the generated clip.
8.  **Clean Up:** It deletes the temporary audio and video section files it used, leaving you with the final clip and the post text before moving to the next URL in the file.

## Requirements

Before running the script, ensure you have the following:

1.  **Python:** You **must** have Python version **3.12.x** installed. (Check with `python --version` or `python3 --version` in your terminal). Get it from [python.org](https://www.python.org/downloads/).
2.  **pip:** Python's package installer (usually included with Python).
3.  **yt-dlp:** A tool for downloading videos. It needs to be installed and accessible system-wide (in your PATH). See [yt-dlp Installation](https://github.com/yt-dlp/yt-dlp#installation).
4.  **FFmpeg:** A multimedia tool used for processing video/audio. It also needs to be installed and accessible system-wide (in your PATH). See [FFmpeg Download](https://ffmpeg.org/download.html).
    *   *Tip:* On macOS: `brew install yt-dlp ffmpeg`. On Linux (Debian/Ubuntu): `sudo apt update && sudo apt install yt-dlp ffmpeg`.
5.  **Font File:** A Korean TrueType Font (`.ttf`) file for the subtitles. The script looks for `NanumGothicBold.ttf` in the *same directory* as the script by default. You might need to download it (e.g., from [Naver Hangeul Fonts](https://hangeul.naver.com/font)) and place it there. You can change the font name in the script's constants if you use a different file.
6.  **OpenAI API Key:** This script uses OpenAI's AI models, which requires an API key.
    *   Visit [platform.openai.com](https://platform.openai.com/).
    *   Sign up or log in.
    *   Find the API Keys section in your account settings.
    *   Create a **new secret key**. Copy it immediately and save it somewhere secure (you won't see it again).
    *   **Costs:** Using the OpenAI API is **not free**. Costs depend on how much you use it (transcription, analysis, translation, post generation). Check [OpenAI Pricing](https://openai.com/pricing) and ensure you have set up billing if necessary.

## Setup Instructions

1.  **Install Prerequisites:** Ensure Python 3.12, `yt-dlp`, and `ffmpeg` are installed and work from your terminal/command prompt.

2.  **Get the Code:** Download or clone the project files (especially `video.py`).

3.  **Create Input Files:**
    *   **`urls.txt`:** Create a plain text file named `urls.txt` in the *same directory* as `video.py`. Add the full YouTube video URLs you want to process, **one URL per line**. You can add comments by starting a line with `#`.
        ```txt
        # My list of videos
        https://www.youtube.com/watch?v=xxxxxxxxxxx
        https://www.youtube.com/watch?v=yyyyyyyyyyy
        # https://www.youtube.com/watch?v=zzzzzzzzzzz # Skip this one for now
        ```
    *   **`.env`:** Create another plain text file named `.env` (starting with a dot) in the *same directory*. Add your OpenAI API key like this:
        ```env
        OPENAI_API_KEY="YOUR_SECRET_API_KEY_HERE"
        ```
        Replace the placeholder with your actual secret key. **Keep this file private!**

4.  **Add Font File:** Place the Korean `.ttf` font file (e.g., `NanumGothicBold.ttf`) in the *same directory* as `video.py`.

5.  **Set Up Python Environment (Recommended):**
    *   Open your terminal/command prompt and navigate to the project directory.
    *   Create a virtual environment specifically for this project using your Python 3.12 installation:
        ```bash
        python3.12 -m venv venv
        ```
    *   Activate the environment:
        *   macOS/Linux: `source venv/bin/activate`
        *   Windows: `venv\Scripts\activate.bat` (or `.\venv\Scripts\Activate.ps1` in PowerShell)
        *   Your prompt should now show `(venv)`.

6.  **Install Python Packages:** Create a file named `requirements.txt` in the project directory with the following content:

    ```txt
    openai>=1.0.0
    moviepy==1.0.3
    Pillow<10.0.0
    pydub
    python-dotenv
    ```
    *Note the specific versions for `moviepy` and `Pillow` – these are important for compatibility!*
    Now, run this command in your activated environment:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Script

1.  Make sure your virtual environment is activated (`(venv)` should be visible in your terminal prompt).
2.  Ensure you are in the directory containing `video.py`, `urls.txt`, `.env`, and the font file.
3.  Run the script simply by typing:

    ```bash
    python video.py
    ```
    *(You don't need to provide URLs on the command line anymore – it reads them from `urls.txt`)*

4.  The script will start processing the URLs from `urls.txt` one by one. You'll see log messages showing the progress for each video. Processing time varies depending on video length, audio complexity, and API speeds.

## Output

For each successfully processed URL, the script will create files in the `output` folder:

*   `video_{youtube_id}_{timestamp}_final_subtitled.mp4`: The final video clip (usually 1080p or 720p) with Korean subtitles.
*   `video_{youtube_id}_{timestamp}_social_post.txt`: A text file with the suggested Korean social media post.

If the subtitling step fails for a video (but other steps succeeded), the `.mp4` file might not be created or might be empty. Check the logs for details. Temporary files are deleted after each video is processed.

## Configuration (Advanced)

You can tweak settings near the top of the `video.py` script:

*   `URL_INPUT_FILE`: Change the name of the file containing URLs.
*   `DELAY_BETWEEN_VIDEOS_SECONDS`: Add a pause between processing videos.
*   `SUBTITLE_FONT`, `SUBTITLE_FONTSIZE`, etc.: Adjust subtitle appearance.

## Troubleshooting

*   **`urls.txt` Not Found / Empty:** Make sure the file exists in the same directory as the script and contains valid YouTube URLs (one per line, no extra spaces).
*   **Dependencies Missing:** Double-check that `yt-dlp` and `ffmpeg` are installed and accessible system-wide (check your PATH). Ensure the font file exists where the script expects it.
*   **OpenAI API Errors (401, 403, 429):** Check your API key in `.env`, ensure it's active, and check your usage limits/billing on the OpenAI platform.
*   **`KeyError: 'video_fps'` or other MoviePy errors:** This might indicate the video section downloaded by `yt-dlp` was corrupted or incompatible. This can sometimes happen with specific video codecs or formats on YouTube, even with the exclusion filters. Check the `yt-dlp` command logs (especially `stderr`) for clues from `ffmpeg`. Sometimes trying the URL again later might work if it was a temporary issue. Ensure you installed the correct `Pillow<10.0.0` version.
*   **Download Errors ("Requested format not available"):** Although the script tries to be flexible (<=1080p), it's possible a specific video doesn't offer *any* compatible format matching the `-S` and `-f` criteria. You can manually test with `yt-dlp -F URL` to see available formats for that video.
*   **Translation/Subtitle Quality:** AI isn't perfect. Results vary. The script uses context but might still produce awkward phrasing sometimes. If translation fails entirely for a segment, the original English text is used as a fallback.

---

*This README provides instructions for setting up and running the script. Please use responsibly and be mindful of OpenAI API costs.*
