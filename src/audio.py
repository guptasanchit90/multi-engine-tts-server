import subprocess

try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonzero_sound
except ImportError:
    AudioSegment = None


def trim_silence(wav_path: str) -> None:
    if AudioSegment is None:
        return
    try:
        audio = AudioSegment.from_file(wav_path, format="wav")
        nonsilent = detect_nonzero_sound(audio, min_silence_len=100, silence_thresh=-50)
        if nonsilent:
            trimmed = audio[nonsilent[0][0]:nonsilent[-1][1]]
            trimmed.export(wav_path, format="wav")
    except Exception:
        pass


def wav_to_mp3(wav_path: str, mp3_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", wav_path,
        "-codec:a", "libmp3lame", "-qscale:a", "2",
        mp3_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def wav_to_pcm(wav_path: str, pcm_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", wav_path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        pcm_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
