from pathlib import Path

# create foreground folder
fg_folder = Path("~/mixtures/foreground").expanduser()
fg_folder.mkdir(parents=True, exist_ok=True)

# create background folder (even if we dont use it)
bg_folder = Path("~/mixtures/background").expanduser()
bg_folder.mkdir(parents=True, exist_ok=True)

# path to orignal audio files
audio_path = Path("~/audio_source_separation/data/audio").expanduser()

for item in audio_path.iterdir():
    print(item.name)
    break
