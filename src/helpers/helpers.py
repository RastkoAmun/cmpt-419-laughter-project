import os
import librosa
import torch
import numpy as np

# Implemented by Matt 
def get_full_path(base_path, audio_id):
	wav_file = os.path.join(base_path, f"{audio_id}.wav")
	mp3_file = os.path.join(base_path, f"{audio_id}.mp3")
	ogg_file = os.path.join(base_path, f"{audio_id}.ogg")

	if os.path.isfile(wav_file):
		file_path = wav_file
	elif os.path.isfile(mp3_file):
		file_path = mp3_file
	elif os.path.isfile(ogg_file):
		file_path = ogg_file
	else:
		print(f"No audio found for ID {audio_id}")

	return file_path


# Implemented by Matt (modified by Rastko)
def extract_mfcc(file_path, n_mfcc=13, max_len=130):
	y, sr = librosa.load(file_path, sr=16000)
	mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

	# Pad or truncate to consistent size
	if mfcc.shape[1] < max_len:
		pad_width = max_len - mfcc.shape[1]
		mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
	else:
		mfcc = mfcc[:, :max_len]

	return torch.tensor(mfcc).unsqueeze(0)  # shape: [1, n_mfcc, max_len]