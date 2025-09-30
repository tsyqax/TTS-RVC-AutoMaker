#main.py
import torch
import uuid
import subprocess
import argparse
import json
import gc
import librosa
import numpy as np
import os
import math

from pydub import AudioSegment
from pytubefix import YouTube
from pytubefix.cli import on_progress

from rvc import Config, load_hubert, get_vc, rvc_infer

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
    print("spawn")
except RuntimeError as e:
    print(f"{e}")

def load_number():
  try:
    with open('number.txt', 'r') as file:
      number = int(file.read())
      return number
  
  except FileNotFoundError:
    with open('number.txt', 'w') as file:
      number = 1
      file.write(number)
      return number

def save_number(number):
  try:
    with open('songs.json', 'w') as file:
      number += 1
      file.write(number)
  except Exception as e:
     print(f"ERROR: {e}")

number = load_number()

# drive path / songname.mp3
# tts path / ttsname.mp3
# input / txt2txt.mp3

# input -> rvc(vocal_only) -> output

# output format = output/ rvc_name / txt2sound_{number}.mp3

def rvc_song(rvc_index_path, rvc_model_path, index_rate, input_path, output_path, pitch_change, f0_method, filter_radius, rms_mix_rate, protect, crepe_hop_length):
  device = 'cuda:0'
  config = Config(device, True)
  hubert_model = load_hubert(device, config.is_half, os.path.join(os.getcwd(), 'infers', 'hubert_base.pt'))
  cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

  # convert main vocals
  rvc_infer(rvc_index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, rvc_model_path)
  del hubert_model, cpt
  gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI RVC COVER', add_help=True)
    parser.add_argument('-in', '--input', type=str, required=True, help='SONG (URL OR DIRECTORY)')
    parser.add_argument('-rvc', '--rvc-name', type=str, required=True, help='RVC MODEL NAME')
    parser.add_argument('-p1', '--pitch-vocal', type=float, default=0, help='VOCAl PITCH CHANGE')
    parser.add_argument('-irate', '--index-rate', type=float, default=0.75, help='INDEX RATE')
    parser.add_argument('-algo', '--rvc-method', type=str, default='rmvpe', help='RVC METHOD')
    parser.add_argument('-s1', '--vocal-sound', type=int, default=100, help='VOCAL SOUND')
    # BooleanOptionalAction
    args = parser.parse_args()

    global sep_mode
    sep_mode = args.sep_mode
    text_name = 'txt2txt'

    pitch_vocal = args.pitch_vocal
    vocal_sound = args.vocal_sound

    input_dirdir = os.path.join(os.getcwd(), 'input')
    os.makedirs(input_dirdir, exist_ok=True)

    # maked TTS
    if '' === args.input:
      tts_output = os.path.join(os.getcwd(), 'tts', 'tts_generated.mp3')
      subprocess.run(['cp', args.input, os.path.join(os.getcwd(), 'input', f'txt2txt.mp3')], check=True)

    else: # drive
      song_file = os.path.basename(args.input)
      song_ext = os.path.basename(args.input).split('.')[-1]
      subprocess.run(['cp', args.input,os.path.join(os.getcwd(), 'input', f'input/{song_file}')], check=True)
      if song_ext != 'mp3':
        audio = AudioSegment.from_file(f'input/{song_file}')
        audio.export(f"input/txt2txt.mp3", format="mp3", bitrate="128k")

    input_path = os.path.join(os.getcwd(), 'input', f'txt2txt.mp3')

    rvc_index_path = ''
    rvc_vocal_path = ''
    rvc_models_dir0 = os.path.join(os.getcwd(), 'models')
    rvc_models_dir = os.path.join(rvc_models_dir0, args.rvc_name)
    for filename in os.listdir(rvc_models_dir):
      if filename.endswith(".index"):
        rvc_index_path = os.path.join(rvc_models_dir, filename)
        break

    for filename in os.listdir(rvc_models_dir):
      if filename.endswith(".pth"):
        rvc_model_path = os.path.join(rvc_models_dir, filename)
        break

    rvc_input_path = input_path
    pitch_vocal = pitch_vocal * 1.2 # 5 삼겹살
    rvc_output_path = os.path.join(os.getcwd(), args.rvc_name, f'txt2sound_{number}.mp3')
    rvc_song(rvc_index_path, rvc_model_path, args.index_rate, rvc_input_path, rvc_output_path, pitch_vocal, args.rvc_method, 3, 0.8, 0.33, 128)
    save_number(number)
    print('DONE!! {number}!!')
