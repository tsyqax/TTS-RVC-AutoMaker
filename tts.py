import os
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='AI RVC COVER', add_help=True)
  parser.add_argument('-txt', '--text', type=str, required=True, help='TEXT TO SAY')
  parser.add_argument('-cat', '--castle', type=str, default='Male', help='What is your castle')
  parser.add_argument('-lang', '--language', type=str, default='en', help='TTS LANGUAGE')
  # BooleanOptionalAction
  args = parser.parse_args()
  text = args.text
  castle = args.castle
  lang = args.language
  try:
    import edge_tts
    if lang == 'ja':
      if castle == 'Female':
        voice = 'ja-JP-NanamiNeural'
      else:
        voice = 'ja-JP-KeitaNeural'
    elif lang == 'ko':
      if castle == 'Female':
        voice = "ko-KR-SunHiNeural"
      else:
        voice = 'ko-KR-HyunsuMultilingual'
    else:      
      if castle == 'Female':
        voice = 'en-US-EmmaMultilingual'
      else:
        voice = 'en-US-AndrewMultilingual'
  
    tts = edge_tts.Communicate(text, voice)
  
    sound_file = 'tts_generated.mp3'
    tts.save_sync(sound_file)
    print('TTS TTS!! (EdgeTTS!)')
      
  except:
    from gtts import gTTS
    tts2 = gTTS(text, lang=lang) 
    tts2.save('tts_generated.mp3')
    print('TTS TTS!! (gtts!)')
