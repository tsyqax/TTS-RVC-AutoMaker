import os

try:
  import edge_tts 
  text = "안녕하세요. Edge TTS를 사용하여 텍스트를 음성으로 변환해 보겠습니다."
  voice="ko-KR-SunHiNeural"

  tts = edge_tts.Communicate(text, voice)

  sound_file = os.path.join(os.getcwd(), 'tts', 'tts_generated.mp3')
  tts.save_sync(sound_file)
  
except:
  import gtts
