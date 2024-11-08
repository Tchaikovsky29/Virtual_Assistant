from pyt2s.services import stream_elements
from pydub import AudioSegment
from pydub.playback import play
import io

# Use this to test out the different voices provided, just change the name: stream_elements.Voice.{name}.value
data = stream_elements.requestTTS('This is a longer voice recording to see if my voice is natural sounding.', stream_elements.Voice.Zhiyu.value)
# Load audio from the in-memory data
audio = AudioSegment.from_file(io.BytesIO(data), format="mp3")


