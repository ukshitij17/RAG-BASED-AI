import whisper
import json

model = whisper.load_model("large-v2")

result = model.transcribe("audios/test_audio.mp3",
                            language="hi",
                            task="translate",
                            word_timestamps=False)
print(result["text"])

cleared_chunks = []

for seg in result['segments']:
    cleared_chunks.append({
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"]
    })

# print(cleared_chunks, type(cleared_chunks))

with open("output.json", "w") as f:
    json.dump(cleared_chunks, f, indent=4)