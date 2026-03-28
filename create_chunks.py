import whisper
import json
import os

model = whisper.load_model("large-v2")
audios = os.listdir("audios")

for audio in audios:
    if("_" in audio):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]

    
        result = model.transcribe(f"audios/{audio}",
        # result = model.transcribe(f"audios/sample.mp3",
                                language="hi",
                                task="translate",
                                word_timestamps=False)

        cleared_chunks = []

        for seg in result['segments']:
            cleared_chunks.append({
                "number": number,
                "title": title,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            })

        chunks_with_metadata = {"chunks" : cleared_chunks, "text" : result["text"]}

        # print(cleared_chunks, type(cleared_chunks))

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f, indent=4)