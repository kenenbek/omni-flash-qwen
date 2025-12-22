import soundfile as sf
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# 1. Load Model (Same as your code)
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)


# 2. Define a TTS Function
def text_to_speech(text_to_read, filename="tts_output.wav", speaker="Ethan"):
    """
    Forces the Qwen3 Omni model to act as a TTS engine.
    """

    # We instruct the model to "Read this aloud" to prevent it from
    # answering the text as a question.
    prompt_text = f"Read the following text aloud exactly as it is written: {text_to_read}"

    # Construct conversation with ONLY text (no images/input audio)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text}
            ],
        },
    ]

    # Process inputs (use_audio_in_video=False since we have no video)
    text_input = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # process_mm_info handles the empty lists for audio/images correctly
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    inputs = processor(
        text=text_input,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )

    inputs = inputs.to(model.device).to(model.dtype)

    # Inference
    # The 'speaker' argument controls the voice identity
    print(f"Generating audio for: '{text_to_read}'...")
    text_ids, audio_output = model.generate(
        **inputs,
        speaker=speaker,
        thinker_return_dict_in_generate=True,
        use_audio_in_video=False
    )

    # Save Audio
    if audio_output is not None:
        sf.write(
            filename,
            audio_output.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print(f"Saved to {filename}")
    else:
        print("Error: No audio generated.")


# --- Usage Example ---
if __name__ == "__main__":
    my_text = "This is a demonstration of using the Qwen Omni model as a text to speech engine."

    # You can change speaker to "Ethan" or others defined in the model config
    text_to_speech(my_text, filename="my_speech.wav", speaker="Ethan")