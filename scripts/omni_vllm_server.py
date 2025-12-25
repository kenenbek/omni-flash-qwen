"""
Example: Using Qwen2_5OmniTalkerForConditionalGeneration from vllm-omni for TTS

vllm-omni is an extension of vLLM that supports multimodal generation including audio.
The Qwen2_5OmniTalkerForConditionalGeneration model is specifically designed for
text-to-speech synthesis using the Qwen2.5-Omni architecture.
"""

import soundfile as sf
import torch
from typing import Optional

# Option 1: Using vllm-omni with offline inference
def tts_with_vllm_omni_offline(
    text: str,
    model_path: str = "Qwen/Qwen2.5-Omni-7B",
    output_file: str = "output.wav",
    speaker: str = "Ethan"
):
    """
    Generate audio from text using vllm-omni offline inference.

    Args:
        text: The text to convert to speech
        model_path: Path to the Qwen2.5-Omni model
        output_file: Output WAV file path
        speaker: Voice identity (e.g., "Ethan", "Chelsie")
    """
    from vllm import LLM, SamplingParams

    # Initialize the LLM with Qwen2.5-Omni model
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )

    # Create the conversation/prompt for TTS
    prompt = f"Read the following text aloud: {text}"

    # Sampling parameters for generation
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )

    # Generate
    outputs = llm.generate(prompt, sampling_params)

    # Extract and save audio if available
    for output in outputs:
        if hasattr(output, 'audio') and output.audio is not None:
            sf.write(output_file, output.audio, samplerate=24000)
            print(f"Audio saved to {output_file}")

    return outputs


# Option 2: Using the Talker model directly from transformers/vllm-omni
def tts_with_talker_model(
    text: str,
    model_path: str = "Qwen/Qwen2.5-Omni-7B",
    output_file: str = "output.wav",
    speaker: str = "Ethan"
):
    """
    Generate audio using Qwen2_5OmniTalkerForConditionalGeneration directly.

    The Talker model is the audio synthesis component of Qwen2.5-Omni.
    """
    try:
        # Try importing from vllm-omni first
        from vllm_omni.model import Qwen2_5OmniTalkerForConditionalGeneration
    except ImportError:
        # Fallback to transformers
        from transformers import Qwen2_5OmniTalkerForConditionalGeneration

    from transformers import Qwen2_5OmniProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Talker model
    model = Qwen2_5OmniTalkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    # Prepare conversation
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that can speak."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Read aloud: {text}"}]
        }
    ]

    # Process input
    text_input = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = processor(text=text_input, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    # Generate audio
    text_ids, audio = model.generate(
        **inputs,
        speaker=speaker,
        return_dict_in_generate=True,
    )

    # Decode text output
    generated_text = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(f"Generated text: {generated_text}")

    # Save audio
    if audio is not None:
        sf.write(
            output_file,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print(f"Audio saved to {output_file}")

    return generated_text, audio


# Option 3: FastAPI Server for TTS using vllm-omni
def create_tts_server(model_path: str = "Qwen/Qwen2.5-Omni-7B"):
    """
    Create a FastAPI server for TTS using vllm-omni.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    import uuid

    app = FastAPI(title="Qwen2.5-Omni TTS Server")

    # Initialize model once at startup
    model = None
    processor = None

    class TTSRequest(BaseModel):
        text: str
        speaker: str = "Ethan"

    @app.on_event("startup")
    async def load_model():
        nonlocal model, processor
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        print(f"Model loaded: {model_path}")

    @app.post("/tts")
    async def text_to_speech(request: TTSRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Prepare input
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Read aloud: {request.text}"}]
            }
        ]

        text_input = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = processor(text=text_input, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        # Generate
        text_ids, audio = model.generate(
            **inputs,
            speaker=request.speaker,
        )

        if audio is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # Save to temp file
        output_file = f"/tmp/tts_{uuid.uuid4()}.wav"
        sf.write(
            output_file,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )

        return FileResponse(output_file, media_type="audio/wav")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": model is not None}

    return app


# Main entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2.5-Omni TTS")
    parser.add_argument("--mode", choices=["tts", "server"], default="tts",
                        help="Run mode: 'tts' for single generation, 'server' for API")
    parser.add_argument("--text", type=str, default="Hello, this is a text to speech demo.",
                        help="Text to convert to speech")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Omni-7B",
                        help="Model path or name")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output WAV file path")
    parser.add_argument("--speaker", type=str, default="Ethan",
                        help="Speaker voice identity")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")

    args = parser.parse_args()

    if args.mode == "tts":
        # Single TTS generation
        print(f"Generating speech for: {args.text}")
        tts_with_talker_model(
            text=args.text,
            model_path=args.model,
            output_file=args.output,
            speaker=args.speaker
        )
    else:
        # Run server
        import uvicorn
        app = create_tts_server(model_path=args.model)
        uvicorn.run(app, host=args.host, port=args.port)

