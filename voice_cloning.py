import subprocess

def run_f5_tts_infer(model, ref_audio, ref_text, gen_text, config_path=None, output_dir=None, output_file=None):
    command = [
        "f5-tts_infer-cli",
        "--model", model,
        "--ref_audio", ref_audio,
        "--ref_text", ref_text,
        "--gen_text", gen_text,
        "--speed", "0.7"
    ]
    if config_path:
        command += ["--config", config_path]
    if output_dir:
        command += ["--output_dir", output_dir]
    if output_file:
        command += ["--output_file", output_file]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Command output:", result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return None

if __name__ == "__main__":
    ref_audio = "sample_input/cropped_13.wav"
    ref_text = "それじゃあ、頑張りましょうね"
    gen_text = "Let's give it our best shot."
    config_path = "/path/to/f5-tts.yaml"  # Update this path if needed
    output_dir = "sample_output"          # Optional: specify output directory

    run_f5_tts_infer(
        model="F5TTS_v1_Base",
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        # config_path=config_path,
        # output_dir=output_dir
        output_file="output.wav"  # Optional: specify output file name
    )