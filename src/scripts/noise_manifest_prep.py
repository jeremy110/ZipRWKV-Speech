import argparse
from pathlib import Path
from lhotse import CutSet, RecordingSet, Recording

def main():
    # 1. Initialize the Argument Parser
    parser = argparse.ArgumentParser(
        description="Convert an audio directory into Lhotse CutSet Manifests."
    )
    
    # 2. Define Command Line Arguments
    parser.add_argument(
        "--audio_dir", 
        type=str, 
        required=True, 
        help="Root directory containing the source .wav files."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory where the generated manifest will be saved."
    )
    parser.add_argument(
        "--prefix", 
        type=str, 
        default="noise", 
        help="Prefix for the output filename (default: noise)."
    )

    args = parser.parse_args()

    # 3. Path Conversions and Validation
    audio_path = Path(args.audio_dir)
    output_path = Path(args.output_dir)

    if not audio_path.exists():
        print(f"Error: Input directory '{audio_path}' does not exist.")
        return

    # Ensure the output directory exists (create if necessary)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning directory: {audio_path} for .wav files...")

    # ---------------------------------------------------------
    # Step 1: Create RecordingSet
    # Recursively find all .wav files and wrap them in Recording objects
    # ---------------------------------------------------------
    recordings = RecordingSet.from_recordings(
        Recording.from_file(file) for file in audio_path.rglob("*.wav")
    )

    # ---------------------------------------------------------
    # Step 2: Convert Recordings to Cuts
    # Cuts are the basic unit of manipulation in Lhotse
    # ---------------------------------------------------------
    cuts = CutSet.from_manifests(recordings=recordings)

    # ---------------------------------------------------------
    # Step 3: Export to Compressed JSONL
    # Saves the metadata to the specified destination
    # ---------------------------------------------------------
    final_output = output_path / f"{args.prefix}_cuts.jsonl.gz"
    cuts.to_file(final_output)
    
    print(f"✅ Success! Manifest saved to: {final_output}")
    print(f"📊 Total cuts processed: {len(cuts)}")

if __name__ == "__main__":
    main()