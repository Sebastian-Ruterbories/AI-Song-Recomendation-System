import click
import os
from pathlib import Path
import shutil
from kick_detector import KickHardnessAnalyzer

@click.command()
@click.argument('input_paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory (default: same as input)')
@click.option('--copy', '-c', is_flag=True, help='Copy files instead of renaming')
@click.option('--model', '-m', default='models/kick_hardness_model.pkl', 
              help='Path to trained model')
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
def main(input_paths, output_dir, copy, model, recursive):
    """
    Analyze audio files and rename/copy them with kick hardness scores.
    
    INPUT_PATHS: One or more audio files or directories to process
    """
    # Initialize analyzer
    analyzer = KickHardnessAnalyzer(model)
    
    # Collect all audio files to process
    audio_files = []
    supported_formats = {'.wav', '.flac', '.mp3', '.m4a', '.aac'}
    
    for input_path in input_paths:
        path = Path(input_path)
        
        if path.is_file() and path.suffix.lower() in supported_formats:
            audio_files.append(path)
        elif path.is_dir():
            if recursive:
                # Find all audio files recursively
                for ext in supported_formats:
                    audio_files.extend(path.rglob(f'*{ext}'))
            else:
                # Find audio files in directory (non-recursive)
                for ext in supported_formats:
                    audio_files.extend(path.glob(f'*{ext}'))
    
    if not audio_files:
        click.echo("No supported audio files found!")
        return
    
    click.echo(f"Found {len(audio_files)} audio files to process...")
    
    # Process each file
    with click.progressbar(audio_files, label='Analyzing files') as files:
        for audio_file in files:
            process_file(analyzer, audio_file, output_dir, copy)
    
    click.echo("Processing complete!")

def process_file(analyzer, audio_file, output_dir, copy_mode):
    """
    Process a single audio file: analyze and rename/copy with score
    
    Args:
        analyzer: KickHardnessAnalyzer instance
        audio_file: Path to audio file
        output_dir: Output directory (None for same directory)
        copy_mode: True to copy, False to rename
    """
    try:
        # Analyze the file
        score = analyzer.analyze_file(audio_file)
        
        # Create new filename with score
        score_str = f"{score:.2f}"
        new_name = f"{score_str}__{audio_file.name}"
        
        # Determine output path
        if output_dir:
            output_path = Path(output_dir) / new_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = audio_file.parent / new_name
        
        # Copy or rename the file
        if copy_mode:
            shutil.copy2(audio_file, output_path)
            action = "Copied"
        else:
            audio_file.rename(output_path)
            action = "Renamed"
        
        click.echo(f"{action}: {audio_file.name} -> {new_name} (score: {score:.2f})")
        
    except Exception as e:
        click.echo(f"Error processing {audio_file.name}: {str(e)}")

if __name__ == '__main__':
    main()

