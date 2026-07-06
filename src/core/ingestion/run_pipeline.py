# /app/run_pipeline.py
import subprocess
import sys


def run_script(command: list[str]) -> bool:
    print(f"\n Running: {' '.join(command)}")
    print("-" * 60)

    # run the command and stream output directly to the terminal
    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        return False

    print("Step completed successfully.\n")
    return True


def main():
    # Define your pipeline execution steps in order
    pipeline_steps = [
        ["python", "/app/src/core/ingestion/extraction/test_text_extractor.py"],
        [
            "python",
            "-m",
            "src.core.ingestion.post_processing.test_heading_correction",
        ],
        ["python", "/app/src/core/ingestion/post_processing/section_assembler.py"],
        ["python", "/app/src/core/ingestion/chunking/enrich_sections.py"],
        ["python", "/app/src/core/ingestion/analyze_section_tokens.py"],
        ["python", "/app/src/core/ingestion/chunking/chunker.py"],
    ]

    print("============================================================")
    print("Starting EMV Data Ingestion Pipeline")
    print("============================================================")

    for step in pipeline_steps:
        success = run_script(step)
        if not success:
            print("🛑 Pipeline halted due to an error in a critical step.")
            sys.exit(1)

    print("============================================================")
    print("All pipeline steps executed successfully!")
    print("============================================================")


if __name__ == "__main__":
    main()