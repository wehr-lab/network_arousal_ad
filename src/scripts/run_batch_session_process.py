#!/usr/bin/env python3
"""
Batch processing script for arousal network analysis across multiple sessions.
"""

import sys
from pathlib import Path
import warnings
import subprocess

warnings.filterwarnings("ignore")

# Setup paths
repoDir = Path(__file__).resolve().parent.parent.parent
# print(repoDir)
configDir = repoDir / "config"
sys.path.append(str(configDir))

from settings import DATA_PATH, CODE_PATH

sys.path.append(str(CODE_PATH["2-photon"]))


class BatchSessionProcessor:
    """
    Container class to store and run batch processing for multiple tone arousal sessions."""

    def __init__(self, session_list):
        self.session_list = session_list

    def run_session_temporal_alignment(self):
        try:
            for session_path in self.session_list:
                print(f"Processing session: {session_path}")
                if "neural_pupil_network_aligned_temporally.pkl" not in [
                    file.name for file in session_path.iterdir()
                ]:
                    print(f"Running temporal alignment for session: {session_path}")
                    cmd = [
                        "python",
                        str(
                            Path(
                                CODE_PATH["2-photon"],
                                "scripts",
                                "stimuli_phys_network_align_for_batch.py",
                            )
                        ),
                        "--session_path",
                        str(
                            session_path
                        ),  ## passing the absolute path to the session because the script requires it
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=str(Path(CODE_PATH["2-photon"], "scripts")),
                    )
                    if result.stdout:
                        print(result.stdout)
                    if result.returncode != 0:
                        print(
                            f"Error processing session {session_path}: {result.stderr}"
                        )
                else:
                    print(
                        f"Temporal alignment data already exists for session: {session_path}"
                    )
        except:
            print(
                "An error occurred during temporal alignment processing. Skipping to next session."
            )

    def run_session_process_df(self):
        for session_path in self.session_list:
            print(f"Processing session: {session_path}")
            if "sessionProcessDF.pkl" not in [
                file.name for file in session_path.iterdir()
            ]:
                print(f"Running session processing for session: {session_path}")
                cmd = [
                    "python",
                    str(
                        Path(
                            CODE_PATH["2-photon"],
                            "scripts",
                            "stimuli_phys_align_per_event_for_batch.py",
                        )
                    ),
                    "--session_path",
                    str(
                        session_path
                    ),  ## passing the absolute path to the session because the script requires it
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(Path(CODE_PATH["2-photon"], "scripts")),
                )
                if result.stdout:
                    print(result.stdout)
                if result.returncode != 0:
                    print(f"Error processing session {session_path}: {result.stderr}")
            else:
                print(
                    f"Temporal alignment data already exists for session: {session_path}"
                )

    def run_processor(self):
        self.run_session_temporal_alignment()
        self.run_session_process_df()


if __name__ == "__main__":
    DATA_PATH_AROUSAL = Path(
        DATA_PATH["toneDecode"]
    )  ## passing it hardcoded because this path should already be defined in the settings script

    session_list = []
    for mouseDir in DATA_PATH_AROUSAL.iterdir():
        if mouseDir.is_dir() and mouseDir.name.startswith("2022"):
            session_list.append(mouseDir)
    print(f"Found {len(session_list)} sessions to process.")

    batch_processor = BatchSessionProcessor(session_list[220:])
    batch_processor.run_processor()
