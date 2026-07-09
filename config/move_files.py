from pathlib import Path
import subprocess

SOURCE_DIR = Path("/Volumes/Projects/ToneDecodingArousal")


sessionsTotal = []
for sessionPath in SOURCE_DIR.iterdir():
    if sessionPath.is_dir() and "mouse" in sessionPath.name:
        sessionsTotal.append(sessionPath.name)

## sort sessions by name
sessionsTotal = sorted(sessionsTotal, key=lambda x: x.split("-")[-1], reverse=True)

splitsPerExp = 4
numExperiments = 1  ## define the number of experiments to move
numSessionsToMove = splitsPerExp * numExperiments

sessionsToMove = sessionsTotal[:numSessionsToMove]
print(f"Moving {len(sessionsToMove)} sessions:")
print(f"Sessions: {sessionsToMove}")

for sessionName in sessionsToMove:
    sessionPath = Path(SOURCE_DIR, sessionName)
    ephysDirPath = [d for d in sessionPath.iterdir() if d.is_dir()][
        0
    ]  ## there is only 1 sub-directory

    ephysDirName = ephysDirPath.name

    # Build exclude list as separate arguments
    exclude_patterns = [
        "*.mp4",  # All MP4 files
        "*.pickle",  # All pickle files
        f"{ephysDirName}/*.continuous",  # Continuous files in ephys dir
        f"{ephysDirName}/.*",  # Hidden files in ephys dir
        f"{ephysDirName}/*.spikes",  # Spike files in ephys dir
        f"{ephysDirName}/*.npy",  # NumPy files in ephys dir
    ]

    # Build rsync command with list comprehension
    exclude_args = " ".join([f"--exclude='{pattern}'" for pattern in exclude_patterns])

    cmd_run = (
        f"rsync -avP "
        f"{exclude_args} "
        f"{sessionPath} "
        f"praves@login.talapas.uoregon.edu:/gpfs/home/praves/wehrlab/data/tone_decode_data/{sessionName}"
    )

    print(f"\n{'=' * 80}")
    print(f"Syncing: {sessionName}")
    print(f"Ephys dir: {ephysDirName}")
    print(f"Command: {cmd_run}")
    print(f"{'=' * 80}")

    subprocess.run(cmd_run, shell=True)

print("\nDone!")
