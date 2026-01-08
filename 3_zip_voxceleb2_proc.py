#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path
import math
import pdb


def get_speaker_ids(dataset_root):
    """
    Read and sort speaker IDs (folder names).
    """
    speakers = [
        d.name for d in dataset_root.iterdir()
        if d.is_dir()
    ]
    speakers.sort()
    return speakers


def split_into_groups(items, num_groups):
    """
    Split a list into num_groups with roughly equal size.
    """
    group_size = math.ceil(len(items) / num_groups)
    return [
        items[i * group_size:(i + 1) * group_size]
        for i in range(num_groups)
    ]


def zip_group(dataset_root, speaker_ids, output_zip):
    """
    Call Ubuntu zip command for a group of speakers.
    """
    cmd = ["zip", "-r", output_zip] + speaker_ids
    print(f"\n[INFO] Creating {output_zip}")
    subprocess.run(
        cmd,
        cwd=dataset_root,
        check=True
    )


def main(args):
    dataset_root = (Path(args.dataset_root) / "VoxCeleb2" / "train" / "wav").resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    speakers = get_speaker_ids(dataset_root)

    print(f"[INFO] Found {len(speakers)} speakers")

    groups = split_into_groups(speakers, args.num_groups)

    for i, group in enumerate(groups):
        print(f"Group {i+1}: {group[0]} - {group[-1]}")
        # group = group[:8]
        zip_name = output_dir / f"voxceleb2_proc_segments_{i+1}.zip"
        zip_group(dataset_root, group, str(zip_name))

    print("\n[INFO] All zip files created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split VoxCeleb2 speakers into multiple zip files"
    )
    parser.add_argument(
        "--dataset_root",
        default="./voxceleb2_proc_segments/",
        help="Path to VoxCeleb2 root directory"
    )
    parser.add_argument(
        "--output_dir",
        default="./voxceleb2_proc_zips",
        help="Directory to save zip files"
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=5,
        help="Number of speaker groups"
    )

    args = parser.parse_args()
    main(args)
