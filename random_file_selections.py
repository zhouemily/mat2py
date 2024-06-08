#!/usr/bin/env python3

import os
import random
import shutil

def create_test_directories(src_dir, num_files_list, dst_base_dir):
    """
    Randomly select files from the source directory and copy them into new directories.

    Parameters:
    src_dir (str): The source directory containing DICOM files.
    num_files_list (list): A list of integers representing the number of files to select for each test directory.
    dst_base_dir (str): The base destination directory where the new directories will be created.

    """
    # Get the list of all files in the source directory
    all_files = [f for f in os.listdir(src_dir) if f.endswith('.IMA')]

    if not all_files:
        print("No DICOM files found in the source directory.")
        return

    for num_files in num_files_list:
        # Randomly select the specified number of files
        selected_files = random.sample(all_files, num_files)

        # Create a new directory for this selection
        dst_dir = os.path.join(dst_base_dir, f'test_{num_files}_files')
        os.makedirs(dst_dir, exist_ok=True)

        # Copy the selected files to the new directory
        for file_name in selected_files:
            src_file = os.path.join(src_dir, file_name)
            dst_file = os.path.join(dst_dir, file_name)
            shutil.copy2(src_file, dst_file)

        print(f'Copied {num_files} files to {dst_dir}')

if __name__ == "__main__":
    source_directory = 'd005'
    destination_base_directory = 'test_dir_d005'
    files_to_select = [64, 128, 256]

    create_test_directories(source_directory, files_to_select, destination_base_directory)

