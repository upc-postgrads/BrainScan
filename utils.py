import os

def ensure_dir_from_file_path(file_path):
    directory = os.path.dirname(file_path)
    ensure_dir(directory)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def percentage(part, whole):
  return 100 * float(part)/float(whole)
