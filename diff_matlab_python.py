
def clean_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # Remove spaces and newlines
    cleaned_content = ''.join(content.split())
    return cleaned_content

# File paths
matlab_file = 'matlab_pixels.txt'
python_file = 'python_pixels.txt'

# Clean contents
matlab_content = clean_content(matlab_file)
python_content = clean_content(python_file)

# Compare contents
if matlab_content == python_content:
    print("The files are identical (ignoring spaces and newlines).")
else:
    print("The files differ.")

# Optionally print differences
import difflib
diff = difflib.unified_diff(matlab_content, python_content)
print('\n'.join(diff))

