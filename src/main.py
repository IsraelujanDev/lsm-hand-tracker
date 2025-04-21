import os

# Define your base folder (you can change this)
base_folder = "lsm-hands"

# Define the subfolders to create
folders = [
    "data/raw",
    "data/images",
    "data/annotations",
    "data/metadata",
    "scripts",
    "notebooks",
    "models",
    "app/frontend",
    "app/backend",
    "src"
]

def create_project_structure(base, folder_list):
    print(f"📁 Creating project folder: {base}")
    os.makedirs(base, exist_ok=True)

    for folder in folder_list:
        full_path = os.path.join(base, folder)
        os.makedirs(full_path, exist_ok=True)
        print(f"  ✅ Created: {full_path}")

    # Create base files
    open(os.path.join(base, "README.md"), 'w').close()
    open(os.path.join(base, "uv.toml"), 'w').close()
    open(os.path.join(base, ".gitignore"), 'w').close()
    print("📝 Base files created: README.md, uv.toml, .gitignore")

# Run it
if __name__ == "__main__":
    create_project_structure(base_folder, folders)
