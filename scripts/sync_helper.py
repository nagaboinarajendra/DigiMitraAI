import subprocess
import sys
from datetime import datetime
import os

def run_command(command):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error message: {e.stderr}")
        return None

def get_changed_files():
    """Get list of changed files"""
    return run_command("git status --porcelain")

def sync_changes():
    """Sync changes with GitHub repository"""
    # Check if there are changes
    changes = get_changed_files()
    if not changes:
        print("No changes to sync!")
        return True

    # Show changed files
    print("\nChanged files:")
    print(changes)
    
    # Get commit message
    default_message = f"Update {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    message = input(f"\nEnter commit message (press Enter for '{default_message}'): ")
    message = message.strip() or default_message

    # Add changes
    if not run_command("git add ."):
        return False

    # Commit changes
    if not run_command(f'git commit -m "{message}"'):
        return False

    # Pull latest changes
    if not run_command("git pull origin main"):
        return False

    # Push changes
    if not run_command("git push origin main"):
        return False

    print("\n✅ Changes successfully synced to GitHub!")
    return True

def main():
    """Main function"""
    print("🔄 Starting sync process...")
    
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    if not sync_changes():
        print("\n❌ Sync failed! Please resolve any issues and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()