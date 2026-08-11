import os
import sys
import subprocess
import argparse
from typing import List, Tuple

FORBIDDEN_EXTENSIONS = {
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".onnx"
}

FORBIDDEN_PATHS = {
    "models/", "checkpoints/", ".env", ".env.example", ".env.local"
}

MAX_FILE_SIZE_MB = 25.0


def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout + res.stderr


def scan_for_forbidden_files() -> List[str]:
    violations = []
    
    code, output = run_cmd(["git", "status", "--porcelain"])
    if code != 0:
        print(f"[ERROR] git status failed: {output}")
        sys.exit(1)

    lines = [line.strip() for line in output.split("\n") if line.strip()]
    
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        status_flag, rel_path = parts[0], parts[1]
        
        # Check forbidden extensions
        _, ext = os.path.splitext(rel_path)
        if ext.lower() in FORBIDDEN_EXTENSIONS:
            violations.append(f"Forbidden weight extension '{ext}': {rel_path}")

        # Check forbidden directories/files
        for forbidden in FORBIDDEN_PATHS:
            norm_rel = rel_path.replace("\\", "/")
            if norm_rel.startswith(forbidden) or norm_rel.startswith(f"./{forbidden}"):
                violations.append(f"Forbidden directory/secret path '{forbidden}': {rel_path}")

        # Check file size if file exists
        if os.path.isfile(rel_path):
            try:
                size_mb = os.path.getsize(rel_path) / (1024 * 1024)
                if size_mb > MAX_FILE_SIZE_MB:
                    violations.append(f"File size too large ({size_mb:.2f} MB > {MAX_FILE_SIZE_MB} MB): {rel_path}")
            except Exception:
                pass

    return violations


def main():
    parser = argparse.ArgumentParser(description="Safely stage, commit, and push evaluation outputs & code to GitHub.")
    parser.add_argument("--message", type=str, required=True, help="Git commit message.")
    parser.add_argument("--dry_run", action="store_true", help="Perform safety scan without committing or pushing.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    print("--- Starting GitHub Pre-Push Safety Audit ---")
    violations = scan_for_forbidden_files()

    if violations:
        print("\n" + "!" * 60)
        print("          GIT PUSH BLOCKED BY SAFETY AUDIT")
        print("!" * 60)
        for v in violations:
            print(f"  [X] {v}")
        print("!" * 60)
        print("Refusing to stage or push. Remove weight binaries or secrets before committing.")
        sys.exit(1)

    print("[PASSED] Zero weight binaries, checkpoints, secrets, or oversized files detected.")

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] Safety scan clean. No git changes committed.")
        return

    # Stage files
    print("\nStaging code, configs, logs, and outputs/...")
    run_cmd(["git", "add", "main.py", "config.py", "vlm_annotation/", "scripts/", "training/", "outputs/", "logs/", "tests/", "README.md", ".gitignore", "requirements.txt", "pyproject.toml", "openspec/"])

    code, git_status_out = run_cmd(["git", "status", "--short"])
    print("Files ready to be committed:")
    print(git_status_out)

    if not args.yes:
        confirm = input("\nProceed with commit and push? (y/N): ").strip().lower()
        if confirm != "y":
            print("Operation aborted by user.")
            return

    # Commit
    print(f"\nCommitting: '{args.message}'...")
    code, commit_out = run_cmd(["git", "commit", "-m", args.message])
    print(commit_out)

    # Push
    print("\nPushing to remote repository...")
    code, push_out = run_cmd(["git", "push"])
    print(push_out)
    if code == 0:
        print("\nSuccessfully pushed changes to GitHub!")
    else:
        print("\n[ERROR] Git push encountered an error.")


if __name__ == "__main__":
    main()
