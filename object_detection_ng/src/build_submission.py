"""Build and validate a competition submission zip.

Usage:
    python src/build_submission.py \
        --submission_dir submission \
        --output_zip submission.zip

Validates:
    - run.py exists at root
    - Only allowed file types
    - Total uncompressed size <= 420 MB
    - Max 10 .py files
    - Max 3 weight files
    - run.py at zip root (not nested)
"""

import argparse
import zipfile
from pathlib import Path


ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".cfg", ".pt", ".pth", ".onnx", ".safetensors", ".npy"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
MAX_UNCOMPRESSED_SIZE_MB = 420
MAX_PYTHON_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_TOTAL_FILES = 1000


def validate_submission_dir(submission_dir: Path) -> list[str]:
    """Validate the submission directory contents. Returns list of errors."""
    errors = []

    run_py = submission_dir / "run.py"
    if not run_py.exists():
        errors.append("CRITICAL: run.py not found in submission directory")

    all_files = [f for f in submission_dir.rglob("*") if f.is_file()]

    disallowed = []
    py_files = []
    weight_files = []
    total_size = 0

    for f in all_files:
        rel = f.relative_to(submission_dir)
        ext = f.suffix.lower()
        size = f.stat().st_size
        total_size += size

        if ext not in ALLOWED_EXTENSIONS:
            disallowed.append(str(rel))
        if ext == ".py":
            py_files.append(str(rel))
        if ext in WEIGHT_EXTENSIONS:
            weight_files.append((str(rel), size))

    if disallowed:
        errors.append(f"Disallowed file types: {disallowed}")

    if len(py_files) > MAX_PYTHON_FILES:
        errors.append(f"Too many .py files: {len(py_files)} > {MAX_PYTHON_FILES}")

    if len(weight_files) > MAX_WEIGHT_FILES:
        errors.append(f"Too many weight files: {len(weight_files)} > {MAX_WEIGHT_FILES}")

    total_mb = total_size / (1024 * 1024)
    weight_size_mb = sum(s for _, s in weight_files) / (1024 * 1024)

    if total_mb > MAX_UNCOMPRESSED_SIZE_MB:
        errors.append(f"Total size {total_mb:.1f} MB > {MAX_UNCOMPRESSED_SIZE_MB} MB limit")

    if weight_size_mb > MAX_UNCOMPRESSED_SIZE_MB:
        errors.append(f"Weight files size {weight_size_mb:.1f} MB > {MAX_UNCOMPRESSED_SIZE_MB} MB limit")

    if len(all_files) > MAX_TOTAL_FILES:
        errors.append(f"Too many files: {len(all_files)} > {MAX_TOTAL_FILES}")

    return errors


def build_zip(submission_dir: Path, output_zip: Path) -> None:
    """Build a zip with run.py at the root (not nested in a subfolder)."""
    all_files = sorted(f for f in submission_dir.rglob("*") if f.is_file())

    print(f"\nFILE MANIFEST ({len(all_files)} files):")
    print("-" * 60)
    total_size = 0
    for f in all_files:
        rel = f.relative_to(submission_dir)
        size = f.stat().st_size
        total_size += size
        size_str = f"{size / (1024*1024):.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
        print(f"  {str(rel):40s}  {size_str:>10s}")
    print("-" * 60)
    print(f"  TOTAL: {total_size / (1024*1024):.2f} MB")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in all_files:
            arcname = str(f.relative_to(submission_dir))
            zf.write(f, arcname)

    zip_size = output_zip.stat().st_size
    print(f"\nCreated: {output_zip} ({zip_size / (1024*1024):.2f} MB compressed)")

    # Verify zip structure
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()
        if "run.py" not in names:
            print("ERROR: run.py is NOT at the zip root!")
            # Show what's inside
            print("Zip contents:")
            for n in names:
                print(f"  {n}")
        else:
            print("VERIFIED: run.py is at zip root")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build competition submission zip")
    parser.add_argument("--submission_dir", type=str, default="submission", help="Path to submission folder")
    parser.add_argument("--output_zip", type=str, default="submission.zip", help="Output zip path")
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir)
    output_zip = Path(args.output_zip)

    if not submission_dir.exists():
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")

    print("Validating submission directory...")
    errors = validate_submission_dir(submission_dir)

    if errors:
        print("\nVALIDATION ERRORS:")
        for e in errors:
            print(f"  {e}")
        if any("CRITICAL" in e for e in errors):
            print("\nAborting due to critical errors.")
            return

    if not errors:
        print("All validation checks passed.")

    build_zip(submission_dir, output_zip)


if __name__ == "__main__":
    main()
