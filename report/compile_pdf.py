"""
Compile the LaTeX report to PDF using pdflatex (MiKTeX).

Purpose: Automates LaTeX-to-PDF conversion.
Team member: Ishan Biswas
Key function: compile_report()
"""

import subprocess
import sys
import os


def compile_report():
    """Run pdflatex twice (for cross-references) on main.tex."""
    report_dir = os.path.dirname(os.path.abspath(__file__))
    tex_file = os.path.join(report_dir, "main.tex")

    if not os.path.exists(tex_file):
        print(f"ERROR: {tex_file} not found.")
        sys.exit(1)

    for pass_num in (1, 2):
        print(f"\n{'='*60}")
        print(f"  pdflatex pass {pass_num}/2")
        print(f"{'='*60}\n")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_dir, tex_file],
            cwd=report_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            print(f"\nERROR: pdflatex pass {pass_num} failed (exit code {result.returncode}).")
            sys.exit(result.returncode)

    pdf_path = os.path.join(report_dir, "main.pdf")
    if os.path.exists(pdf_path):
        print(f"\nSUCCESS: PDF generated at {pdf_path}")
    else:
        print("\nERROR: PDF file was not created.")
        sys.exit(1)


if __name__ == "__main__":
    compile_report()
