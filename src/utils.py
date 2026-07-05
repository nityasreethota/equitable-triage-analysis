import os
import matplotlib.pyplot as plt
from src.config import OUTPUT_DIR, DPI, FIGURE_TITLE_SUFFIX


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(filename, tight=True):
    ensure_output_dir()
    if tight:
        plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved to {filepath}")


def print_header(title, width=65):
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def print_separator(width=65):
    print("-" * width)


def check_satisfied(value_A, value_B, tol=0.05):
    return "Y" if abs(value_A - value_B) < tol else "N VIOLATED"


def format_comparison_row(label, val_A, val_B,
                           width_label=35, width_val=10):
    satisfied = check_satisfied(val_A, val_B)
    return (f"{label:<{width_label}} "
            f"{val_A:>{width_val}.3f} "
            f"{val_B:>{width_val}.3f} "
            f"{satisfied:>{width_val}}")