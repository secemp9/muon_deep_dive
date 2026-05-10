#!/usr/bin/env python3
"""
Transforms a bare-bones run_experiment.ipynb into a scientifically verbose
research notebook by:
  1. Adding a header markdown cell with title, hypothesis, methodology, expected outcomes
  2. Adding markdown cells before major code sections
  3. Adding print statements for sanity checks and intermediate diagnostics
  4. Adding a conclusions markdown cell at the end

Usage: python transform_notebook.py <notebook_path> <title> <hypothesis> <methodology> <expected_outcomes> <section_annotations_json>

But actually we'll call it as a library from per-notebook scripts.
"""

import json
import os
import sys
import copy
import uuid


def load_notebook(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


def make_markdown_cell(lines):
    """Create a markdown cell from a list of strings (each should end with \\n)."""
    # Ensure each line ends with \n
    processed = []
    for line in lines:
        if not line.endswith('\n'):
            processed.append(line + '\n')
        else:
            processed.append(line)
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": processed
    }


def make_code_cell(lines):
    """Create a code cell from a list of strings."""
    processed = []
    for line in lines:
        if not line.endswith('\n'):
            processed.append(line + '\n')
        else:
            processed.append(line)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": processed
    }


def cell_source_text(cell):
    """Get the full source text of a cell."""
    return ''.join(cell.get('source', []))


def cell_contains(cell, text):
    """Check if cell source contains the given text."""
    return text in cell_source_text(cell)


def find_cell_index(cells, text, start=0):
    """Find the index of the first cell containing the given text, starting from 'start'."""
    for i in range(start, len(cells)):
        if cell_contains(cells[i], text):
            return i
    return -1


def insert_before(cells, index, new_cell):
    """Insert a cell before the given index. Returns new index of the original cell."""
    cells.insert(index, new_cell)
    return index + 1


def add_prints_to_cell(cell, print_lines):
    """Add print statement lines at the end of a code cell's source."""
    if cell['cell_type'] != 'code':
        return
    source = cell['source']
    # Add a blank line separator then the print lines
    if source and not source[-1].endswith('\n'):
        source[-1] += '\n'
    source.append('\n')
    for line in print_lines:
        if not line.endswith('\n'):
            source.append(line + '\n')
        else:
            source.append(line)


def transform_notebook(nb_path, title, hypothesis, methodology, expected_outcomes,
                       section_annotations=None, print_injections=None,
                       conclusions=None):
    """
    Transform a notebook in-place.

    Args:
        nb_path: Path to the .ipynb file
        title: Experiment title
        hypothesis: The hypothesis being tested
        methodology: Methodology overview
        expected_outcomes: Expected outcomes
        section_annotations: List of dicts with keys:
            - 'before_text': text to search for in cells to insert before
            - 'markdown': list of markdown lines to insert
        print_injections: List of dicts with keys:
            - 'cell_text': text to find the target cell
            - 'prints': list of print statement lines to add
        conclusions: List of strings for the conclusions markdown cell
    """
    nb = load_notebook(nb_path)
    cells = nb['cells']

    if section_annotations is None:
        section_annotations = []
    if print_injections is None:
        print_injections = []
    if conclusions is None:
        conclusions = [
            "## Conclusions\n",
            "\n",
            "Results and interpretation will be visible in the output cells above after execution.\n"
        ]

    # =========================================================================
    # Step 1: Replace the first cell (docstring) with a proper markdown header
    # =========================================================================
    header_cell = make_markdown_cell([
        f"# {title}\n",
        "\n",
        "## Hypothesis\n",
        "\n",
        f"{hypothesis}\n",
        "\n",
        "## Methodology\n",
        "\n",
        f"{methodology}\n",
        "\n",
        "## Expected Outcomes\n",
        "\n",
        f"{expected_outcomes}\n",
    ])

    # Check if first cell is the docstring
    if cells and cells[0]['cell_type'] == 'code':
        first_src = cell_source_text(cells[0])
        if first_src.strip().startswith('"""') or first_src.strip().startswith("'''"):
            # Replace docstring cell with markdown header
            cells[0] = header_cell
        else:
            # Insert header before first cell
            cells.insert(0, header_cell)
    else:
        cells.insert(0, header_cell)

    # =========================================================================
    # Step 2: Insert section annotation markdown cells
    # =========================================================================
    # Process in reverse order to maintain indices
    insertions = []
    for ann in section_annotations:
        before_text = ann['before_text']
        idx = find_cell_index(cells, before_text)
        if idx >= 0:
            insertions.append((idx, make_markdown_cell(ann['markdown'])))

    # Sort by index descending so insertions don't shift earlier indices
    insertions.sort(key=lambda x: x[0], reverse=True)
    for idx, cell in insertions:
        cells.insert(idx, cell)

    # =========================================================================
    # Step 3: Add print statements to code cells
    # =========================================================================
    for pi in print_injections:
        cell_text = pi['cell_text']
        idx = find_cell_index(cells, cell_text)
        if idx >= 0 and cells[idx]['cell_type'] == 'code':
            add_prints_to_cell(cells[idx], pi['prints'])

    # =========================================================================
    # Step 4: Add conclusions cell at the end
    # =========================================================================
    cells.append(make_markdown_cell(conclusions))

    # =========================================================================
    # Step 5: Upgrade existing "banner" markdown cells to be more descriptive
    # =========================================================================
    # The auto-converted notebooks have markdown cells that are just section
    # banners like "=====\n CONFIGURATION \n=====" - we leave these as-is
    # since the section_annotations handle the scientific context

    nb['cells'] = cells
    save_notebook(nb, nb_path)
    return nb_path


if __name__ == '__main__':
    # Simple test
    if len(sys.argv) > 1:
        print(f"Would transform: {sys.argv[1]}")
