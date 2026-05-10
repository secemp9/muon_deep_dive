#!/usr/bin/env python3
"""
Auto-transform all 57 notebooks into scientifically verbose research notebooks.

Strategy:
1. Parse the .py docstring to extract title, hypothesis, setup, etc.
2. Scan code cells for structural patterns (function defs, training loops, plotting, etc.)
3. Generate section annotations and print injections automatically
4. Apply via transform_notebook.py
"""

import json
import os
import re
import sys
import uuid
import glob
import textwrap


def load_notebook(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


def make_markdown_cell(lines):
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
    return ''.join(cell.get('source', []))


def extract_docstring(py_path):
    """Extract the module-level docstring from a .py file."""
    with open(py_path, 'r') as f:
        content = f.read()

    # Skip shebang
    lines = content.split('\n')
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#!') or stripped == '' or stripped.startswith('#'):
            continue
        start = i
        break

    # Find docstring
    remaining = '\n'.join(lines[start:])
    # Try triple double quotes
    match = re.match(r'\s*"""(.*?)"""', remaining, re.DOTALL)
    if not match:
        match = re.match(r"\s*'''(.*?)'''", remaining, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_docstring(docstring):
    """Parse a docstring into sections: title, hypothesis, setup, context, etc."""
    result = {
        'title': '',
        'hypothesis': '',
        'setup': '',
        'context': '',
        'measurements': '',
        'key_tests': '',
        'full_text': docstring
    }

    lines = docstring.split('\n')

    # Title is typically the first non-empty, non-decoration line
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('=') and not stripped.startswith('-'):
            result['title'] = stripped
            break

    # Extract sections by keyword matching
    current_section = 'preamble'
    section_lines = {'preamble': [], 'hypothesis': [], 'setup': [], 'context': [],
                     'measurements': [], 'key_tests': [], 'question': [], 'other': []}

    for line in lines[1:]:  # Skip title line
        stripped_upper = line.strip().upper()

        if any(kw in stripped_upper for kw in ['HYPOTHESIS:', 'HYPOTHESIS (', 'PREDICTION:', 'PREDICTION (']):
            current_section = 'hypothesis'
            # Keep the line if it has content after the keyword
            after = line.strip()
            for kw in ['HYPOTHESIS:', 'PREDICTION:']:
                if kw in after.upper():
                    after = after[after.upper().index(kw) + len(kw):].strip()
                    if after:
                        section_lines['hypothesis'].append(after)
            continue
        elif any(kw in stripped_upper for kw in ['THE QUESTION:', 'QUESTION:']):
            current_section = 'question'
            continue
        elif any(kw in stripped_upper for kw in ['SETUP:', 'PROTOCOL:', 'ARCHITECTURE:']):
            current_section = 'setup'
            continue
        elif any(kw in stripped_upper for kw in ['CONTEXT:', 'MOTIVATION:', 'BACKGROUND:',
                                                   'CRITICAL CONTEXT', 'FROM ']):
            if 'FROM ' in stripped_upper and current_section == 'setup':
                # "FROM 2.11:" at start of setup
                section_lines[current_section].append(line)
                continue
            current_section = 'context'
            continue
        elif any(kw in stripped_upper for kw in ['MEASUREMENT', 'METRIC', 'WHAT WE MEASURE']):
            current_section = 'measurements'
            continue
        elif any(kw in stripped_upper for kw in ['KEY TEST', 'TESTS:', 'KEY QUESTION']):
            current_section = 'key_tests'
            continue
        elif any(kw in stripped_upper for kw in ['OPTIMIZERS COMPARED', 'OPTIMIZER']):
            current_section = 'setup'
            section_lines[current_section].append(line)
            continue
        elif any(kw in stripped_upper for kw in ['FIX:', 'ALSO TEST', 'VARIANTS:']):
            current_section = 'setup'
            section_lines[current_section].append(line)
            continue
        elif stripped_upper.startswith('===') or stripped_upper.startswith('---'):
            continue

        section_lines[current_section].append(line)

    for key in section_lines:
        text = '\n'.join(section_lines[key]).strip()
        if text:
            if key == 'hypothesis':
                result['hypothesis'] = text
            elif key == 'question':
                if not result['hypothesis']:
                    result['hypothesis'] = text
                else:
                    result['hypothesis'] = text + '\n\n' + result['hypothesis']
            elif key == 'setup':
                result['setup'] = text
            elif key == 'context':
                result['context'] = text
            elif key == 'measurements':
                result['measurements'] = text
            elif key == 'key_tests':
                result['key_tests'] = text
            elif key == 'preamble' and not result['hypothesis']:
                # Use preamble as hypothesis if no explicit hypothesis found
                result['hypothesis'] = text

    return result


def generate_expected_outcomes(hypothesis_text, key_tests_text):
    """Generate expected outcomes from hypothesis and key tests."""
    parts = []
    if hypothesis_text:
        for line in hypothesis_text.split('\n'):
            stripped = line.strip()
            if any(kw in stripped.lower() for kw in ['expect', 'predict', 'should', 'will',
                                                       'would', 'lambda', 'ratio']):
                parts.append(f"- {stripped}")
    if key_tests_text:
        if parts:
            parts.append("")
        parts.append("**Key tests to evaluate:**")
        for line in key_tests_text.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('=') and not stripped.startswith('-'):
                if not stripped.startswith('- '):
                    parts.append(f"- {stripped}")
                else:
                    parts.append(stripped)
    if not parts:
        return "See hypothesis section above. Results will be evaluated against predictions after execution."
    return '\n'.join(parts[:20])


def detect_key_variables(py_path):
    """Detect key variables and patterns in the code."""
    with open(py_path, 'r') as f:
        content = f.read()

    return {
        'has_loss_curves': bool(re.search(r'loss_curve|losses|loss_mean', content)),
        'has_condition_number': bool(re.search(r'cond|kappa|condition', content, re.IGNORECASE)),
        'has_svd': bool(re.search(r'svd|singular', content, re.IGNORECASE)),
        'has_diversity': bool(re.search(r'diversity|pairwise', content, re.IGNORECASE)),
        'has_hessian': bool(re.search(r'[Hh]essian|hess_', content)),
        'has_alignment': bool(re.search(r'alignment|cosine|dot_product', content, re.IGNORECASE)),
        'has_spectrum': bool(re.search(r'spectrum|eigenval|eigenvec', content, re.IGNORECASE)),
        'has_lr_sweep': bool(re.search(r'lr_sweep|find_best_lr|candidates', content)),
        'has_multiple_optimizers': bool(re.search(r'sgd.*muon|muon.*sgd|optimizer', content, re.IGNORECASE)),
    }


def transform_single_notebook(nb_dir):
    """Transform a single notebook directory."""
    py_path = os.path.join(nb_dir, 'run_experiment.py')
    nb_path = os.path.join(nb_dir, 'run_experiment.ipynb')

    if not os.path.exists(nb_path):
        print(f"  SKIP: No notebook at {nb_path}")
        return False
    if not os.path.exists(py_path):
        print(f"  SKIP: No .py at {py_path}")
        return False

    # Load and parse
    docstring = extract_docstring(py_path)
    parsed = parse_docstring(docstring)
    patterns = detect_key_variables(py_path)

    nb = load_notebook(nb_path)
    cells = nb['cells']
    original_count = len(cells)

    # =========================================================================
    # Build the header
    # =========================================================================
    title = parsed['title'] or os.path.basename(nb_dir).replace('_', ' ')

    # Clean hypothesis -- remove the title echo if present
    hypothesis = parsed['hypothesis'] or "See the experiment description below."
    # Remove lines that are just the title repeated
    hyp_lines = hypothesis.split('\n')
    hyp_lines = [l for l in hyp_lines if l.strip() != title.strip() and not l.strip().startswith('=')]
    hypothesis = '\n'.join(hyp_lines).strip()
    if not hypothesis:
        hypothesis = parsed.get('full_text', 'See experiment code below.')[:500]

    methodology = parsed['setup'] or "See the configuration and code cells below for detailed setup."
    expected = generate_expected_outcomes(parsed['hypothesis'], parsed['key_tests'])

    header_lines = [
        f"# {title}\n",
        "\n",
        "## Hypothesis\n",
        "\n",
    ]
    for line in hypothesis.split('\n'):
        header_lines.append(f"{line}\n")
    header_lines.extend([
        "\n",
        "## Methodology\n",
        "\n",
    ])
    for line in methodology.split('\n'):
        header_lines.append(f"{line}\n")
    header_lines.extend([
        "\n",
        "## Expected Outcomes\n",
        "\n",
    ])
    for line in expected.split('\n'):
        header_lines.append(f"{line}\n")

    if parsed['context']:
        header_lines.extend(["\n", "## Prior Context\n", "\n"])
        for line in parsed['context'].split('\n'):
            header_lines.append(f"{line}\n")

    if parsed['measurements']:
        header_lines.extend(["\n", "## Measurements\n", "\n"])
        for line in parsed['measurements'].split('\n'):
            header_lines.append(f"{line}\n")

    header_cell = make_markdown_cell(header_lines)

    # =========================================================================
    # Replace or insert header
    # =========================================================================
    # Check if first cell is the docstring code cell
    if cells and cells[0]['cell_type'] == 'code':
        first_src = cell_source_text(cells[0])
        if first_src.strip().startswith('"""') or first_src.strip().startswith("'''"):
            cells[0] = header_cell
        else:
            cells.insert(0, header_cell)
    elif cells and cells[0]['cell_type'] == 'markdown':
        # Check if this is our previously-inserted header (from prior run)
        first_src = cell_source_text(cells[0])
        if '## Hypothesis' in first_src:
            # Already transformed -- replace
            cells[0] = header_cell
        else:
            cells.insert(0, header_cell)
    else:
        cells.insert(0, header_cell)

    # =========================================================================
    # Process cells: insert annotations and print statements
    # =========================================================================
    # We'll build a new cell list with insertions
    new_cells = [cells[0]]  # Start with header

    # Track what kinds of annotations we've already added to avoid duplicates
    added_sections = set()
    prev_cell_type = 'markdown'

    for i in range(1, len(cells)):
        cell = cells[i]
        src = cell_source_text(cell)

        if cell['cell_type'] == 'markdown':
            # Upgrade existing thin markdown cells
            stripped = src.strip()
            if stripped.startswith('=') and stripped.endswith('='):
                inner = stripped.strip('= \n').strip()
                if inner and len(inner) < 100:
                    cell['source'] = [f"---\n", f"### {inner}\n"]
            elif stripped.startswith('-') and stripped.endswith('-'):
                inner = stripped.strip('- \n').strip()
                if inner and len(inner) < 100:
                    cell['source'] = [f"### {inner}\n"]

            new_cells.append(cell)
            prev_cell_type = 'markdown'
            continue

        # It's a code cell
        prev_is_md = (prev_cell_type == 'markdown')

        # --- Imports ---
        if (re.search(r'^import numpy|^import torch|^from ', src, re.MULTILINE)
            and len(new_cells) <= 4 and 'imports' not in added_sections):
            if not prev_is_md:
                new_cells.append(make_markdown_cell([
                    "## Environment Setup\n",
                    "\n",
                    "Import required libraries and configure the computational environment.\n"
                ]))
                added_sections.add('imports')

        # --- Random seed ---
        if re.search(r'np\.random\.seed|torch\.manual_seed|random\.seed', src):
            if 'def ' not in src:
                source = cell['source']
                if source and not source[-1].endswith('\n'):
                    source[-1] += '\n'
                source.append('\nprint(f"Random seed set for reproducibility")\n')

        # --- Configuration block ---
        config_vars = re.findall(r'^([A-Z][A-Z_0-9]{2,})\s*=\s*(.+)', src, re.MULTILINE)
        if config_vars and 'def ' not in src and len(config_vars) >= 2:
            if not prev_is_md and 'config' not in added_sections:
                new_cells.append(make_markdown_cell([
                    "## Experimental Configuration\n",
                    "\n",
                    "Define the hyperparameters and experimental setup. These parameters control\n",
                    "the network architecture, training duration, and evaluation protocol.\n"
                ]))
                added_sections.add('config')

            # Add config summary print
            config_print_lines = ['print("\\n--- Experimental Configuration ---")\n']
            for var_name, var_val in config_vars:
                if var_name not in ['OPTIMIZER_NAMES', 'OPTIMIZER_LABELS', 'OPTIMIZER_COLORS',
                                     'SCRIPT_DIR'] and 'dict' not in var_val and '{' not in var_val:
                    config_print_lines.append(f'print(f"  {var_name} = {{{var_name}}}")\n')
            if len(config_print_lines) > 1:
                source = cell['source']
                if source and not source[-1].endswith('\n'):
                    source[-1] += '\n'
                source.append('\n')
                for line in config_print_lines:
                    source.append(line)

        # --- Data generation ---
        data_vars = re.findall(r'^(W_target|X_data|X_test|X_train|Y_train|y_target|X_val)\s*=',
                               src, re.MULTILINE)
        if data_vars and 'def ' not in src:
            if not prev_is_md and 'data_gen' not in added_sections:
                new_cells.append(make_markdown_cell([
                    "## Data Generation\n",
                    "\n",
                    "Generate the training and test data. Fixed random targets and inputs ensure\n",
                    "reproducible comparisons across optimizers and experimental conditions.\n"
                ]))
                added_sections.add('data_gen')

            # Add data shape prints
            data_print_lines = ['print("\\n--- Data Shapes & Statistics ---")\n']
            for var in data_vars:
                data_print_lines.append(
                    f'print(f"  {var}: shape={{{var}.shape}}, mean={{{var}.mean():.6f}}, std={{{var}.std():.6f}}")\n'
                )
            source = cell['source']
            if source and not source[-1].endswith('\n'):
                source[-1] += '\n'
            source.append('\n')
            for line in data_print_lines:
                source.append(line)

        # --- Function definitions ---
        func_matches = re.findall(r'^def (\w+)\(', src, re.MULTILINE)
        if func_matches and not prev_is_md:
            func_names = ', '.join(f'`{f}`' for f in func_matches)
            is_network = any(kw in f for f in func_matches
                           for kw in ['forward', 'loss', 'gradient', 'backward'])
            is_optimizer = any(kw in f for f in func_matches
                             for kw in ['step', 'optim', 'newton', 'schulz', 'muon'])
            is_training = any(kw in f for f in func_matches
                            for kw in ['train', 'run_', 'sweep', 'find_best'])
            is_analysis = any(kw in f for f in func_matches
                            for kw in ['measure', 'compute_', 'analyze', 'eval'])
            is_init = any(kw in f for f in func_matches
                        for kw in ['init', 'create', 'build', 'make'])
            is_hessian = any(kw in f for f in func_matches
                           for kw in ['hessian', 'hess'])
            is_util = not (is_network or is_optimizer or is_training or is_analysis or is_init or is_hessian)

            # Decide section title based on function type
            if is_network and 'network_arch' not in added_sections:
                section_title = "## Network Architecture"
                section_desc = f"Define the neural network components: forward pass, loss computation, and gradient calculation.\n\nFunctions defined: {func_names}"
                added_sections.add('network_arch')
            elif is_optimizer and 'optimizer_def' not in added_sections:
                section_title = "## Optimizer Definitions"
                section_desc = f"Define the optimization algorithms being compared.\n\nFunctions defined: {func_names}"
                added_sections.add('optimizer_def')
            elif is_training and 'training_engine' not in added_sections:
                section_title = "## Training Engine"
                section_desc = f"Core training loop and evaluation utilities.\n\nFunctions defined: {func_names}"
                added_sections.add('training_engine')
            elif is_analysis and 'analysis_funcs' not in added_sections:
                section_title = "## Analysis Functions"
                section_desc = f"Metric computation and analysis utilities.\n\nFunctions defined: {func_names}"
                added_sections.add('analysis_funcs')
            elif is_hessian and 'hessian_funcs' not in added_sections:
                section_title = "## Hessian Computation"
                section_desc = f"Functions for computing and analyzing the Hessian matrix.\n\nFunctions defined: {func_names}"
                added_sections.add('hessian_funcs')
            elif is_init and 'init_funcs' not in added_sections:
                section_title = "## Initialization"
                section_desc = f"Weight initialization and network construction.\n\nFunctions defined: {func_names}"
                added_sections.add('init_funcs')
            else:
                section_title = None
                section_desc = None

            if section_title:
                new_cells.append(make_markdown_cell([
                    f"{section_title}\n",
                    "\n",
                    f"{section_desc}\n"
                ]))

        # --- Main experiment execution (non-function code blocks with training) ---
        is_main_exec = (
            'def ' not in src
            and len(src) > 100
            and (re.search(r'for\s+(net_type|method|opt|m)\s+in', src)
                 or re.search(r'find_best_lr\(|run_training\(|measure_convergence|run_experiment', src)
                 or re.search(r'Phase 1|Phase 2', src))
        )

        if is_main_exec and not prev_is_md and 'main_exec' not in added_sections:
            new_cells.append(make_markdown_cell([
                "## Main Experiment Execution\n",
                "\n",
                "Run the full experimental protocol. This is the core computation block\n",
                "where all optimizers are trained and compared. Results are printed\n",
                "as each phase completes for progress monitoring.\n"
            ]))
            added_sections.add('main_exec')

        # --- Results tables ---
        if (re.search(r'RESULTS TABLE|COMPREHENSIVE NUMBER', src) and 'def ' not in src
            and not prev_is_md):
            if 'results_table' not in added_sections:
                new_cells.append(make_markdown_cell([
                    "## Results Summary\n",
                    "\n",
                    "Display complete experimental results for systematic comparison across\n",
                    "all optimizers and conditions.\n"
                ]))
                added_sections.add('results_table')

        # --- Hypothesis tests ---
        if (re.search(r'KEY HYPOTHESIS|HYPOTHESIS TEST|total_pass|total_tests', src)
            and 'def ' not in src and not prev_is_md):
            if 'hyp_tests' not in added_sections:
                new_cells.append(make_markdown_cell([
                    "## Hypothesis Testing\n",
                    "\n",
                    "Evaluate experimental results against the stated hypotheses. Each test\n",
                    "compares observed quantities to predicted thresholds to determine\n",
                    "whether the theoretical framework is supported by the data.\n"
                ]))
                added_sections.add('hyp_tests')

        # --- Critical comparison / verdict ---
        if (re.search(r'CRITICAL COMPARISON|FINAL VERDICT|PAPER THESIS|CONCLUSION', src)
            and 'def ' not in src and not prev_is_md):
            if 'verdict' not in added_sections:
                new_cells.append(make_markdown_cell([
                    "## Interpretation & Verdict\n",
                    "\n",
                    "Synthesize all experimental evidence to determine whether the hypothesis\n",
                    "is supported, partially supported, or refuted. The verdict integrates\n",
                    "quantitative test results with qualitative patterns across conditions.\n"
                ]))
                added_sections.add('verdict')

        # --- Plotting ---
        if (re.search(r'plt\.figure|fig,\s*ax|plt\.subplot|fig\s*=\s*plt', src)
            and 'def ' not in src and not prev_is_md):
            if 'plotting' not in added_sections:
                new_cells.append(make_markdown_cell([
                    "## Visualization\n",
                    "\n",
                    "Generate diagnostic plots to visualize the experimental results. These\n",
                    "figures provide visual confirmation of the quantitative findings\n",
                    "reported in the tables above.\n"
                ]))
                added_sections.add('plotting')

        new_cells.append(cell)
        prev_cell_type = 'code'

    # =========================================================================
    # Add conclusions cell at the end
    # =========================================================================
    conclusion_lines = [
        "## Conclusions\n",
        "\n",
        f"This experiment tested: **{title}**\n",
        "\n",
    ]

    if parsed['hypothesis']:
        first_hyp_line = parsed['hypothesis'].split('\n')[0].strip()
        if len(first_hyp_line) > 10:
            conclusion_lines.extend([
                f"**Core hypothesis:** {first_hyp_line}\n",
                "\n",
            ])

    conclusion_lines.extend([
        "**Key findings** (visible in output cells above after execution):\n",
        "\n",
        "- Review the PASS/FAIL test results to determine overall hypothesis support\n",
        "- Examine the quantitative metrics in the results tables for effect sizes\n",
        "- Check the visualization plots for qualitative patterns and anomalies\n",
        "- Consider how these results connect to the broader Muon-as-RG-Gauge-Fixing framework\n",
    ])

    new_cells.append(make_markdown_cell(conclusion_lines))

    # =========================================================================
    # Fix SCRIPT_DIR for notebook context
    # =========================================================================
    for cell in new_cells:
        if cell['cell_type'] == 'code':
            src = cell_source_text(cell)
            if "os.path.dirname(os.path.abspath(__file__))" in src:
                new_src = src.replace(
                    "os.path.dirname(os.path.abspath(__file__))",
                    "os.path.dirname(os.path.abspath('.'))"
                )
                cell['source'] = [line + '\n' for line in new_src.rstrip('\n').split('\n')]

    # =========================================================================
    # Save
    # =========================================================================
    nb['cells'] = new_cells
    save_notebook(nb, nb_path)

    new_count = len(new_cells)
    print(f"  OK: {original_count} -> {new_count} cells ({new_count - original_count:+d})")
    return True


def main():
    base_dir = "/home/secemp9/Muon_as_RG_Gauge_Fixing/experiments"

    # Find all notebooks
    notebooks = []
    for root, dirs, files in os.walk(base_dir):
        if 'run_experiment.ipynb' in files:
            notebooks.append(root)
    notebooks.sort()

    print(f"Found {len(notebooks)} notebooks to transform\n")

    success = 0
    failed = 0
    for i, nb_dir in enumerate(notebooks):
        rel_path = os.path.relpath(nb_dir, base_dir)
        print(f"[{i+1:2d}/{len(notebooks)}] {rel_path}")
        try:
            if transform_single_notebook(nb_dir):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nDone: {success} transformed, {failed} failed/skipped")


if __name__ == '__main__':
    main()
