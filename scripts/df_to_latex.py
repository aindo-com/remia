import pandas as pd
from pathlib import Path
from numbers import Integral, Real
from typing import Any


r"""EXAMPLE OF LATEX TABLE
\begin{tabular}{lll}
    \toprule
    \multicolumn{2}{c}{Part}                   \\
    \cmidrule(r){1-2}
    Name     & Description     & Size ($\mu$m) \\
    \midrule
    Dendrite & Input terminal  & $\approx$100     \\
    Axon     & Output terminal & $\approx$10      \\
    Soma     & Cell body       & up to $10^6$  \\
    \bottomrule
\end{tabular}
"""


def _escape_latex(value: object) -> str:
    if isinstance(value, str) and ("$" in value or "\\" in value):
        return value
    epsilon_token = "\ue000"
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = str(value).replace("ε", epsilon_token)
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    escaped = escaped.replace(epsilon_token, r"$\epsilon$")
    return escaped


def _escape_latex_partial(value: str) -> str:
    """Escape only dangerous LaTeX characters, preserve math mode and formatting."""
    if "$" in value or "\\" in value:
        return value
    epsilon_token = "\ue000"
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
    }
    escaped = str(value).replace("ε", epsilon_token)
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    escaped = escaped.replace(epsilon_token, r"$\epsilon$")
    return escaped


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return bool(result) if isinstance(result, bool) else False


def _format_cell(value: Any, digits: int) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, str):
        return _escape_latex_partial(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, Real):
        return f"{float(value):.{digits}g}"
    return str(value)


def _as_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", ""}:
            return False
    return bool(value)


def _normalize_for_compare(value: Any) -> Any | None:
    if _is_missing(value):
        return None
    return value


def _format_header_label(value: Any) -> str:
    if _is_missing(value):
        return ""
    return _escape_latex(value)


def _is_numeric_value(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and not _is_missing(value)
    )


def _format_fixed_decimal(value: Any, digits: int) -> str:
    return f"{abs(float(value)):.{digits}f}"


def _format_aligned_numeric_value(
    value: Any,
    digits: int,
    integer_width: int,
    pad_sign: bool,
    is_integer_column: bool = False,
) -> str:
    if is_integer_column:
        int_str = str(int(round(float(value))))
        padding_width = max(integer_width - len(int_str), 0)
        padding = rf"\phantom{{{'0' * padding_width}}}" if padding_width else ""
        sign = "-" if float(value) < 0 else (r"\phantom{-}" if pad_sign else "")
        return f"{sign}{padding}{int_str}"

    fixed_value = _format_fixed_decimal(value, digits)
    integer_part, fractional_part = fixed_value.split(".")
    padding_width = max(integer_width - len(integer_part), 0)
    padding = rf"\phantom{{{'0' * padding_width}}}" if padding_width else ""
    sign = "-" if float(value) < 0 else (r"\phantom{-}" if pad_sign else "")
    return f"{sign}{padding}{integer_part}.{fractional_part}"


def _build_header_lines(
    columns: pd.Index, index_header: str, numeric_columns: list[bool]
) -> list[str]:
    if not isinstance(columns, pd.MultiIndex):
        header_cells = [index_header]
        for col_pos, column in enumerate(columns):
            label = _format_header_label(column)
            if numeric_columns[col_pos]:
                label = rf"\multicolumn{{1}}{{c}}{{{label}}}"
            header_cells.append(label)
        return ["    " + " & ".join(header_cells) + r" \\"]

    tuples = list(columns.to_list())
    nlevels = columns.nlevels
    lines: list[str] = []

    for level in range(nlevels):
        row_cells = [index_header if level == nlevels - 1 else ""]
        col_pos = 0

        while col_pos < len(tuples):
            current_label = tuples[col_pos][level]
            current_parent = tuple(
                _normalize_for_compare(v) for v in tuples[col_pos][:level]
            )
            span = 1

            while col_pos + span < len(tuples):
                next_label = tuples[col_pos + span][level]
                next_parent = tuple(
                    _normalize_for_compare(v) for v in tuples[col_pos + span][:level]
                )
                if _normalize_for_compare(next_label) != _normalize_for_compare(
                    current_label
                ):
                    break
                if next_parent != current_parent:
                    break
                span += 1

            label = _format_header_label(current_label)
            if span > 1:
                row_cells.append(rf"\multicolumn{{{span}}}{{c}}{{{label}}}")
            else:
                if level == nlevels - 1 and numeric_columns[col_pos]:
                    label = rf"\multicolumn{{1}}{{c}}{{{label}}}"
                row_cells.append(label)

            col_pos += span

        lines.append("    " + " & ".join(row_cells) + r" \\")

    return lines


def df_to_latex(
    *,
    df: pd.DataFrame,
    df_std: pd.DataFrame | None = None,
    highlight: pd.DataFrame | None = None,
    filename: str,
    digits: int = 3,
) -> None:
    """Convert a DataFrame to a LaTeX tabular and write it to filename.

    Cells are underlined when the corresponding value in highlight is truthy.
    Supports both flat and MultiIndex columns.
    Numeric values are formatted with ``digits`` decimal places and aligned on
    the decimal point within each column.
    Standard deviations from df_std are appended as smaller ± values.
    """
    if not isinstance(digits, int) or digits < 1:
        raise ValueError("'digits' must be an integer >= 1")

    if highlight is None:
        highlight = pd.DataFrame(False, index=df.index, columns=df.columns)

    if df.shape != highlight.shape:
        raise ValueError("'df' and 'highlight' must have the same shape")
    if not df.index.equals(highlight.index):
        raise ValueError("'df' and 'highlight' must have the same row index")
    if not df.columns.equals(highlight.columns):
        raise ValueError("'df' and 'highlight' must have the same columns")

    if df_std is not None:
        if df.shape != df_std.shape:
            raise ValueError("'df' and 'df_std' must have the same shape")
        if not df.index.equals(df_std.index):
            raise ValueError("'df' and 'df_std' must have the same row index")
        if not df.columns.equals(df_std.columns):
            raise ValueError("'df' and 'df_std' must have the same columns")

    numeric_column_settings: list[tuple[int, bool, bool] | None] = []
    for col_pos in range(len(df.columns)):
        column_values = list(df.iloc[:, col_pos])
        if df_std is not None:
            column_values.extend(list(df_std.iloc[:, col_pos]))

        numeric_values = [value for value in column_values if _is_numeric_value(value)]
        if not numeric_values:
            numeric_column_settings.append(None)
            continue

        integer_width = 1
        pad_sign = any(float(value) < 0 for value in numeric_values)
        is_integer_column = all(
            float(value) == int(float(value)) for value in numeric_values
        )

        for value in numeric_values:
            if is_integer_column:
                int_str = str(int(round(float(value))))
                integer_width = max(integer_width, len(int_str))
            else:
                fixed_value = _format_fixed_decimal(value, digits)
                integer_part = fixed_value.split(".")[0]
                integer_width = max(integer_width, len(integer_part))

        numeric_column_settings.append((integer_width, pad_sign, is_integer_column))

    numeric_columns = [settings is not None for settings in numeric_column_settings]
    column_alignment_parts = ["l"]
    for settings in numeric_column_settings:
        if settings is None:
            column_alignment_parts.append("l")
            continue
        integer_width, _pad_sign, is_integer_column = settings
        if is_integer_column:
            table_format = f"{integer_width}"
        else:
            table_format = f"{integer_width}.{digits}"
        if df_std is not None:
            table_format = f"{table_format}({digits})"
        column_alignment_parts.append(f"S[table-format={table_format}]")
    column_alignment = "".join(column_alignment_parts)
    index_header = _escape_latex(df.index.name) if df.index.name else ""
    header_lines = _build_header_lines(df.columns, index_header, numeric_columns)

    lines = [
        f"\\begin{{tabular}}{{{column_alignment}}}",
        "    \\toprule",
        *header_lines,
        "    \\midrule",
    ]

    for row_pos, index_value in enumerate(df.index):
        row_cells = [_escape_latex(index_value)]
        for col_pos in range(len(df.columns)):
            cell_value = df.iat[row_pos, col_pos]
            cell = _format_cell(cell_value, digits)

            numeric_settings = numeric_column_settings[col_pos]
            std_value = None if df_std is None else df_std.iat[row_pos, col_pos]
            if numeric_settings is not None and _is_numeric_value(cell_value):
                value_str = f"{float(cell_value):.{digits}f}"
                if std_value is not None and _is_numeric_value(std_value):
                    std_str = f"{float(std_value):.{digits}f}"
                    cell = rf"\num{{{value_str} +- {std_str}}}"
                else:
                    cell = rf"\num{{{value_str}}}"
            elif numeric_settings is not None:
                cell = r"\multicolumn{1}{c}{-}"
            elif std_value is not None and not _is_missing(std_value):
                std_str = _format_cell(std_value, digits)
                if std_str and std_str != "-":
                    cell = f"{cell}{{\\footnotesize$\\pm{std_str}$}}"

            if _as_bool(highlight.iat[row_pos, col_pos]) and cell:
                cell = rf"\underline{{{cell}}}"
            row_cells.append(cell)
        lines.append("    " + " & ".join(row_cells) + r" \\")

    lines.extend(["    \\bottomrule", "\\end{tabular}"])

    output_path = Path(filename.lower().replace(" ", "_"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
