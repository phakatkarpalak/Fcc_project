"""Print a compact summary of an Excel file so we can see how its data is laid out.

Usage:  python inspect_xlsx.py "July 2022.xlsx"
Shows every sheet name, its size, and the first rows with each cell cut to 40 characters
(so a giant settings string does not flood the screen). Reads in read-only mode, so even a
50 MB file opens reasonably fast. Nothing is modified.
"""
import sys

from openpyxl import load_workbook


def short(v, n=40):
    s = "" if v is None else str(v).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python inspect_xlsx.py "file.xlsx" [rows_to_show]')
    path = sys.argv[1]
    show = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    wb = load_workbook(path, read_only=True, data_only=True)
    print(f"File: {path}")
    print(f"Sheets ({len(wb.sheetnames)}): {wb.sheetnames[:12]}{' ...' if len(wb.sheetnames) > 12 else ''}\n")
    for name in wb.sheetnames[:4]:
        ws = wb[name]
        print(f"=== Sheet {name!r}  (reported size: {ws.max_row} rows x {ws.max_column} columns)")
        for i, row in enumerate(ws.iter_rows(min_row=1, max_row=show, values_only=True), start=1):
            cells = [short(v) for v in list(row)[:10]]
            print(f"  row {i}: {cells}")
        print()
    if len(wb.sheetnames) > 4:
        print(f"(only the first 4 of {len(wb.sheetnames)} sheets shown)")


if __name__ == "__main__":
    main()
