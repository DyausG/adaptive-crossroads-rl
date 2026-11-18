"""Rebuild the editable academic report using its retained Word format.

Install requirements-report.txt, then run this entry point. On Windows with
Microsoft Word installed, scripts/render_project_report.ps1 refreshes fields
and exports the matching PDF.
"""
from build_template_report import main


if __name__ == '__main__':
    main()
