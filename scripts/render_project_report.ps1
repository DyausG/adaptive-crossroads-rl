param(
    [string]$DocumentPath = (Join-Path $PSScriptRoot '..\reports\final\Adaptive_Crossroads_Project_Report.docx'),
    [string]$PdfPath = (Join-Path $PSScriptRoot '..\reports\final\Adaptive_Crossroads_Project_Report.pdf')
)

$ErrorActionPreference = 'Stop'
$inputReport = (Resolve-Path -LiteralPath $DocumentPath).Path
$outputReport = [System.IO.Path]::GetFullPath($PdfPath)
$wordApp = $null
$wordDoc = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $wordApp.AutomationSecurity = 3
    $wordDoc = $wordApp.Documents.Open($inputReport, $false, $false, $false)
    $null = $wordDoc.Fields.Update()
    for ($tocIndex = 1; $tocIndex -le $wordDoc.TablesOfContents.Count; $tocIndex++) {
        $wordDoc.TablesOfContents.Item($tocIndex).Update()
    }
    $wordDoc.Repaginate()
    for ($tocIndex = 1; $tocIndex -le $wordDoc.TablesOfContents.Count; $tocIndex++) {
        $wordDoc.TablesOfContents.Item($tocIndex).UpdatePageNumbers()
    }
    $wordDoc.Save()
    $wordDoc.ExportAsFixedFormat($outputReport, 17)
    Write-Output ('Rendered report: ' + $outputReport)
}
finally {
    if ($null -ne $wordDoc) {
        try { $wordDoc.Close(0) } catch { Write-Verbose $_ }
    }
    if ($null -ne $wordApp) {
        try { $wordApp.Quit(0) } catch { Write-Verbose $_ }
        try { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) } catch { Write-Verbose $_ }
    }
}
