<#
PowerShell script to rename files in a folder by creation time.
Usage:
  .\rename_by_creation.ps1 -Folder 'C:\Users\yiutse\Desktop\8330' -WhatIf

This will rename files to: YYYYMMDD_HHMMSS_originalname.ext
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$Folder,
    [switch]$WhatIf
)

$dir = Get-Item -Path $Folder
if (-not $dir.Exists) {
    Write-Error "Folder not found: $Folder"
    exit 1
}

Get-ChildItem -Path $Folder -File | Sort-Object Name | ForEach-Object {
    $created = $_.CreationTime
    $prefix = $created.ToString('yyyyMMdd_HHmmss')
    $newName = "${prefix}_$($_.Name)"
    $target = Join-Path $Folder $newName
    $counter = 1
    while (Test-Path $target) {
        $newName = "${prefix}_$('{0:D3}' -f $counter)_$($_.Name)"
        $target = Join-Path $Folder $newName
        $counter++
    }
    if ($WhatIf) {
        Write-Output "WhatIf: $($_.Name) -> $newName"
    } else {
        Rename-Item -Path $_.FullName -NewName $newName
        Write-Output "Renamed: $($_.Name) -> $newName"
    }
}
