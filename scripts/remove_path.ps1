$path = [Environment]::GetEnvironmentVariable('Path', 'User')
$remove = @(
    'D:\Python3.9\Scripts',
    'D:\Python3.9',
    'C:\Users\Lenovo\AppData\Local\Microsoft\WindowsApps'
)
$newPath = ($path -split ';' | Where-Object { 
    $_.Trim() -ne '' -and $_ -notin $remove 
}) -join ';'
[Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
Write-Host "已从用户 PATH 中移除："
$remove | ForEach-Object { Write-Host "  $_" }
