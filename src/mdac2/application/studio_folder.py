"""Explicit local user action: choose a delivery folder in the Windows dialog."""
import base64
import os
import subprocess


def choose_folder():
    if os.name!='nt':raise ValueError('当前目录选择器需要Windows桌面')
    # Fixed code, no path or user text interpolated into the shell script.
    script='''Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select Custom_Albums folder'
$dialog.ShowNewFolderButton = $true
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
try {
  if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Write([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($dialog.SelectedPath)))
  }
} finally { $dialog.Dispose(); $owner.Dispose() }
'''
    try:
        result=subprocess.run(['powershell.exe','-NoProfile','-STA','-EncodedCommand',
            base64.b64encode(script.encode('utf-16le')).decode()],capture_output=True,timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW,check=True)
        return base64.b64decode(result.stdout.strip(),validate=True).decode('utf-8') if result.stdout.strip() else None
    except (subprocess.SubprocessError,ValueError,OSError):raise ValueError('目录选择未完成，请重试') from None
