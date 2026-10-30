param(
    [Parameter(Mandatory=$true)][string]$Url,
    [Parameter(Mandatory=$true)][string]$OutFile,
    [int]$StallSeconds = 45,
    [int]$TotalMinutes = 30
)

$ErrorActionPreference = 'Stop'
$sslExe = 'D:\Program Files\Git\usr\bin\openssl.exe'

function Get-RawResponse {
    param([string]$HostName, [string]$Path)

    $req = "GET $Path HTTP/1.1`r`nHost: $HostName`r`nUser-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)`r`nAccept: */*`r`nAccept-Encoding: identity`r`nConnection: close`r`n`r`n"

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $sslExe
    $psi.Arguments = "s_client -quiet -connect ${HostName}:443 -servername $HostName"
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false

    $p = [System.Diagnostics.Process]::Start($psi)
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($req)
    $p.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
    $p.StandardInput.BaseStream.Flush()
    $p.StandardInput.Close()

    $stream = $p.StandardOutput.BaseStream
    $ms = New-Object System.IO.MemoryStream
    $buf = New-Object byte[] 65536
    $total = 0
    $headerEnd = -1
    $contentLength = -1
    $chunked = $false
    $deadline = (Get-Date).AddMinutes($TotalMinutes)
    $lastReport = 0

    while ((Get-Date) -lt $deadline) {
        $task = $stream.ReadAsync($buf, 0, $buf.Length)
        if (-not $task.Wait($StallSeconds * 1000)) { break }
        $n = $task.Result
        if ($n -le 0) { break }
        $ms.Write($buf, 0, $n)
        $total += $n

        if ($headerEnd -lt 0 -and $total -ge 4) {
            $arr = $ms.GetBuffer()
            for ($i = 0; $i -lt $total - 3; $i++) {
                if ($arr[$i] -eq 13 -and $arr[$i+1] -eq 10 -and $arr[$i+2] -eq 13 -and $arr[$i+3] -eq 10) {
                    $headerEnd = $i
                    $headerText = [System.Text.Encoding]::ASCII.GetString($arr, 0, $i)
                    if ($headerText -match 'Transfer-Encoding:\s*chunked') { $chunked = $true }
                    elseif ($headerText -match 'Content-Length:\s*(\d+)') { $contentLength = [int]$Matches[1] }
                    break
                }
            }
        }

        if ($total - $lastReport -ge 1048576) {
            $lastReport = $total
            Write-Host ("  ... {0:N1} MB" -f ($total / 1MB))
        }

        if ($headerEnd -ge 0 -and -not $chunked -and $contentLength -ge 0) {
            if ($total -ge ($headerEnd + 4 + $contentLength)) { break }
        }
        if ($chunked -and $total -ge 5) {
            $arr = $ms.GetBuffer()
            if ($arr[$total-5] -eq 48 -and $arr[$total-4] -eq 13 -and $arr[$total-3] -eq 10 -and $arr[$total-2] -eq 13 -and $arr[$total-1] -eq 10) { break }
        }
    }

    try { if (-not $p.HasExited) { $p.Kill() } } catch {}
    return $ms.ToArray()
}

function Split-Response {
    param([byte[]]$Raw)
    $sep = -1
    for ($i = 0; $i -lt $Raw.Length - 3; $i++) {
        if ($Raw[$i] -eq 13 -and $Raw[$i+1] -eq 10 -and $Raw[$i+2] -eq 13 -and $Raw[$i+3] -eq 10) {
            $sep = $i
            break
        }
    }
    if ($sep -lt 0) { throw "Cannot parse HTTP headers (got $($Raw.Length) bytes)" }
    $header = [System.Text.Encoding]::ASCII.GetString($Raw, 0, $sep)
    $bodyStart = $sep + 4

    if ($header -match 'Transfer-Encoding:\s*chunked') {
        $outMs = New-Object System.IO.MemoryStream
        $pos = $bodyStart
        while ($pos -lt $Raw.Length) {
            $nl = -1
            for ($i = $pos; $i -lt $Raw.Length - 1; $i++) {
                if ($Raw[$i] -eq 13 -and $Raw[$i+1] -eq 10) { $nl = $i; break }
            }
            if ($nl -lt 0) { break }
            $sizeLine = [System.Text.Encoding]::ASCII.GetString($Raw, $pos, $nl - $pos).Trim()
            $semi = $sizeLine.IndexOf(';')
            if ($semi -ge 0) { $sizeLine = $sizeLine.Substring(0, $semi) }
            if ([string]::IsNullOrWhiteSpace($sizeLine)) { $pos = $nl + 2; continue }
            $size = [Convert]::ToInt32($sizeLine, 16)
            if ($size -eq 0) { break }
            $start = $nl + 2
            $take = [Math]::Min($size, $Raw.Length - $start)
            if ($take -gt 0) { $outMs.Write($Raw, $start, $take) }
            $pos = $start + $size + 2
        }
        $body = $outMs.ToArray()
    } else {
        $len = $Raw.Length - $bodyStart
        if ($header -match 'Content-Length:\s*(\d+)') {
            $len = [Math]::Min([int]$Matches[1], $len)
        }
        $body = New-Object byte[] $len
        if ($len -gt 0) { [Array]::Copy($Raw, $bodyStart, $body, 0, $len) }
    }

    return [pscustomobject]@{ Header = $header; Body = $body }
}

$current = $Url
$maxRedirects = 8
for ($i = 0; $i -le $maxRedirects; $i++) {
    $u = [uri]$current
    if ($u.Scheme -ne 'https') { throw "Only https is supported: $current" }
    $path = $u.PathAndQuery
    if ([string]::IsNullOrEmpty($path)) { $path = '/' }

    Write-Output "GET $current"
    $raw = Get-RawResponse -HostName $u.Host -Path $path
    $resp = Split-Response -Raw $raw

    if ($resp.Header -match '^HTTP/\S+\s+(301|302|303|307|308)') {
        if ($resp.Header -match '(?im)^Location:\s*(\S+)') {
            $loc = $Matches[1].Trim()
            if ($loc -match '^https?://') { $current = $loc }
            else { $current = ([uri]::new($u, $loc)).AbsoluteUri }
            continue
        }
        throw "Redirect without Location"
    }
    if ($resp.Header -match '^HTTP/\S+\s+(\d+)') {
        $code = [int]$Matches[1]
        if ($code -ne 200) { throw "HTTP $code for $current" }
    }

    $dir = Split-Path -Parent $OutFile
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    [System.IO.File]::WriteAllBytes($OutFile, $resp.Body)
    Write-Output ("SAVED {0} ({1} bytes)" -f $OutFile, $resp.Body.Length)
    exit 0
}
throw "Too many redirects"
