# Initialize build environment
Write-Host 'Update dependencies...' -ForegroundColor Magenta
.\update_dependencies.ps1

Write-Host 'Finding Visual Studio...' -ForegroundColor Magenta
$vsPath = .\Tools\vswhere.exe -latest -property installationPath
Write-Host $vsPath

Write-Host 'Importing environment variables...' -ForegroundColor Magenta
cmd.exe /c "call `"$vsPath\VC\Auxiliary\Build\vcvars64.bat`" && set > %temp%\vcvars.txt"
Get-Content "$env:temp\vcvars.txt" | Foreach-Object {
  if ($_ -match "^(.*?)=(.*)$") {
      Set-Content "env:\$($matches[1])" $matches[2]
  }
}

# Build projects
$coreCount = (Get-CimInstance -class Win32_ComputerSystem).NumberOfLogicalProcessors
$configurations = "Debug", "Release"
$platforms = "x64"

foreach ($platform in $platforms) {
  foreach ($config in $configurations) {
    Write-Host "Building $platform $config..." -ForegroundColor Magenta
    MSBuild.exe .\Axodox.MachineLearning.sln -p:Configuration=$config -p:Platform=$platform -m:$coreCount -v:m
  }
}

# Pack nuget
New-Item -Path '.\Output' -ItemType Directory -Force

$xml = [xml](Get-Content Axodox.MachineLearning.nuspec)

$xml.package.metadata.version = if ($null -ne $env:APPVEYOR_BUILD_VERSION) { $env:APPVEYOR_BUILD_VERSION } else { "1.0.0.0" }
$xml.package.metadata.repository.branch = if ($null -ne $env:APPVEYOR_REPO_BRANCH) { $env:APPVEYOR_REPO_BRANCH } else { "main" }
$commit = if ($null -ne $env:APPVEYOR_REPO_COMMIT) { $env:APPVEYOR_REPO_COMMIT } else { $null }
if ($null -ne $commit) {
  $xml.package.metadata.repository.SetAttribute("commit", $commit)
}

$xml.Save('.\Axodox.MachineLearning.Patched.nuspec')
.\Tools\nuget.exe pack .\Axodox.MachineLearning.Patched.nuspec -OutputDirectory .\Output
Remove-Item -Path '.\Axodox.MachineLearning.Patched.nuspec'