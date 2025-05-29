$dependencies = @("Axodox.Common")

$projects = Resolve-Path "./*/*.vcxproj"
$nuspecs = Resolve-Path "*.nuspec"

foreach ($dependency in $dependencies) {
  Write-Host "Fetching version of $dependency..."

  $latestPackage = .\Tools\nuget.exe list -allversions $dependency | Where-Object { $_ -match "^$dependency\s+" }
  $latestVersion = [regex]::Match($latestPackage, "\d+(\.\d+)+").Value

  Write-Host "  Found: $latestVersion"

  # Update projects
  foreach ($projectPath in $projects) {
    # Update project file
    $projectContent = Get-Content -Path $projectPath -Raw
    $projectContent = $projectContent -replace "\\packages\\$dependency\.([^\\]+)\\", ("\packages\$dependency.$1" + $latestVersion + '\')
    Set-Content -Path $projectPath -Value $projectContent -NoNewline

    # Update packages.config
    $configPath = Join-Path (Split-Path $projectPath) 'packages.config'
    $configContent = [xml](Get-Content -Path $configPath -Encoding UTF8)
    $dependencyNodes = $configContent.SelectNodes("//package[contains(@id, '$dependency')]")
    foreach ($dependencyNode in $dependencyNodes) {
      $dependencyNode.version = $latestVersion
    }
    $configContent.Save($configPath)
  }

  #Update nuspecs
  foreach ($nuspec in $nuspecs) {
    $nuspecContent = [xml]::new()
    $nuspecContent.PreserveWhitespace = $true
    $nuspecContent.Load($nuspec)
    
    $dependencyNodes = $nuspecContent.package.metadata.dependencies.SelectNodes("//*[contains(@id, '$dependency')]")
    foreach ($dependencyNode in $dependencyNodes) {
      $dependencyNode.version = $latestVersion
    }
    $nuspecContent.Save($nuspec)
  }
}