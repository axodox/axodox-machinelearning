$dependencies = @("Axodox.Common")

$pluginProjects = Resolve-Path "./*/*.vcxproj"

foreach ($dependency in $dependencies) {
  $latestPackage = .\Tools\nuget.exe list -allversions $dependency
  $latestVersion = [regex]::Match($latestPackage, "\d+(\.\d+)+").Value

  foreach ($projectPath in $pluginProjects) {
    # Update project file
    $projectContent = Get-Content -Path $projectPath -Raw
    $projectContent = $projectContent -replace "\\packages\\$dependency\.([^\\]+)\\", ("\packages\$dependency.$1" + $latestVersion + '\')
    Set-Content -Path $projectPath -Value $projectContent -NoNewline

    # Update packages.config
    $configPath = Join-Path (Split-Path $projectPath) 'packages.config'
    $configContent = [xml](Get-Content -Path $configPath)
    $dependencyNodes = $configContent.SelectNodes("//package[contains(@id, '$dependency')]")
    foreach ($dependencyNode in $dependencyNodes) {
      $dependencyNode.version = $latestVersion
    }
    $configContent.Save($configPath)
  }
}