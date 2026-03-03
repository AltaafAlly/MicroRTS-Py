#!/usr/bin/env bash
# Build microrts.jar using only javac and jar (no Ant).
# Run from repo root:  bash gym_microrts/microrts/build_no_ant.sh
# Or from this dir:    bash build_no_ant.sh

set -e
cd "$(dirname "$0")"

echo "Building microrts.jar (no Ant)..."

rm -rf build fat_jar_temp microrts.jar
mkdir -p build fat_jar_temp

# List all Java sources (avoid passing as args to avoid "invalid flag" from paths)
find ./src ./test -name '*.java' 2>/dev/null | sort > sources.txt
n=$(wc -l < sources.txt)
echo "Compiling $n source files..."
javac -d build -cp "./lib/*" -sourcepath "src:test" "@sources.txt"
rm -f sources.txt

# Copy compiled classes into fat temp dir
cp -a build/. fat_jar_temp/

# Extract dependency JARs into fat temp (our classes take precedence)
for j in lib/*.jar; do
  [ -f "$j" ] || continue
  echo "Adding dependency: $j"
  unzip -o -q "$j" -d fat_jar_temp/ 2>/dev/null || true
done

# Optional: remove weka to keep JAR smaller
rm -f fat_jar_temp/weka.jar
rm -rf fat_jar_temp/bots

# Build single JAR with Main-Class for GUI
echo "Main-Class: gui.frontend.FrontEnd" > manifest.txt
jar cfm microrts.jar manifest.txt -C fat_jar_temp .
rm -f manifest.txt

# Cleanup
rm -rf build fat_jar_temp

echo "Done: microrts.jar ($(du -h microrts.jar | cut -f1))"
