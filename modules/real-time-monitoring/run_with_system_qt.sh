#!/bin/bash
# Wrapper script to run Python with system Qt libraries

# Unset problematic environment variables
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH

# Find system Qt5 libraries
QT5_LIB_PATH=$(find /usr/lib -name "libQt5Core.so.5" 2>/dev/null | head -1 | xargs dirname)

if [ -z "$QT5_LIB_PATH" ]; then
    # Try alternative paths
    QT5_LIB_PATH=$(find /usr/lib/x86_64-linux-gnu -name "libQt5Core.so.5" 2>/dev/null | head -1 | xargs dirname)
fi

if [ ! -z "$QT5_LIB_PATH" ]; then
    echo "Using system Qt libraries from: $QT5_LIB_PATH"
    export LD_LIBRARY_PATH="$QT5_LIB_PATH:$LD_LIBRARY_PATH"
fi

# Run the Python script
exec python3 "$@"