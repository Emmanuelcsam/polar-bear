#!/bin/bash
# Start the connector service

cd "$(dirname "$0")"
echo "Starting Analysis Connector..."
python connector.py