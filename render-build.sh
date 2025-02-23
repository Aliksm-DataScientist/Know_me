#!/bin/bash

echo "Setting up environment..."

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt