#!/bin/bash
# Wrapper around condor_submit that ensures required directories exist.
# Usage: bash submit.sh   (from ~/  on ap2001)

mkdir -p output snapshots logs
condor_submit inpainting.sub
