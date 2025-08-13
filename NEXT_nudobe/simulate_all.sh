#!/bin/bash

START_TIME=$(date +%s)

echo "simulating 1bar leptoquark events, moving to 0.1% diffusion folder"
source simulate_events.sh -p 1 -d 0.1

echo "simulating 1bar leptoquark events, moving to 0.05% diffusion folder"
source simulate_events.sh -p 1 -d 0.05

echo "simulating 1bar leptoquark events, moving to 0.25% diffusion folder"
source simulate_events.sh -p 1 -d 0.25

echo "simulating 1bar leptoquark events, moving to 5% diffusion folder"
source simulate_events.sh -p 1 -d 5

echo "simulating 5bar leptoquark events, moving to 0.1% diffusion folder"
source simulate_events.sh -p 5 -d 0.1

echo "simulating 5bar leptoquark events, moving to 0.05% diffusion folder"
source simulate_events.sh -p 5 -d 0.05

echo "simulating 5bar leptoquark events, moving to 0.25% diffusion folder"
source simulate_events.sh -p 5 -d 0.25

echo "simulating 5bar leptoquark events, moving to 5% diffusion folder"
source simulate_events.sh -p 5 -d 5

END_TIME=$(date +%s)

TOTAL_TIME=$((END_TIME - $START_TIME))

echo "completed in $TOTAL_TIME seconds."