#!/usr/bin/env bash
spark-submit --master local[*] --total-executor-cores 14 --executor-memory 6g reco_server.py