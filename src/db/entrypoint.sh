#!/bin/sh
echo "Initialiting database ... or not"
sqlite3 /data/db.sqlite < /app/init.sql
tail -f /dev/null
