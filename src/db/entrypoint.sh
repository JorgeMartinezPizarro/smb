#!/bin/sh
sqlite3 /data/db.sqlite < /init.sql
tail -f /dev/null
