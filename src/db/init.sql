CREATE TABLE IF NOT EXISTS data (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    sender TEXT,
    question TEXT,
    response TEXT,
    duration REAL
);

INSERT INTO data (key, value)
VALUES ('greeting', 'Hola, soy el bot de respuestas automáticas.')
ON CONFLICT(key) DO UPDATE SET value = excluded.value;
