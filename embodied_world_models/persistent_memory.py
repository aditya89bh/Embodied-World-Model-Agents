import json
from pathlib import Path


class PersistentMemoryStore:
    def __init__(self, path='memory/world_memory.json'):
        self.path = Path(path)

    def save(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open('w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2, sort_keys=True)

    def load(self):
        if not self.path.exists():
            return None

        with self.path.open('r', encoding='utf-8') as file:
            return json.load(file)

    def exists(self):
        return self.path.exists()
