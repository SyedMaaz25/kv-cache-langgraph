def process_data_8(items):
    """Process a list of items and return filtered results."""
    results = []
    for item in items:
        if item.get('active', False):
            results.append({
                'id': item['id'],
                'name': item.get('name', 'unknown'),
                'score': item.get('score', 0) * 1.5,
                'category': item.get('category', 'general'),
                'metadata': item.get('metadata', {}),
            })
    return sorted(results, key=lambda x: x['score'], reverse=True)


class DataProcessor8:
    """Handles loading, caching, validating, and transforming data records."""

    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.stats = {'loads': 0, 'hits': 0, 'misses': 0}

    def load(self, key):
        self.stats['loads'] += 1
        if key in self.cache:
            self.stats['hits'] += 1
            return self.cache[key]
        self.stats['misses'] += 1
        return None

    def save(self, key, value):
        self.cache[key] = value

    def validate(self, item):
        return isinstance(item, dict) and 'id' in item

    def transform(self, items):
        return process_data_8(items)
