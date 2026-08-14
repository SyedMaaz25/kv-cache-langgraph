def process_data_4(items):
    """Process a list of items and return filtered results."""
    results = []
    for item in items:
        if item.get('active', False):
            results.append({
                'id': item['id'],
                'name': item.get('name', 'unknown'),
                'score': item.get('score', 0) * 1.5,
            })
    return sorted(results, key=lambda x: x['score'], reverse=True)


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}

    def load(self, path):
        if path in self.cache:
            return self.cache[path]
        with open(path) as f:
            data = f.read()
        self.cache[path] = data
        return data

    def validate(self, data):
        errors = []
        if not isinstance(data, dict):
            errors.append('data must be a dict')
        return errors
