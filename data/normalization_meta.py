class NormalizationMetaData:
    def __init__(self):
        self.data = {}

    def set_col_metadata(self, col_name, col_data):
        self.data[col_name] = col_data
