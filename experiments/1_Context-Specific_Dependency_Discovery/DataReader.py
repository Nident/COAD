from pathlib import Path

import pandas as pd
import yaml


class DataReader:
    def __init__(self, config_path):
        config_path = Path(config_path)

        with config_path.open(encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.data_path = (config_path.parent / self.config["data"]["data_path"]).resolve()

    def read(self, table_name):
        file_name = self.config['data']["files"][table_name]
        return pd.read_csv(self.data_path / file_name, low_memory=False)

    def read_all(self):
        return {
            table_name: self.read(table_name)
            for table_name in self.config["data"]["files"]
        }
