from dtlpy.miscellaneous import Zipping
from urllib.request import urlretrieve
import tempfile
import pathlib
import json
import os
import logging
import dtlpy as dl
import pandas as pd

logger = logging.getLogger(name='[Kaggle] V2 Plant Seedlings Dataset Raw')


class DatasetKaggle(dl.BaseServiceRunner):
    def __init__(self):
        dl.use_attributes_2(state=True)
        self.tmp_path = os.getcwd()

    def upload_dataset(self, dataset: dl.Dataset, source: str, progress: dl.Progress = None):
        data_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-kaggle/V2%20Plant%20Seedlings%20Dataset%20Raw%20(1).zip'
        with tempfile.TemporaryDirectory() as temp_dir:
            if progress:
                progress.update(message="Preparing data")
            # Downloading
            tmp_zip_path = os.path.join(temp_dir, 'data.zip')
            urlretrieve(data_url, tmp_zip_path)
            # Unzip
            data_dir = os.path.join(temp_dir, 'data')
            Zipping.unzip_directory(zip_filename=tmp_zip_path,
                                    to_directory=data_dir)
            if progress:
                progress.update(message="Uploading dataset")
            self.upload_dataset_items(
                dataset=dataset,
                data_path=data_dir,
                progress=progress
            )

    @staticmethod
    def upload_dataset_items(data_path, dataset: dl.Dataset, progress: dl.Progress = None):
        ontology_json_folder_path = os.path.join(data_path, 'ontology')
        items_folder_path = os.path.join(data_path, 'items')

        # Upload ontology if exists
        if os.path.exists(ontology_json_folder_path) is True:
            ontology_json_filepath = list(pathlib.Path(ontology_json_folder_path).rglob('*.json'))[0]
            with open(ontology_json_filepath, 'r') as f:
                ontology_json = json.load(f)
            ontology: dl.Ontology = dataset.ontologies.list()[0]
            ontology.copy_from(ontology_json=ontology_json)
        item_binaries = sorted(list(filter(lambda x: x.is_file(), pathlib.Path(items_folder_path).rglob('*'))))

        uploads = list()
        for item_file in item_binaries:
            # Construct item remote path
            remote_path = f"/{item_file.parent.stem}"
            uploads.append(dict(local_path=str(item_file),
                                remote_path=remote_path))

        # Upload
        progress_tracker = {'last_progress': 0}

        def progress_callback(**kwargs):
            p = kwargs.get('progress')  # p is between 0-100
            progress_int = round(p / 10) * 10  # round to 10th
            if progress_int % 10 == 0 and progress_int != progress_tracker['last_progress']:
                if progress is not None:
                    progress.update(progress=80 * progress_int / 100)
                progress_tracker['last_progress'] = progress_int

        dl.client_api.callbacks.add(event='itemUpload', func=progress_callback)
        dataset.items.upload(local_path=pd.DataFrame(uploads))
        return dataset


def test_dataset_import():
    dataset_id = "dataset_id"

    dataset = dl.datasets.get(dataset_id=dataset_id)
    sr = DatasetKaggle()
    sr.upload_dataset(dataset=dataset, source="")


def main():
    test_dataset_import()


if __name__ == '__main__':
    main()
