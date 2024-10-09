from dtlpy.miscellaneous import Zipping
from urllib.request import urlretrieve
import tempfile
import pathlib
import json
import os

import numpy as np
import dtlpy as dl
import pandas as pd

LOSS = (np.array([[0.9, 0.8, 0.75, 0.72, 0.7, 0.68, 0.66, 0.65, 0.63, 0.62],
                  [0.85, 0.78, 0.72, 0.7, 0.68, 0.65, 0.63, 0.61, 0.6, 0.58],
                  [0.8, 0.75, 0.7, 0.68, 0.65, 0.62, 0.6, 0.58, 0.57, 0.55]]) +
        np.random.normal(0, 0.01, (3, 10)))
ACCURACY = (np.array([[0.6, 0.65, 0.68, 0.7, 0.72, 0.74, 0.75, 0.76, 0.78, 0.79],
                      [0.65, 0.7, 0.72, 0.74, 0.75, 0.76, 0.78, 0.79, 0.8, 0.82],
                      [0.7, 0.73, 0.75, 0.76, 0.78, 0.79, 0.8, 0.82, 0.83, 0.85]]) +
            np.random.normal(0, 0.01, (3, 10)))


class Loader:
    def __init__(self):
        self.tmp_path = None

    def load_unannotated(self, dataset: dl.Dataset, source: str, progress: dl.Progress = None):
        self.upload_data(dataset=dataset, source=source, progress=progress)

    def load_annotated(self, dataset: dl.Dataset, source: str, progress: dl.Progress = None):
        # upload data
        self.upload_data(dataset=dataset, source=source, progress=progress)
        weight_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-agriculture/models.zip'
        with tempfile.TemporaryDirectory() as temp_dir:
            if progress:
                progress.update(message="Preparing data")
            # Downloading
            tmp_zip_path = os.path.join(temp_dir, 'models.zip')
            urlretrieve(weight_url, tmp_zip_path)
            models_dir = os.path.join(temp_dir, 'models')

            if progress:
                progress.update(message="Creating models")
            Zipping.unzip_directory(zip_filename=tmp_zip_path,
                                    to_directory=models_dir)
            self.clone_models(dataset=dataset,
                              metrics_path=os.path.join(models_dir, 'metrics'),
                              weight_filepath=os.path.join(models_dir, 'best.pth'),
                              progress=progress)

    def upload_data(self, source: str, dataset: dl.Dataset, progress: dl.Progress):
        with tempfile.TemporaryDirectory() as temp_dir:
            if progress:
                progress.update(message="Preparing data")
            # Downloading
            tmp_zip_path = os.path.join(temp_dir, 'data.zip')
            urlretrieve(source, tmp_zip_path)
            # Unzip
            data_dir = os.path.join(temp_dir, 'data')
            Zipping.unzip_directory(zip_filename=tmp_zip_path,
                                    to_directory=data_dir)
            if progress:
                progress.update(message="Uploading dataset")
            self.upload_dataset(dataset=dataset,
                                data_path=data_dir,
                                progress=progress)

    @staticmethod
    def upload_dataset(data_path, dataset: dl.Dataset, progress: dl.Progress):
        ontology_json_folder_path = os.path.join(data_path, 'ontology')
        items_folder_path = os.path.join(data_path, 'items')
        annotation_jsons_folder_path = os.path.join(data_path, 'json')

        # Upload ontology if exists
        if os.path.exists(ontology_json_folder_path) is True:
            ontology_json_filepath = list(pathlib.Path(ontology_json_folder_path).rglob('*.json'))[0]
            with open(ontology_json_filepath, 'r') as f:
                ontology_json = json.load(f)
            ontology: dl.Ontology = dataset.ontologies.list()[0]
            ontology.copy_from(ontology_json=ontology_json)
        item_binaries = sorted(list(filter(lambda x: x.is_file(), pathlib.Path(items_folder_path).rglob('*'))))
        annotation_jsons = sorted(list(pathlib.Path(annotation_jsons_folder_path).rglob('*.json')))

        # Validations
        if len(item_binaries) != len(annotation_jsons):
            annotation_jsons = [""] * len(item_binaries)

        uploads = list()
        for item_file, annotation_file in zip(item_binaries, annotation_jsons):
            # Construct item remote path
            remote_path = f"/{item_file.parent.stem}"

            # Upload with annotations
            if os.path.isfile(annotation_file):
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)

                # Extract tags
                item_metadata = dict()
                tags_metadata = annotation_data.get("metadata", dict()).get("system", dict()).get('tags', None)
                if tags_metadata is not None:
                    item_metadata.update({"system": {"tags": tags_metadata}})

                uploads.append(dict(local_path=str(item_file),
                                    local_annotations_path=str(annotation_file),
                                    remote_path=remote_path,
                                    item_metadata=item_metadata))
            # Upload without annotations
            else:
                uploads.append(dict(local_path=str(item_file),
                                    remote_path=remote_path))

        # Upload
        progress_tracker = {'last_progress': 0}

        def progress_callback(**kwargs):
            p = kwargs.get('progress')  # p is between 0-100
            progress_int = round(p / 10) * 10  # round to 10th
            if progress_int % 10 == 0 and progress_int != progress_tracker['last_progress']:
                progress.update(progress=80 * progress_int / 100)
                progress_tracker['last_progress'] = progress_int

        dl.client_api.callbacks.add(event='itemUpload', func=progress_callback)
        dataset.items.upload(local_path=pd.DataFrame(uploads))
        return dataset

    @staticmethod
    def clone_models(dataset: dl.Dataset, metrics_path: str, weight_filepath: str, progress: dl.Progress):
        filters = dl.Filters(field='app.dpkName', values="resnet", resource=dl.FILTERS_RESOURCE_MODEL)
        filters.add(field='status', values='pre-trained')
        pages = dataset.project.models.list(filters)
        if pages.items_count == 0:
            raise ValueError("Couldn't find a pretrained model found for 'resnet' app")
        pretrained_model = pages.items[0]
        for i_model in range(3):
            model: dl.Model = pretrained_model.clone(model_name=f"agri-classification-v{i_model + 1}",
                                                     dataset=dataset)
            model.status = "trained"
            filename = os.path.basename(weight_filepath)
            model.configuration['weights_filename'] = filename
            if 'evaluate' not in model.metadata['system']:
                model.metadata['system']['evaluate'] = dict()
            if 'datasets' not in model.metadata['system']['evaluate']:
                model.metadata['system']['evaluate']['datasets'] = list()
            model.metadata['system']['evaluate']['datasets'].append(dataset.id)
            model = model.update(system_metadata=True)
            model.artifacts.upload(filepath=weight_filepath)
            for i_metric in range(10):
                plots = [dl.PlotSample(figure='loss',
                                       legend='train',
                                       x=i_metric,
                                       y=float(LOSS[i_model, i_metric])),
                         dl.PlotSample(figure='accuracy',
                                       legend='train',
                                       x=i_metric,
                                       y=float(ACCURACY[i_model, i_metric])),
                         dl.PlotSample(figure='loss',
                                       legend='val',
                                       x=i_metric,
                                       y=float(LOSS[i_model, i_metric])),
                         dl.PlotSample(figure='accuracy',
                                       legend='val',
                                       x=i_metric,
                                       y=float(ACCURACY[i_model, i_metric]))
                         ]
                model.metrics.create(samples=plots,
                                     dataset_id=model.dataset_id)
            dataset.items.upload(local_path=os.path.join(metrics_path, f"{i_model}-interpolated.json"),
                                 remote_name=f"{model.id}-interpolated.json",
                                 remote_path=f"/.modelscores",
                                 overwrite=True)
            dataset.items.upload(local_path=os.path.join(metrics_path, f"{i_model}.csv"),
                                 remote_name=f"{model.id}.csv",
                                 remote_path=f"/.modelscores",
                                 overwrite=True)
            if progress:
                progress.update(progress=85 + (5 * (i_model + 1)))


if __name__ == "__main__":
    Loader().load_annotated(dl.datasets.get(dataset_id='66c63a7973198484a6e7cfa5'), source="")
