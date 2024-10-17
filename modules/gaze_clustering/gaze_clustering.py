""" Gaze clustering

Author:
    Tobias Hallmen <tobias.hallmen@uni-a.de>
Date:
    17.10.2024

"""
import numpy as np
from finch import FINCH
import pandas as pd
from discover_utils.interfaces.server_module import Processor

INPUT_ID = "openface"
OUTPUT_ID = "gaze"

# Setting defaults
_default_options = {"metric": "l1", 'window': 9, 'cluster': 20, 'threshold': 0.7, 'greedy': 'distance'}

# TODO: allow for window to enter seconds and translate into frames
# TODO: compute confidence by measuring distance to cluster centroids
class GazeCluster(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        
        # string treatment for number extraction
        try:
            self.options['window'] = int(self.options['window'])
        except ValueError:
            print(f'window field contains invalid characters - using default {_default_options["window"]}')
            self.options['window'] = _default_options['window']
        try:
            self.options['cluster'] = int(self.options['cluster'])
        except ValueError:
            print(f'cluster field contains invalid characters - using default {_default_options["cluster"]}')
            self.options['cluster'] = _default_options['cluster']
        try:
            self.options['threshold'] = float(self.options['threshold'])
        except ValueError:
            print(f'threshold field contains invalid characters - using default {_default_options["threshold"]}')
            self.options['threshold'] = _default_options['threshold']

        self.distance = {'l1': lambda x, y: np.linalg.norm(x-y, 1), 'l2': lambda x, y: np.linalg.norm(x-y, 2),
                         'cosine': lambda x, y: 1 - np.inner(x, y)/(np.linalg.norm(x, 2) * np.linalg.norm(y, 2))}
        self.classname = 'DIRECTED'
        self.session_manager = None
    
    def to_output(self, data: pd.DataFrame):
        stride = 1000.0 / self.session_manager.input_data[INPUT_ID].meta_data.sample_rate
        anno_data = []
        k_old, v_old = 0, data['label'].values[0]
        for k, v in enumerate(data['label'].values[1:]):
            if v != v_old:
                anno_data.append((round(k_old * stride, 3), round(k * stride, 3), v_old, 1))
                k_old = k
                v_old = v
        anno_data.append((round(k_old * stride, 3), round(k * stride, 3), v_old, 1))
        
        self.session_manager.output_data_templates[OUTPUT_ID].data = anno_data
        return self.session_manager.output_data_templates

    def process_data(self, ds_manager) -> dict:
        self.session_manager = self.get_session_manager(ds_manager)

        # openface2 dimensions gaze angle x and y : 634, 635
        data = self.session_manager.input_data[INPUT_ID].data[:, 634:636]
        df = pd.DataFrame(data)
        df = df.rolling(int(self.options['window']), min_periods=1).median()
        c, n, p = FINCH(df, req_clust=int(self.options['cluster']), distance=self.options['metric'], verbose=False)
        df['label'] = p

        abc = df.label.value_counts(normalize=True, sort=True)
        if self.options['greedy'] == 'mass':
            mass = 0
            labels = []
            for l in zip(abc.index, abc):
                if mass < float(self.options['threshold']):
                    labels.append(l[0])
                    mass += l[1]
                else:
                    break
            df.label = df.label.apply(lambda x: 0 if x in labels else 1)
        elif self.options['greedy'] == 'distance':
            while abc[abc.index[0]] < float(self.options['threshold']):
                centroids = {lab: df[df.label == lab].iloc[:, :2].mean().values for lab in abc.index}
                biggest_centroid = centroids.pop(abc.index[0])
                distances = {k: self.distance[self.options['metric']](biggest_centroid, v) for k, v in centroids.items()}
                add_label = sorted(distances.items(), key=lambda x: x[1])[0][0]
                df.label = df.label.apply(lambda x: abc.index[0] if x == add_label else x)
                abc = df.label.value_counts(normalize=True, sort=True)
            df.label = df.label.apply(lambda x: 0 if x == abc.index[0] else 1)
        else:
            raise ValueError(f'Wrong greedy-option transmitted: {self.options["greedy"]}')

        print(f'{self.session_manager.input_data[INPUT_ID].meta_data.session}: {self.session_manager.input_data[INPUT_ID].meta_data.role}: {df.label.value_counts(normalize=True)[0]*100:.2f}% classified as {self.classname}.')
        return df
