# Copyright (c) 2023, Tobias Hallmen
from nova_utils.interfaces.server_module import Predictor
from nova_utils.ssi_utils.ssi_anno_utils import Anno, Scheme, SchemeType

REQUIREMENTS = [
    'finch-clust',
    'pynndescent',
]

class Cluster(Predictor):
    def __init__(self, logger, request_form=None):
        super().__init__(logger, request_form)
        import numpy as np
        self.ds_iter = None
        self.ds_iter_list = None
        self.model = None
        self.logger = logger
        self.data = None
        self.predictions = None
        self.DEPENDENCIES = []
        self.options = {'metric': 'l2', 'window': 10, 'cluster': 10, 'threshold': 0.7, 'greedy': 'distance'} | self.options
        self.distance = {'l1': lambda x, y: np.linalg.norm(x-y, 1), 'l2': lambda x, y: np.linalg.norm(x-y, 2),
                         'cosine': lambda x, y: 1 - np.inner(x, y)/(np.linalg.norm(x, 2) * np.linalg.norm(y, 2))}
        self.request_form = request_form
        self.delete = []
        self.classname = 'DIRECTED'

    def preprocess_sample(self, sample):
        return

    def process_sample(self, sample):
        pass

    def postprocess_sample(self, sample):
        pass

    def to_anno(self, data):
        annos = []
        for stream_name, df in data.items():
            # append previous contiguous segment when detecting a class switch
            anno_data = []
            to_iter = list(df.label.to_dict().items())
            k_old, v_old = to_iter[0]
            for k, v in to_iter:
                if v != v_old:
                    anno_data.append((f'{k_old * self.ds_iter.stride_ms / 1000:.3f}', f'{k * self.ds_iter.stride_ms / 1000:.3f}', v_old, 1))
                    k_old = k
                    v_old = v
            anno_data.append((f'{k_old * self.ds_iter.stride_ms / 1000:.3f}', f'{k * self.ds_iter.stride_ms / 1000:.3f}', v, 1))
    
            # clean up garbage class
            anno_data = [x for x in anno_data if x[2] != 1]
            
            anno = Anno(
                    role=stream_name,
                    annotator=self.request_form["annotator"],
                    scheme=Scheme(name=self.request_form['scheme'], type=SchemeType.DISCRETE),
                    data=anno_data,
                )
            annos.append(anno)
        return annos

    def process_data(self, ds_iter) -> dict:
        from finch import FINCH
        import pandas as pd
        self.ds_iter = ds_iter
        self.ds_iter_list = list(self.ds_iter)
        data_info = self.ds_iter.data_info.items()
        dfs = {}
        for stream_name, data_object in data_info:
            # openface2 dimensions gaze angle x and y : 634, 635
            data = {i: x[stream_name][:, 634:636].squeeze() for i, x in enumerate(self.ds_iter_list)}
            df = pd.DataFrame.from_dict(data, orient='index')
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

            self.logger.info(f'{self.ds_iter.sessions[0]}: {stream_name.split(".")[0]}: {df.label.value_counts(normalize=True)[0]*100:.2f}% classified as {self.classname}.')
            dfs[stream_name.split(".")[0]] = df
        return dfs
