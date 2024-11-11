""" Diarisation

Author:
    Tobias Hallmen <tobias.hallmen@uni-a.de>,
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    05.12.2023

"""

import os
import discover_utils
from discover_utils.interfaces.server_module import Processor
from pathlib import Path

class Diarisation(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = None
        self.ds_manager = None
        self.current_session = None
        self.options = {'metric': 'cosine', 'roles': None, 'role_samples': None, 'speaker_embedding': 'speechbrain',
                        'method': 'finch'} | self.options
        if self.options['role_samples'] == '':
            self.options['role_samples'] = None

    def to_output(self, data):
        import copy
        annos = {}
        # (from, to, id/name, conf) ; id = 0 = SPEECH ; name = free text
        dd = self.current_session.data_description
        out_idx = [i for i,x in enumerate(dd) if  x['id'] == 'out'][0]
        dd_out = dd[out_idx]
        for k, v in data.items():
            anno_data = [(x['from']*1000, x['to']*1000, x['name'], x['conf']) for x in v]

            anno = copy.deepcopy(self.current_session.output_data_templates['out'])
            #anno.meta_data.role = k
            anno.data = anno_data
            dd_out_copy = copy.deepcopy(dd_out)
            dd_out_copy['role'] = k
            dd_out_copy['id'] = f'out_{k}'

            # if out put is file type
            if discover_utils.utils.request_utils.Origin.FILE == origin:
                orig_uri = Path(dd_out_copy['uri'])
                dd_out_copy['uri'] = orig_uri.with_name(f'{orig_uri.stem}_{k}{orig_uri.suffix}')

            dd.append(dd_out_copy)

            # TODO: remove workaround for empty annos once data structure is updated
            if anno.data is None:
                anno.data = []

            annos[dd_out_copy['id']] = anno
        
        return annos

    def process_data(self, ds_manager) -> dict:
        self.ds_manager = ds_manager
        self.current_session = self.get_session_manager(ds_manager)

        # TODO: word aligned diarisation is underwhelming;
        #  combine it with sentence-based embeddings or other modalities, eg video: open mouth
        print('diarising')
        import numpy as np
        import torch
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.cuda.empty_cache()
        from sklearn.cluster import AgglomerativeClustering
        from pyannote.audio import Audio
        from pyannote.core import Segment
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
        from pyannote.audio.pipelines.clustering import AgglomerativeClustering as pyannAC
        from pathlib import Path
        
        # valid other models
        models = {'pyannote': "pyannote/embedding", 'speechbrain': "speechbrain/spkrec-ecapa-voxceleb",
                  'titanet': "nvidia/speakerverification_en_titanet_large",
                  'wespeaker': str(Path(os.environ["CACHE_DIR"]) / "diarisation" / 'wespeaker.en.onnx')}
        
        # note: speechbrain uses symlinks, which require on windows admin privileges.
        # but as admin in windows, you don't see network mapped drives per default
        # fix: run server on linux or do this
        # https://learn.microsoft.com/en-us/troubleshoot/windows-client/networking/mapped-drives-not-available-from-elevated-command

        # model input min duration:
        # pyannote:     4771/16000 = 0.2981875s -> forget word alignment
        # speechbrain:   640/16000 = 0.04s
        # nvidia:       unknown
        # wespeaker:    1680/16000 = 0.105s

        # TODO: these models are downloaded probably to pyannote (PYANNOTE_CACHE) or hf cache (HF_HOME);
        #  use nova server tmp dir; remove personal hf-token
        os.environ['TORCH_HOME'] = str((Path(os.environ['CACHE_DIR']) / 'torch').resolve())
        os.environ['HF_HOME'] = str((Path(os.environ['CACHE_DIR']) / 'huggingface').resolve())
        os.environ['PYANNOTE_CACHE'] = str((Path(os.environ['CACHE_DIR']) / 'pyannote').resolve())

        # define distance/confidence functions; numpy broadcasting does not work here
        def softmax(x):
            return 1 - np.exp(x) / np.reshape(np.repeat(np.sum(np.exp(x), axis=-1), x.shape[1], axis=-1), x.shape)
        
        if self.options['metric'] == 'cosine':
            def dist(x, y):
                return np.stack([1 - np.inner(x, z) / (np.linalg.norm(x, 2, axis=-1) * np.linalg.norm(z, 2, axis=-1)) for z in y], axis=-1)
        elif self.options['metric'] == 'euclidean':
            def dist(x, y):
                return np.stack([np.linalg.norm(x - z, 2, axis=-1) for z in y], axis=-1)
        elif self.options['metric'] == 'manhattan':
            def dist(x, y):
                return np.stack([np.linalg.norm(x - z, 1, axis=-1) for z in y], axis=-1)
        else:
            raise ValueError(f'unknown metric specified: {self.options["metric"]}')

        #TODO: remove hf-token
        if self.options['speaker_embedding'] == 'pyannote':
            spkr_embed_model = PretrainedSpeakerEmbedding(models[self.options['speaker_embedding']], device=self.device,
                                                          use_auth_token='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')

        elif self.options['speaker_embedding'] == 'wespeaker':
            if not (path := Path(models[self.options['speaker_embedding']])).exists():
                from discover_utils.utils.cache_utils import retreive_from_url
                url = 'https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx'
                path.parent.mkdir(exist_ok=True)
                retreive_from_url(url, path)
            spkr_embed_model = PretrainedSpeakerEmbedding(models[self.options['speaker_embedding']], device=self.device)
            
            #TODO: add once Apple's CoreML is fully supported
            # current error: [W:onnxruntime:, helper.cc:66 IsInputSupported] Dynamic shape is not supported for now
            #import onnxruntime as rt
            #spkr_embed_model.session_ = rt.InferenceSession(models[self.options['speaker_embedding']],
            #                                                sess_options=spkr_embed_model.session_._sess_options,
            #                                                providers=['CoreMLExecutionProvider'])

        elif self.options['speaker_embedding'] == 'speechbrain':
            spkr_embed_model = PretrainedSpeakerEmbedding(models[self.options['speaker_embedding']], device=self.device)
        
        # TODO: nvidia: unresolvable dependencies - not working
        elif self.options['speaker_embedding'] == 'titanet':
            # nemo-toolkit relies on deprecated pytorch_lightning.TrainingEpochLoop ... ModuleNotFoundError
            raise NotImplementedError()
            spkr_embed_model = PretrainedSpeakerEmbedding(models[self.options['speaker_embedding']], device=self.device)
        else:
            raise ValueError(f'unknown embedding model specified: {self.options["speaker_embedding"]}')

        roles = self.options['roles'].split(',')
        if (l := len(roles)) < 2:
            raise ValueError(f'only {l} roles specified - nothing to diarise')

        # get anno and convert to dictionary with seconds instead of milliseconds
        sentences = self.current_session.input_data['in'].data
        sentences = sentences.astype(np.dtype({'names': sentences.dtype.names, 'formats': (float, float, np.object_, float)}))
        sentences['from'] = sentences['from']/1000
        sentences['to'] = sentences['to']/1000
        sentences = [dict(zip(sentences.dtype.names, x)) for x in sentences]

        if len(sentences) < 1:
            raise ValueError('selected annotation is empty')

        audio = Audio(sample_rate=spkr_embed_model.sample_rate, mono='downmix')
        audio_file = self.current_session.input_data['audio'].meta_data.file_path
        # audio_duration = audio.get_duration(audio_file)

        if self.options['role_samples'] is not None:
            intervals = [[float(x) for x in tup.split(',')] for tup in
                         self.options['role_samples'].replace('),(', ';').replace('(', '').replace(')', '').split(';')]
            assert len(roles) == len(intervals), 'different amount of roles and samples given'

        spkr_embeds = None
        eps_start = 1e-2
        eps = eps_start
        min_dur = spkr_embed_model.min_num_samples / spkr_embed_model.sample_rate
        # assuming it is only one single file/path
        for i, sent in enumerate(sentences):
            # pyannote crashes immediately if segments are too short, enlarge them; might still return nans...
            if sent['to'] - sent['from'] < min_dur:
                diff = min_dur - (sent['to'] - sent['from'])
                sta = sent['from'] - diff / 2 - eps
                sto = sent['to'] + diff / 2 + eps
                data, sr = audio.crop(audio_file, Segment(sta, sto), mode='pad')
            else:
                data, sr = audio.crop(audio_file, Segment(sent['from'], sent['to']), mode='pad')

            if spkr_embeds is None:
                spkr_embeds = np.zeros((len(sentences), (t := spkr_embed_model(data[np.newaxis])).shape[1]))
                spkr_embeds[0] = t
            else:
                spkr_embeds[i] = spkr_embed_model(data[np.newaxis])  # add batch axis

            # there might still be nans in pyannote despite fulfilling minimum duration; extending segment until embedding not nan
            pr = False
            while np.isnan(spkr_embeds[i]).any():
                eps *= 2
                data, sr = audio.crop(audio_file, Segment(sent['from'] - eps, sent['to'] + eps), mode='pad')
                spkr_embeds[i] = spkr_embed_model(data[np.newaxis])
                pr = True
            if pr:
                print(eps)
            eps = eps_start

        # pyannote embeddings unreliable; debug info
        if np.isnan(spkr_embeds).any():
            indices = set(np.where(np.isnan(spkr_embeds))[0])
        
        diarisation = {x: [] for x in roles}

        # use given speaker samples to diarise
        if self.options['role_samples'] is not None:
            sample_embeds = []
            for sta, sto in intervals:
                data, sr = audio.crop(audio_file, Segment(sta, sto), mode='pad')
                sample_embeds.append(spkr_embed_model(data[np.newaxis]))
            
            sample_embeds = list(np.array(sample_embeds).squeeze())
            
            centroids = sample_embeds
            distances = dist(spkr_embeds, centroids)
            confidences = softmax(distances)
            labels = np.argmax(confidences, axis=-1)
            
            for x in roles:
                diarisation[x] = [sentences[i] | {'conf': confidences[i][l]} for i, l in enumerate(labels)
                                  if roles[l] == x]

        # try agglomerative approach
        else:
            if self.options['method'] == 'agglomerative':
                # sklearn: pyannote uses cosine metric, but then "warding" doesn't work in sklearn
                if self.options['metric'] == 'euclidean':
                    spkr_cluster = AgglomerativeClustering(len(roles), metric=self.options['metric']).fit(spkr_embeds)
                else:
                    spkr_cluster = (AgglomerativeClustering(len(roles), metric=self.options['metric'], linkage='average')
                                    .fit(spkr_embeds))

                spkr_label_to_role = {l: r for l, r in zip(set(spkr_cluster.labels_), roles)}

                centroids = [spkr_embeds[np.where(spkr_cluster.labels_ == i)].mean(axis=0) for i in range(len(roles))]
                distances = dist(spkr_embeds, centroids)
                confidences = softmax(distances)
                
                for x in roles:
                    diarisation[x] = [sentences[i] | {'conf': confidences[i][l]} for i, l in enumerate(spkr_cluster.labels_) if spkr_label_to_role[l] == x]

            elif self.options['method'] == 'pyannote':
                # non-functional as of now
                raise NotImplementedError()
                # pyannote probably needs its own pyannote segmentation beforehand; author not helpful in git issues
                spkr_cluster = pyannAC(metric='cosine')
                spkr_cluster.instantiate({"method": "average", "threshold": 1.0, "min_cluster_size": 1})
                cluster, _ = spkr_cluster(spkr_embeds, min_clusters=1, max_clusters=np.inf, num_clusters=len(roles))

            elif self.options['method'] == 'finch':
                from finch import FINCH
                cluster_partition, n_part_clust, part_labels = FINCH(spkr_embeds, req_clust=len(roles),
                                                                     distance=self.options['metric'],
                                                                     verbose=False)
                    
                spkr_label_to_role = {l: r for l, r in zip(set(part_labels), roles)}

                centroids = [spkr_embeds[np.where(part_labels == i)].mean(axis=0) for i in range(len(roles))]
                distances = dist(spkr_embeds, centroids)
                confidences = softmax(distances)
                
                for x in roles:
                    diarisation[x] = [sentences[i] | {'conf': confidences[i][l]} for i, l in enumerate(part_labels) if spkr_label_to_role[l] == x]
            else:
                raise ValueError(f'unknown method {self.options["method"]} specified')
        
        print('assigned' + ','.join([f' {len(diarisation[x])} annos to {x}' for x in diarisation.keys()]))
        return diarisation
