import appdirs
import os
import sys
import torch


class QMult():
    def __init__(self, dataset: str, device: str):
        self.device = device
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../convhyper'))
        from util.data import Data
        from util.helper_classes import Reproduce

        convhyper_data_dir = appdirs.user_data_dir('ConvHyper')
        model_path = os.path.join(convhyper_data_dir, 'PretrainedModels', dataset.upper(), 'QMultBatch')
        data_dir = os.path.join(convhyper_data_dir, 'KGs', dataset, '')
        print(f'{model_path=}')
        print(f'{data_dir=}')
        self.dataset = Data(data_dir=data_dir, train_plus_valid=False, reverse=False, tail_pred_constraint=False, out_of_vocab_flag=False)
        self.model = Reproduce().load_model(model_path=model_path, model_name='QMult').to(device)
        self.entity_idxs = {self.dataset.entities[i]: i for i in range(len(self.dataset.entities))}
        self.relation_idxs = {self.dataset.relations[i]: i for i in range(len(self.dataset.relations))}

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def encode(self, data):
        # assert data.shape[1] == 3
        return torch.tensor([(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))], device=self.device)

    def ent_emb(self, vec):
        return torch.cat((self.model.emb_ent_real(vec), self.model.emb_ent_i(vec), self.model.emb_ent_j(vec), self.model.emb_ent_k(vec)), 1)

    def rel_emb(self, vec):
        return torch.cat((self.model.emb_rel_real(vec), self.model.emb_rel_i(vec), self.model.emb_rel_j(vec), self.model.emb_rel_k(vec)), 1)


def load_quate_idxs(file):
    with open(file) as fd:
        i = iter(fd)
        next(i)  # header with the amount of lines
        return {k: int(v) for k, v in (line.rstrip().split()[0:2] for line in fd)}


class QuatWrapper():
    def __init__(self, dataset: str, device: str):
        dimension = 100

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../quate'))
        from config import Config
        from models.QuatE import QuatE

        path = os.path.join(os.path.dirname(__file__), '../embeddings', 'quate_' + dataset.replace('-', ''), 'QuatE.ckpt')
        in_path = os.path.join(os.path.dirname(__file__), '../quate/benchmarks', dataset.upper().replace('-', '')) + '/'
        print(f'{path=}')
        print(f'{in_path=}')

        # Most of that needs to be set but does not matter in this task.
        con = Config()
        con.set_in_path(in_path)
        con.set_work_threads(8)
        con.set_nbatches(10)
        con.set_alpha(0.1)
        con.set_bern(1)
        con.set_dimension(dimension)
        con.set_margin(1.0)
        con.set_rel_neg_rate(0)
        con.set_opt_method("adagrad")
        con.set_save_steps(1000)
        con.set_valid_steps(1000)
        con.set_early_stopping_patience(10)
        con.set_checkpoint_dir(None)
        con.set_result_dir(None)
        con.set_test_link(True)
        con.set_test_triple(True)
        con.init()

        self.model = QuatE(config=con)
        self.model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.entity_idxs = load_quate_idxs(os.path.join(os.path.dirname(__file__), '../quate/benchmarks', dataset.upper().replace('-', ''), 'entity2id.txt'))
        self.relation_idxs = load_quate_idxs(os.path.join(os.path.dirname(__file__), '../quate/benchmarks', dataset.upper().replace('-', ''), 'relation2id.txt'))

    def encode(self, data):
        # assert data.shape[1] == 3
        return torch.tensor([(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))])

    def ent_emb(self, x):
        return torch.stack([getattr(self.model, f'emb_{i}_a')(x) for i in 'sxyz']).permute((1, 0, 2)).flatten(start_dim=1, end_dim=-1).detach()

    def rel_emb(self, x):
        return torch.stack([getattr(self.model, f'rel_{i}_b')(x) for i in 'sxyz']).permute((1, 0, 2)).flatten(start_dim=1, end_dim=-1).detach()


class SedWrapper():
    def __init__(self, dataset: str, device: str):
        dimension = 25

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../quate'))
        from config import Config
        from models.SedeniE import SedeniE

        path = os.path.join(os.path.dirname(__file__), '../embeddings/sed_' + dataset + '/SedeniE.ckpt')

        # Most of that needs to be set but does not matter in this task.
        con = Config()
        con.set_in_path(os.path.join(os.path.dirname(__file__), '../quate/benchmarks/' + dataset.upper().replace('-', '') + '/'))
        con.set_work_threads(8)
        con.set_nbatches(10)
        con.set_alpha(0.1)
        con.set_bern(1)
        con.set_dimension(dimension)
        con.set_margin(1.0)
        con.set_rel_neg_rate(0)
        con.set_opt_method("adagrad")
        con.set_save_steps(1000)
        con.set_valid_steps(1000)
        con.set_early_stopping_patience(10)
        con.set_checkpoint_dir(None)
        con.set_result_dir(None)
        con.set_test_link(True)
        con.set_test_triple(True)
        con.init()

        self.model = SedeniE(config=con)
        self.model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.entity_idxs = load_quate_idxs(os.path.join(os.path.dirname(__file__), '../quate/benchmarks/' + dataset.upper().replace('-', '') + '/entity2id.txt'))
        self.relation_idxs = load_quate_idxs(os.path.join(os.path.dirname(__file__), '../quate/benchmarks/' + dataset.upper().replace('-', '') + '/relation2id.txt'))

    def encode(self, data):
        # assert data.shape[1] == 3
        return torch.tensor([(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))])

    def ent_emb(self, vec):
        return self.model.embed_entities(vec).permute((1, 0, 2)).flatten(start_dim=1, end_dim=-1).detach()

    def rel_emb(self, vec):
        return self.model.embed_relations(vec).permute((1, 0, 2)).flatten(start_dim=1, end_dim=-1).detach()


classes = {
    'QMult': QMult,
    'QuatE': QuatWrapper,
    'SedeniE': SedWrapper,
}


def load_embedding_model(model_name, dataset, device):
    return classes[model_name](dataset, device)
