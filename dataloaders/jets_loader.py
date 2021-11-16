import os
import uproot3 as uproot
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from datetime import datetime
from tqdm import tqdm

data_dir = 'data/'
node_features_list = ['trk_d0', 'trk_z0', 'trk_phi', 'trk_ctgtheta', 'trk_pt', 'trk_charge']
jet_features_list = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_M']

node_features_list_real_data = ['Track_dxy', 'Track_dz', 'Track_phi', 'Track_eta', 'Track_pt', 'Track_charge']
jet_features_list_real_data = ['Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass']

track_vertex_key = b'trk_vtx_index'
track_vertex_key_real_data = b'Jet_SV_multi'

jet_flav_key = b'jet_flav'
jet_flav_key_real_data = b'Jet_hadronFlavour'

def get_data_loader(which_set, batch_size, debug_load=False,add_jet_flav=False,add_rave_file=False,real_data=False):
    jets_data = JetGraphDataset(which_set, debug_load,add_jet_flav,add_rave_file,real_data)
    batch_sampler = JetsBatchSampler(jets_data.n_nodes, batch_size)
    data_loader = DataLoader(jets_data, batch_sampler=batch_sampler)

    return data_loader


def transform_features(transform_list, arr):
    new_arr = np.zeros_like(arr)
    for col_i, (mean, std) in enumerate(transform_list):
        new_arr[col_i,:] = (arr[col_i,:] - mean) / std
    return new_arr


class JetGraphDataset(Dataset):
    def __init__(self, which_set, debug_load=False,add_jet_flav=False,add_rave_file=False,real_data=False,correct_jet_flav=False):
        """
        Initialization
        :param which_set: either "train", "validation" or "test"
        :param debug_load: if True, will load only a small subset
        
        """
        self.add_jet_flav = add_jet_flav
        self.add_rave_file = add_rave_file
        self.return_rave_prediction = False
        self.real_data = real_data
        self.correct_jet_flav = correct_jet_flav

        assert which_set in ['train', 'validation', 'test']
        fname = {'train': 'training_data.root', 'validation': 'valid_data.root', 'test': 'test_data.root'}
        fname_rave = {'train': 'rave_training_data.root', 'validation': 'rave_validation_data.root', 'test': 'rave_test_data_large.root'}
        fname_real_data = {'train': 'training_real_data.root', 'validation': 'valid_real_data.root', 'test': 'test_real_data.root'}
        if real_data:
            print(" --> Initializing JetGraphDataset with real data")
            self.filename = os.path.join(data_dir, which_set, fname_real_data[which_set])
        else:
            print(" --> Initializing JetGraphDataset with MC data")
            self.filename = os.path.join(data_dir, which_set, fname[which_set])
        if self.add_jet_flav:
            print(" --> Using jet flavour")
        with uproot.open(self.filename) as f:
            if real_data:
                tree = f['btagana/ttree']
            else:
                tree = f['tree']
                self.n_jets = int(tree.numentries)
            if real_data:
                self.handle_real_data(tree)
            else:
                self.n_nodes = np.array([len(x) for x in tree.array(track_vertex_key)])
                self.jet_arrays = tree.arrays(jet_features_list + node_features_list + [track_vertex_key])

                self.sets, self.partitions, self.partitions_as_graphs = [], [], []

                if self.add_jet_flav:
                    if self.correct_jet_flav:
                        flav_dict = {5: 0, 4: 1, 0: 2}
                        self.jet_flavs = tree.array(jet_flav_key)
                        self.jet_arrays[b'jet_flav'] = [flav_dict[x] for x in self.jet_flavs]
                        self.jet_flavs = torch.LongTensor([flav_dict[x] for x in self.jet_flavs])
                        print(" --> Loaded corrected jet_flavs tensor", self.jet_flavs)
                    else:
                        self.jet_flavs = tree.array(jet_flav_key)
                        self.jet_arrays[b'jet_flav'] = self.jet_flavs
                        self.jet_flavs = torch.LongTensor(self.jet_flavs)
                        print(" --> Loaded jet_flavs tensor", self.jet_flavs)
             
        if self.add_rave_file:
            print("Using rave file")
            with uproot.open(os.path.join(data_dir, which_set, fname_rave[which_set])) as f:
                tree = f['rave_prediction']
                self.rave_track_vtx_assignment = tree.array('track_vtx_assignment')
                self.rave_edge_scores = []
            self.return_rave_prediction = True

        if debug_load:
            self.n_jets = 100
            self.n_nodes = self.n_nodes[:100]

        start_load = datetime.now()

        for set_, partition, partition_as_graph in self.get_all_items():
            if torch.cuda.is_available():
                set_ = torch.tensor(set_, dtype=torch.float, device='cuda')
                partition = torch.tensor(partition, dtype=torch.long, device='cuda')
                partition_as_graph = torch.tensor(partition_as_graph, dtype=torch.float, device='cuda')
            self.sets.append(set_)
            self.partitions.append(partition)
            self.partitions_as_graphs.append(partition_as_graph)

        self.only_jet_features = np.stack(self.only_jet_features)

        self.only_jet_features = torch.tensor(self.only_jet_features, dtype=torch.float)
        if torch.cuda.is_available():
            self.only_jet_features = self.only_jet_features.cuda()

        if not torch.cuda.is_available():
            self.sets = np.array(self.sets)
            self.partitions = np.array(self.partitions)
            self.partitions_as_graphs = np.array(self.partitions_as_graphs)

        print(f' {str(datetime.now() - start_load).split(".")[0]}', flush=True)
        


    def __len__(self):
        """Returns the length of the dataset"""
        return self.n_jets


    def handle_real_data(self, tree):

        print(" --> Handling real data")

        jet_pt = tree['Jet_pt'].array()
        jet_eta = tree['Jet_eta'].array()
        jet_phi = tree['Jet_phi'].array()
        jet_mass = tree['Jet_mass'].array()
        jet_ntracks = tree['Jet_ntracks'].array()
        jet_flav = tree['Jet_hadronFlavour'].array()

        trk_pt = tree['Track_pt'].array()
        trk_phi = tree['Track_phi'].array()
        trk_eta = tree['Track_eta'].array()
        trk_d0 = tree['Track_dxy'].array()
        trk_z0 = tree['Track_dz'].array()
        trk_charge = tree['Track_charge'].array()
        trk_PV = tree['Track_PV'].array()
        trk_SV = tree['Track_SV'].array()

        out_jet_pt = []
        out_jet_eta = []
        out_jet_phi = []
        out_jet_mass = []
        out_jet_flav = []
        out_jet_num_sv = []
        out_jet_num_pv = []
        out_jet_num_v = []

        out_trk_pt = []
        out_trk_eta = []
        out_trk_phi = []
        out_trk_d0 = []
        out_trk_z0 = []
        out_trk_charge = []
        out_trk_vtx_index = []

        i = 0
        outEveryX = 100

        for jet_pt_i, jet_eta_i, jet_phi_i, jet_mass_i, jet_ntracks_i, jet_flav_i, trk_pt_i, trk_eta_i, trk_phi_i, trk_d0_i, trk_z0_i, trk_charge_i, trk_PV_i, trk_SV_i in zip(jet_pt, jet_eta, jet_phi, jet_mass, jet_ntracks, jet_flav, trk_pt, trk_eta, trk_phi, trk_d0, trk_z0, trk_charge, trk_PV, trk_SV):
            begin = 0
            for jet_pt_ij, jet_eta_ij, jet_phi_ij, jet_mass_ij, jet_ntracks_ij, jet_flav_ij in zip(jet_pt_i, jet_eta_i, jet_phi_i, jet_mass_i, jet_ntracks_i, jet_flav_i):
                if jet_ntracks_ij > 0:
                    end = begin + jet_ntracks_ij
                    trk_PV_tmp = trk_PV_i[begin:end]
                    trk_SV_tmp = trk_SV_i[begin:end]
                    trk_pt_tmp = trk_pt_i[begin:end]
                    trk_eta_tmp = trk_eta_i[begin:end]
                    trk_phi_tmp = trk_phi_i[begin:end]
                    trk_d0_tmp = trk_d0_i[begin:end]
                    trk_z0_tmp = trk_z0_i[begin:end]
                    trk_charge_tmp = trk_charge_i[begin:end]



                    if len(trk_PV_tmp) != len(trk_SV_tmp):
                        print("!! FATALERROR !!")
                        return

                    trk_vtx_index_tmp = []
                    tmp_jet_npv = 0
                    tmp_jet_nsv = 0
                    max_pv_index = max(trk_PV_tmp)
                    removeAt = []
                    for j in range(len(trk_SV_tmp)):
                        trk_pv_index = trk_PV_tmp[j]
                        trk_sv_index = trk_SV_tmp[j]
                        if trk_pv_index >= 0:
                            trk_vtx_index_tmp.append(trk_pv_index)
                            tmp_jet_npv += 1
                        else:
                            if trk_sv_index >= 0:
                                trk_vtx_index_tmp.append(trk_sv_index + max_pv_index + 1)
                                tmp_jet_nsv += 1
                            else:
                                #print(" --> Found track without any vertex, removing...")
                                removeAt.append(j)

                    if len(trk_vtx_index_tmp) > 0:

                        trk_pt_tmp = np.delete(trk_pt_tmp, removeAt)
                        trk_eta_tmp = np.delete(trk_eta_tmp, removeAt)
                        trk_phi_tmp = np.delete(trk_phi_tmp, removeAt)
                        trk_d0_tmp = np.delete(trk_d0_tmp, removeAt)
                        trk_z0_tmp = np.delete(trk_z0_tmp, removeAt)
                        trk_charge_tmp = np.delete(trk_charge_tmp, removeAt)

                        if len(trk_vtx_index_tmp) != len(trk_pt_tmp) != len(trk_eta_tmp) != len(trk_phi_tmp) != len(
                                trk_d0_tmp) != len(trk_z0_tmp) != len(trk_charge_tmp):
                            print("!! FATAL ERROR !!")
                            return

                        out_trk_pt.append(np.array(trk_pt_tmp, dtype=np.float32))
                        out_trk_eta.append(np.array(trk_eta_tmp, dtype=np.float32))
                        out_trk_phi.append(np.array(trk_phi_tmp, dtype=np.float32))
                        out_trk_d0.append(np.array(trk_d0_tmp, dtype=np.float32))
                        out_trk_z0.append(np.array(trk_z0_tmp, dtype=np.float32))
                        out_trk_charge.append(np.array(trk_charge_tmp, dtype=np.float32))
                        out_trk_vtx_index.append(np.array(trk_vtx_index_tmp, dtype=np.float32))

                        out_jet_pt.append(jet_pt_ij)
                        out_jet_eta.append(jet_eta_ij)
                        out_jet_phi.append(jet_phi_ij)
                        out_jet_mass.append(jet_mass_ij)
                        out_jet_num_pv.append(tmp_jet_npv)
                        out_jet_num_sv.append(tmp_jet_nsv)
                        out_jet_num_v.append(tmp_jet_nsv+tmp_jet_npv)
                        if self.add_jet_flav:
                            #print("using jet flav")
                            out_jet_flav.append(jet_flav_ij)

                i += 1
                begin += jet_ntracks_ij




        print(" --> Loaded",len(out_jet_pt),"Jets with",len(out_trk_vtx_index),"vtx info arrays from real data")
        self.n_jets = len(out_jet_pt)



        """

        node_features_list = ['trk_d0', 'trk_z0', 'trk_phi', 'trk_ctgtheta', 'trk_pt', 'trk_charge']
        jet_features_list = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_M']

        node_features_list_real_data = ['Track_dxy', 'Track_dz', 'Track_phi', 'Track_eta', 'Track_pt', 'Track_charge']
        jet_features_list_real_data = ['Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass']

        """

        # self.jet_arrays = tree.arrays(jet_features_list + node_features_list + [track_vertex_key])

        out_jet_pt = np.array(out_jet_pt, dtype=np.float32)
        out_jet_eta = np.array(out_jet_eta, dtype=np.float32)
        out_jet_phi = np.array(out_jet_phi, dtype=np.float32)
        out_jet_mass = np.array(out_jet_mass, dtype=np.float32)
        out_jet_flav = np.array(out_jet_flav, dtype=np.float32)
        out_jet_num_v = np.array(out_jet_num_v, dtype=np.float32)
        out_jet_num_sv = np.array(out_jet_num_sv, dtype=np.float32)
        out_jet_num_pv = np.array(out_jet_num_pv, dtype=np.float32)

        out_trk_pt = np.array(out_trk_pt, dtype=np.object)
        out_trk_eta = np.array(out_trk_eta, dtype=np.object)
        out_trk_phi = np.array(out_trk_phi, dtype=np.object)
        out_trk_d0 = np.array(out_trk_d0, dtype=np.object)
        out_trk_z0 = np.array(out_trk_z0, dtype=np.object)
        out_trk_charge = np.array(out_trk_charge, dtype=np.object)
        out_trk_vtx_index = np.array(out_trk_vtx_index, dtype=np.object)

        self.n_nodes = np.array([len(x) for x in out_trk_vtx_index])
        self.jet_arrays = {
            b'jet_pt': out_jet_pt,
            b'jet_eta': out_jet_eta,
            b'jet_phi': out_jet_phi,
            b'jet_M': out_jet_mass,
            b'jet_npv': out_jet_num_pv,
            b'jet_nsv': out_jet_num_sv,
            b'jet_nv': out_jet_num_v,
            b'trk_d0': out_trk_d0,
            b'trk_z0': out_trk_z0,
            b'trk_phi': out_trk_phi,
            b'trk_ctgtheta': out_trk_eta,
            b'trk_pt': out_trk_pt,
            b'trk_charge': out_trk_charge,
            b'trk_vtx_index': out_trk_vtx_index,
        }

        if self.add_jet_flav:
            if self.correct_jet_flav:
                flav_dict = {5: 0, 4: 1, 0: 2}
                self.jet_flavs = out_jet_flav
                self.jet_arrays[b'jet_flav'] = [flav_dict[x] for x in self.jet_flavs]
                self.jet_flavs = torch.LongTensor([flav_dict[x] for x in self.jet_flavs])
                print(" --> Loaded corrected jet_flavs tensor", self.jet_flavs)
            else:
                self.jet_flavs = out_jet_flav
                self.jet_arrays[b'jet_flav'] = self.jet_flavs
                self.jet_flavs = torch.LongTensor(self.jet_flavs)
                print(" --> Loaded jet_flavs tensor", self.jet_flavs)


        self.sets, self.partitions, self.partitions_as_graphs = [], [], []

    def get_all_items(self):
        node_feats = np.array([np.asarray(self.jet_arrays[str.encode(x)]) for x in node_features_list])
        jet_feats = np.array([np.asarray(self.jet_arrays[str.encode(x)]) for x in jet_features_list])
        n_labels = np.array(self.jet_arrays[b'trk_vtx_index'], dtype=np.object)
        self.only_jet_features = []

        for i in tqdm(range(self.n_jets)):
            n_nodes = self.n_nodes[i]
            node_feats_i = np.stack(node_feats[:, i], axis=0)  # shape (6, n_nodes)
            jet_feats_i = jet_feats[:, i]  # shape (4, )
            jet_feats_i = jet_feats_i[:, np.newaxis]  # shape (4, 1)

            node_feats_i = transform_features(FeatureTransform.node_feature_transform_list, node_feats_i)
            jet_feats_i = transform_features(FeatureTransform.jet_features_transform_list, jet_feats_i)

            self.only_jet_features.append(jet_feats_i.flatten())
            jet_feats_i = np.repeat(jet_feats_i, n_nodes, axis=1)  # change shape to (4, n_nodes)
            set_i = np.concatenate([node_feats_i, jet_feats_i]).T  # shape (n_nodes, 10)

            partition_i = n_labels[i]

            sort_order = np.argsort(node_feats_i[4])
            set_i = set_i[sort_order]

            tile = np.tile(partition_i, (self.n_nodes[i], 1))
            partition_as_graph_i = np.where((tile - tile.T), 0, 1)

            if self.return_rave_prediction:
                rave_partition_i = np.array(self.rave_track_vtx_assignment[i])
                tile = np.tile(rave_partition_i, (self.n_nodes[i], 1))
                rave_partition = np.where((tile - tile.T), 0, 1)
                self.rave_edge_scores.append(torch.tensor(rave_partition, dtype=torch.float, device='cuda'))

            yield set_i, partition_i, partition_as_graph_i


    def __getitem__(self, idx):
        """Generates a single instance of data"""
 
        if self.add_jet_flav:
            if self.return_rave_prediction:
                return self.sets[idx], self.partitions[idx], self.partitions_as_graphs[idx], self.only_jet_features[idx], self.jet_flavs[idx], self.rave_edge_scores[idx]
            else:
                return self.sets[idx], self.partitions[idx], self.partitions_as_graphs[idx], self.only_jet_features[idx], self.jet_flavs[idx]
        else:
            return self.sets[idx], self.partitions[idx], self.partitions_as_graphs[idx]


class JetsBatchSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size):
        """
        Initialization
        :param n_nodes_array: array of sizes of the jets
        :param batch_size: batch size
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            if n_nodes_i <= 1:
                continue
            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = max(n_of_size / self.batch_size, 1)

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]


class FeatureTransform(object):
    # Based on mean and std values of TRAINING set only
    node_feature_transform_list = [
        (0.0006078152, 14.128961),
        (0.0038490593, 10.688491),
        (-0.0026713554, 1.8167108),
        (0.0047640945, 1.889725),
        (5.237357, 7.4841413),
        (-0.00015662189, 1.0)]

    jet_features_transform_list = [
        (75.95093, 49.134453),
        (0.0022607117, 1.2152709),
        (-0.0023569583, 1.8164033),
        (9.437994, 6.765137)]
