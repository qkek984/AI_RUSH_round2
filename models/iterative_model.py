import torch
from torch import nn

class Iterative_Model(nn.Module):
    def __init__(self, model, mode='train', alpha=0, starting_epoch=2, sample_size=8):
        super(Iterative_Model, self).__init__()
        self.name = "Iterative Model"
        self.model = model
        self.prototypes = {i : [] for i in range(5)} 
        self.alpha = 0
        self.mode = mode
        self.starting_epoch = starting_epoch
        self.sample_size = sample_size

    def forward(self, x, epoch, onehot_feat=None):
        feats = self.model.feat_extract(x)
        
        if self.model.onehot:
            feats_w_cat = torch.cat([feats,onehot_feat], axis=1)
            out = self.model.fc(feats_w_cat)
        else:
            out = self.model.fc(feats)

        if self.mode == 'train' and epoch > self.starting_epoch:
            yhat = self.pseudo_labeling(feats)
            return out, yhat
        else:
            pred = torch.argmax(out, dim=-1)
            return out, pred

    def pseudo_labeling(self, feats):
        with torch.no_grad():
            for i in range(feats.size(0)):
                vector = feats[i,:]
                if i == 0:
                    pseudo_labels = self.vec2label(vector)
                else:
                    pseudo_labels = torch.cat([pseudo_labels,self.vec2label(vector)])
        return pseudo_labels

    def vec2label(self, vector):
        with torch.no_grad():
            best_score = torch.Tensor([0]).cuda()
            for cls_ in self.prototypes:
                score = torch.Tensor([0]).cuda()
                for i in range(self.prototypes[cls_].size(0)):
                    prototype = self.prototypes[cls_][i,:]
                    score += torch.sum(vector * prototype) / (torch.sum(vector **2) * torch.sum(prototype **2)) ** 0.5
                score = score / self.sample_size
                if score > best_score:
                    best_score = score
                    final_cls = cls_
            
        return torch.Tensor([final_cls]).long().cuda()

    def prototype_update(self, class_samples, device):
        with torch.no_grad():
            cls_feats = {i : [] for i in range(5)}
            for j, loader in enumerate(class_samples):
                for data in loader:
                    img_name = data['image_name']
                    x = data['image']
                    img_label = data['label']

                    x = x.to(device)
                    feats = self.model.feat_extract(x)
                    cls_feats[j].append(feats)
                cls_feats[j] = torch.cat(cls_feats[j], axis=0)
                self.prototypes[j] = self.select_prototype(cls_feats[j])

    def select_prototype(self, feats):
        n = feats.size(0)
        density = torch.zeros(n)
        uniqueness = torch.zeros(n)
        similar_matrix, S_c = self.get_similar_mat(feats)
        for i in range(n):
            rowcol = torch.cat([similar_matrix[i,:], similar_matrix[:,i]])
            rowcol = rowcol[(rowcol != 0) & (rowcol != 1)] - S_c
            density[i] = rowcol[rowcol >= 0].shape[0]
            density[i] -= rowcol[rowcol < 0].shape[0]

        val, indices = torch.topk(density, n)
        for j, idx in enumerate(indices):         
            rowcol = torch.cat([similar_matrix[:,j], similar_matrix[j,:]])
            rowcol = torch.cat([rowcol[:j],rowcol[j+1:]])
            rowcol = rowcol[rowcol != 0]    
            if j == 0:
                uniqueness[idx] = torch.min(rowcol)
            else:          
                tmp_vals, tmp_indices = torch.topk(rowcol, rowcol.shape[0])
                tmp_vals = tmp_vals[1:]
                tmp_indices = tmp_indices[1:]

                for k in tmp_indices:
                    if density[idx] < density[k]:
                        uniqueness[idx] = rowcol[k]
                        break
        uniqueness_val, uniqueness_idx = torch.topk(uniqueness,uniqueness.shape[0],largest=False) 
        tmp_idx = (uniqueness_val < 0.95).nonzero().squeeze(1)[:self.sample_size]
        final_idx = uniqueness_idx[tmp_idx]
        print("Prototype uniquness =", uniqueness[final_idx])
        print("Prototype density : ",density[final_idx])

        return feats[final_idx]

    def get_similar_mat(self, feats):
        n = feats.size(0)
        mat = torch.zeros(n,n).cuda()
        for i in range(feats.size(0)):
            for j in range(i, feats.size(0)):
                v1 = feats[i,:]
                v2 = feats[j,:]
                val = torch.sum(v1 * v2) / (torch.sum(v1 **2) * torch.sum(v2 **2)) ** 0.5
                mat[i][j] = val
        mat_cp = mat.clone().reshape(-1)
        mat_cp = mat_cp[mat_cp != 0]
        mat_cp_without = mat_cp[mat_cp != 1]
        
        assert mat_cp.shape[0] - mat_cp_without.shape[0] == n
        S_c, _ = torch.topk(mat_cp_without, int(n * (n -1) / 2 * 0.6))
        S_c = S_c[-1]
        return mat, S_c

        
