import torch
from torch.utils.data import Dataset
import os


class FeatureBagDataset(Dataset):
    def __init__(self, pt_files, clin_data_dict=None, is_train=True, strict_clin_match=True):
        """
        :param pt_files: .pt 文件路径列表
        :param clin_data_dict: {filename: normalized_tensor} 外部传入的归一化数据字典
        :param is_train: 是否为训练模式
        :param strict_clin_match: 若提供了 clin_data_dict，但当前样本缺失对应 key，则直接报错
        """
        self.files = pt_files
        self.clin_data_dict = clin_data_dict
        self.is_train = is_train
        self.strict_clin_match = strict_clin_match

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        fname = os.path.basename(path)

        data = torch.load(path, map_location='cpu')

        if 'img_feats' not in data:
            raise KeyError(f"Missing 'img_feats' in {path}")
        if 'label' not in data:
            raise KeyError(f"Missing 'label' in {path}")

        img_feats = data['img_feats']
        label = data['label']

        if img_feats.dim() > 2:
            img_feats = img_feats.view(img_feats.size(0), -1)

        if img_feats.size(0) == 0:
            raise ValueError(f"Empty img_feats in {path}")

        # patch_labels
        if 'patch_labels' in data:
            patch_labels = data['patch_labels']
            if not isinstance(patch_labels, torch.Tensor):
                patch_labels = torch.tensor(patch_labels, dtype=torch.float32)
            else:
                patch_labels = patch_labels.float()
        else:
            patch_labels = torch.zeros(img_feats.size(0), dtype=torch.float32)

        if patch_labels.numel() != img_feats.size(0):
            raise ValueError(
                f"patch_labels length mismatch in {path}: "
                f"{patch_labels.numel()} vs {img_feats.size(0)}"
            )

        # 临床特征：优先使用外部 fold 内标准化结果
        if self.clin_data_dict is not None:
            if fname in self.clin_data_dict:
                clin_feats = self.clin_data_dict[fname]
            else:
                if self.strict_clin_match:
                    raise KeyError(
                        f"Clinical features for '{fname}' not found in clin_data_dict. "
                        f"This usually means filename mismatch between pt files and fold clinical dict."
                    )
                if 'clin_feats' not in data:
                    raise KeyError(f"Missing fallback 'clin_feats' in {path}")
                clin_feats = data['clin_feats']
        else:
            if 'clin_feats' not in data:
                raise KeyError(f"Missing 'clin_feats' in {path}")
            clin_feats = data['clin_feats']

        if not isinstance(clin_feats, torch.Tensor):
            clin_feats = torch.tensor(clin_feats, dtype=torch.float32)
        else:
            clin_feats = clin_feats.float()

        return {
            'img_feats': img_feats.float(),
            'clin_feats': clin_feats,
            'label': torch.tensor(label, dtype=torch.long),
            'patch_labels': patch_labels,
            'file_name': fname
        }