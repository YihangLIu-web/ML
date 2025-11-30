from pathlib import Path
import pandas as pd
import torch
from ase.data import atomic_numbers
from ase.io import read as ase_read
from torch_geometric.data import Data, Dataset
from config import cfg


class Graph_list:
    """生成Graph_list，每一个list的迭代对象是一个PYG的Data对象，返回的list可以送入Dataset管线"""

    def __init__(self, data_dir, excel_path):
        self.data_dir = data_dir
        self.excel_path = excel_path
        self.energy_dict = {}
        self.graph_list = []
        self.load_energy_table()

    def load_energy_table(self):
        """读取 Excel：第1列 = 名称，第2列 = 能量"""
        df = pd.read_excel(self.excel_path)
        name_col = df.columns[0]
        energy_col = df.columns[1]
        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            energy = float(row[energy_col]) * cfg.data.energy_scale
            self.energy_dict[name] = energy

    def xsd_to_data(self, xsd_path, energy):
        """用 ASE 读取 xsd，转成 PyG 的 Data(z, pos, y)"""
        atoms = ase_read(xsd_path)
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()  # Å

        z = torch.tensor([atomic_numbers[s] for s in symbols], dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float32)
        y = torch.tensor([energy], dtype=torch.float32)
        return Data(z=z, pos=pos, y=y)

    def build_graph_list(self):
        """遍历目录下所有 xsd，按 Excel 匹配能量，生成 Data 列表"""
        energy_table = self.energy_dict
        graph_list = self.graph_list
        struct_dir = Path(self.data_dir)
        for xsd_file in struct_dir.glob("*.xsd"):
            name = xsd_file.stem  # 去掉后缀
            if name not in energy_table:
                print(f"[WARN] {name} 在 {self.excel_path} 里找不到能量，跳过")
                continue
            energy = energy_table[name]
            print(energy)
            data = self.xsd_to_data(str(xsd_file), energy)
            print(f'data is {data}')
            print(f"data.z is {data.z}")
            data.name = name
            graph_list.append(data)
        print(f"{struct_dir}: 读取到 {len(graph_list)} 个分子")
        return graph_list


Graph_generator = Graph_list(cfg.paths.TRAIN_DIR, cfg.paths.TRAIN_XLSX)
for i, j in Graph_generator.energy_dict.items():
    print(f'name is {i} and energy is {j}')

train_graphs = Graph_generator.build_graph_list()


class MolecularDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]
