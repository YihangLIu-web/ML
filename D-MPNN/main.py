import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet
from model import SimpleEGNN
from Dataset_process import Graph_list, MolecularDataset
from Trainer import Trainer
from config import cfg

if __name__ == '__main__':

    train_graphs_generator = Graph_list(cfg.paths.TRAIN_DIR, cfg.paths.TRAIN_XLSX)
    test_graphs_generator = Graph_list(cfg.paths.VAL_DIR, cfg.paths.VAL_XLSX)

    train_graphs_list = train_graphs_generator.build_graph_list()
    test_graphs_list = test_graphs_generator.build_graph_list()

    train_dateset = MolecularDataset(train_graphs_list)
    test_dateset = MolecularDataset(test_graphs_list)

    # train_loader = DataLoader(train_dateset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    # val_loader = DataLoader(test_dateset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_graphs_list, batch_size=cfg.data.batch_size, shuffle=False)
    val_loader = DataLoader(test_graphs_list, batch_size=cfg.data.batch_size, shuffle=False)

    model = SimpleEGNN(
        hidden_channels=cfg.model.hidden_channels,
        num_filters=cfg.model.num_filters,
        num_interactions=cfg.model.num_interactions,
        num_gaussians=cfg.model.num_gaussians,
        cutoff=cfg.model.cutoff,
        max_num_neighbors=cfg.model.max_num_neighbors,
    ).to(cfg.train.device)

    trainer = Trainer(
        model=model,
        train_dataloder=train_loader,
        val_dataloder=val_loader,
        loss_fn=torch.nn.MSELoss(reduction='mean')
    )
    trainer.train()
